import logging

import gensim
from sklearn.decomposition import PCA

from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.workload import Query

from .boo import BagOfOperators


class WorkloadEmbedder(object):
    def __init__(self, query_texts, representation_size, database_connector, columns=None, retrieve_plans=False):
        self.STOPTOKENS = [
            "as",
            "and",
            "or",
            "min",
            "max",
            "avg",
            "join",
            "on",
            "substr",
            "between",
            "count",
            "sum",
            "case",
            "then",
            "when",
            "end",
            "else",
            "select",
            "from",
            "where",
            "by",
            "cast",
            "in",
        ]
        self.INDEXES_SIMULATED_IN_PARALLEL = 1000
        self.query_texts = query_texts
        self.representation_size = representation_size


        self.true_representation_size = representation_size  + 10 # dim pca(V) = 10 
        self.database_connector = database_connector
        self.plans = None
        self.columns = columns

        if retrieve_plans:
            # for q_idx in range(1):
                cost_evaluation = CostEvaluation(self.database_connector)
                # [without indexes], [with indexes]
                self.plans = ([], [])
                for query_idx, query_texts_per_query_class in enumerate(query_texts):
                    for query_text in query_texts_per_query_class:
                        if isinstance(query_text,list):
                            query_text = query_text[0]
                        query = Query(query_idx, query_text)
                        plan = self.database_connector.get_plan(query)
                        self.plans[0].append(plan)

                for n, n_column_combinations in enumerate(self.columns):
                    logging.critical(f"Creating all indexes of width {n+1}.")

                    created_indexes = 0
                    while created_indexes < len(n_column_combinations):
                        potential_indexes = []
                        for i in range(self.INDEXES_SIMULATED_IN_PARALLEL):
                            potential_index = Index(n_column_combinations[created_indexes])
                            cost_evaluation.what_if.simulate_index(potential_index, True)
                            potential_indexes.append(potential_index)
                            created_indexes += 1
                            if created_indexes == len(n_column_combinations):
                                break

                        for query_idx, query_texts_per_query_class in enumerate(query_texts):
                            for query_text in  query_texts_per_query_class:
                                # query_text = query_texts_per_query_class[q_idx]
                                if isinstance(query_text,list):
                                    query_text = query_text[0]
                                query = Query(query_idx, query_text)
                                plan = self.database_connector.get_plan(query)
                                self.plans[1].append(plan)

                        for potential_index in potential_indexes:
                            cost_evaluation.what_if.drop_simulated_index(potential_index)

                        logging.critical(f"Finished checking {created_indexes} indexes of width {n+1}.")

        self.database_connector = None
        

    def get_embeddings(self, workload):
        raise NotImplementedError


class SQLWorkloadEmbedder(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, query_texts, representation_size, database_connector)

        tagged_queries = []

        for query_idx, query_texts_per_query_class in enumerate(query_texts):
            query_text = query_texts_per_query_class[0]
            tokens = gensim.utils.simple_preprocess(query_text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            tagged_queries.append(gensim.models.doc2vec.TaggedDocument(tokens, [query_idx]))

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=2, epochs=500)
        self.model.build_vocab(tagged_queries)
        logger = logging.getLogger("gensim.models.base_any2vec")
        logger.setLevel(logging.CRITICAL)
        self.model.train(tagged_queries, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logger.setLevel(logging.INFO)

    def get_embeddings(self, workload):
        embeddings = []

        for query in workload.queries:
            tokens = gensim.utils.simple_preprocess(query.text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            vector = self.model.infer_vector(tokens)

            embeddings.append(vector)

        return embeddings


class SQLWorkloadLSI(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, query_texts, representation_size, database_connector)

        self.processed_queries = []

        for query_idx, query_texts_per_query_class in enumerate(query_texts):
            query_text = query_texts_per_query_class[0]
            tokens = gensim.utils.simple_preprocess(query_text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            self.processed_queries.append(tokens)

        self.dictionary = gensim.corpora.Dictionary(self.processed_queries)
        self.bow_corpus = [self.dictionary.doc2bow(query) for query in self.processed_queries]
        self.lsi_bow = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )

    def get_embeddings(self, workload):
        embeddings = []

        for query in workload.queries:
            tokens = gensim.utils.simple_preprocess(query.text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            bow = self.dictionary.doc2bow(tokens)
            result = self.lsi_bow[bow]
            result = [x[1] for x in result]

            embeddings.append(result)

        return embeddings


class PlanEmbedder(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns, without_indexes=False):
        WorkloadEmbedder.__init__(
            self, query_texts, representation_size, database_connector, columns, retrieve_plans=True
        )

        self.plan_embedding_cache = {}
        self.value_embedding_cache = {}
        self.marknum=0

        self.relevant_operators = []
        self.relevant_operators_wo_indexes = []
        self.relevant_operators_with_indexes = []

        self.boo_creator = BagOfOperators()
        for plan in self.plans[0]:
            boo = self.boo_creator.boo_from_plan(plan)
            self.relevant_operators.append(boo)
            self.relevant_operators_wo_indexes.append(boo)

        if without_indexes is False:
            for plan in self.plans[1]:
                boo = self.boo_creator.boo_from_plan(plan)
                self.relevant_operators.append(boo)
                self.relevant_operators_with_indexes.append(boo)

        # Deleting the plans to avoid costly copying later.
        self.plans = None

        self.dictionary = gensim.corpora.Dictionary(self.relevant_operators)
        logging.warning(f"Dictionary has {len(self.dictionary)} entries.")
        self.bow_corpus = [self.dictionary.doc2bow(query) for query in self.relevant_operators]

        self._create_model()

        # Deleting the bow_corpus to avoid costly copying later.
        self.bow_corpus = None

    def _create_model(self):
        raise NotImplementedError

    def _infer(self, bow, boo):
        raise NotImplementedError

    def get_embeddings(self, plans):
        return self.get_embeddings_boo(plans)
        
    
    def get_embeddings_boo(self, plans):
        """
        该方法用于获取计划的嵌入表示，并结合PCA处理的值向量。

        参数:
        plans (list): 计划列表

        返回:
        list: 包含嵌入向量的列表
        """
        embeddings = []
        import numpy as np
        from sklearn.decomposition import PCA
        # 用于存储值向量的临时列表
        temp = []
        # 初始化PCA对象，指定主成分数量为10
        pca = PCA(n_components=10)
        ###
        # 记录异常发生的次数
        self.err = 0
        # 记录最近一次异常对象
        self.e = 0
        # 遍历每个计划
        for plan in plans:
            # 生成计划的缓存键
            cache_key = str(plan)
            # 检查缓存中是否存在该计划的嵌入向量
            if cache_key not in self.plan_embedding_cache:
                # 从计划中提取操作符袋
                boo = self.boo_creator.boo_from_plan(plan)
                # 将操作符袋转换为词袋表示
                bow = self.dictionary.doc2bow(boo)

                # 初始化值向量，长度为字典的大小
                value_vec = [0] * len(self.dictionary)
                try:
                    # 从计划中提取值
                    value = self.boo_creator.value_from_plan(plan)
                    # 遍历每个值
                    for i, vv in enumerate(value):
                        # 检查值是否不为None
                        if vv != None:
                            # 将操作符转换为词袋表示
                            bow_i = self.dictionary.doc2bow([boo[i]])
                            # 检查词袋表示是否为空
                            if len(bow_i)==0:
                                break
                            # 将值累加到值向量的相应位置
                            value_vec[bow_i[0][0]] += vv
                except Exception as e:
                    # 异常发生时，错误计数加1
                    self.err = self.err+1
                    # 记录异常对象
                    self.e = e

                # 推断计划的嵌入向量
                vector = self._infer(bow, boo)

                # 将值向量添加到临时列表中
                temp.append(value_vec)

                # 将计划的嵌入向量存入缓存
                self.plan_embedding_cache[cache_key] = vector
                # 将值向量存入缓存
                self.value_embedding_cache[cache_key] = value_vec
            else:
                # 从缓存中获取计划的嵌入向量
                vector = self.plan_embedding_cache[cache_key]
                # 从缓存中获取值向量
                value_vec = self.value_embedding_cache[cache_key]

                # 将值向量添加到临时列表中
                temp.append(value_vec)

            # 将计划的嵌入向量添加到结果列表中
            embeddings.append(vector)
        # 使用临时列表中的值向量拟合PCA模型
        pca.fit(temp)
        # 对临时列表中的值向量进行PCA转换
        val_pca = pca.transform(temp)

        # 初始化索引
        ind = 0
        # 存储最终结果的列表
        res = []
        # 遍历每个嵌入向量
        for emb in embeddings:
            # 将嵌入向量和PCA转换后的值向量相加
            emb = (emb+val_pca[ind].tolist())
            # 索引加1
            ind = ind+1
            # 将结果添加到最终结果列表中
            res.append(emb)
        return res

    
    def get_embeddings_noboo(self, plans):
        embeddings = []

        for plan in plans:
            cache_key = str(plan)
            if cache_key not in self.plan_embedding_cache:
                boo = self.boo_creator.boo_from_plan(plan)
                bow = self.dictionary.doc2bow(boo)

                vector = self._infer(bow, boo)

                self.plan_embedding_cache[cache_key] = vector
            else:
                vector = self.plan_embedding_cache[cache_key]

            embeddings.append(vector)

        return embeddings


class PlanEmbedderPCA(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns)

    def _to_full_corpus(self, corpus):
        new_corpus = []
        for bow in corpus:
            new_bow = [0 for i in range(len(self.dictionary))]
            for elem in bow:
                index, value = elem
                new_bow[index] = value
            new_corpus.append(new_bow)

        return new_corpus

    def _create_model(self):
        new_corpus = self._to_full_corpus(self.bow_corpus)

        self.pca = PCA(n_components=self.representation_size)
        self.pca.fit(new_corpus)

        assert (
            sum(self.pca.explained_variance_ratio_) > 0.8
        ), f"Explained variance must be larger than 80% (is {sum(self.pca.explained_variance_ratio_)})"

    def _infer(self, bow, boo):
        new_bow = self._to_full_corpus([bow])

        return self.pca.transform(new_bow)


class PlanEmbedderDoc2Vec(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns, without_indexes=False):
        self.without_indexes = without_indexes

        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns, without_indexes)

    def _create_model(self):
        tagged_plans = []
        for plan_idx, operators in enumerate(self.relevant_operators):
            tagged_plans.append(gensim.models.doc2vec.TaggedDocument(operators, [plan_idx]))

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=2, epochs=1000)
        self.model.build_vocab(tagged_plans)
        logger = logging.getLogger("gensim.models.base_any2vec")
        logger.setLevel(logging.CRITICAL)
        self.model.train(tagged_plans, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logger.setLevel(logging.INFO)

    def _infer(self, bow, boo):
        vector = self.model.infer_vector(boo)

        return vector


class PlanEmbedderDoc2VecWithoutIndexes(PlanEmbedderDoc2Vec):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedderDoc2Vec.__init__(
            self, query_texts, representation_size, database_connector, columns, without_indexes=True
        )


class PlanEmbedderBOW(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns)

    def _create_model(self):
        assert self.representation_size == len(self.dictionary), f"{self.representation_size} == {len(self.dictionary)}"

    def _to_full_bow(self, bow):
        new_bow = [0 for i in range(len(self.dictionary))]
        for elem in bow:
            index, value = elem
            new_bow[index] = value

        return new_bow

    def _infer(self, bow, boo):
        return self._to_full_bow(bow)


class PlanEmbedderLSIBOW(PlanEmbedder):
    """
    PlanEmbedderLSIBOW类继承自PlanEmbedder，用于基于LSI（Latent Semantic Indexing）模型对查询计划进行嵌入表示。

    参数:
    query_texts (list): 查询文本列表
    representation_size (int): 嵌入表示的维度
    database_connector (object): 数据库连接器对象
    columns (list): 列名列表
    without_indexes (bool): 是否不考虑索引，默认为False
    """
    def __init__(self, query_texts, representation_size, database_connector, columns, without_indexes=False):
        # 调用父类的构造函数
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns, without_indexes)

    def _create_model(self):
        """
        创建并保存LSI模型，然后加载该模型，并验证主题数量是否与表示大小匹配。
        """
        # 创建LSI模型
        self.lsi_bow = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )
        # 保存LSI模型到文件
        self.lsi_bow.save('tpcds_lsi.model')
        # 从文件中加载LSI模型
        self.lsi_bow = gensim.models.LsiModel.load('tpcds_lsi.model')
        print(f"len(self.lsi_bow.get_topics()) :  {len(self.lsi_bow.get_topics())}")

        # 断言检查LSI模型的主题数量是否与表示大小一致
        assert (
            len(self.lsi_bow.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_bow.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, boo):
        """
        根据LSI模型推断查询计划的嵌入向量。

        参数:
        bow (list): 词袋表示
        boo (list): 操作符袋表示

        返回:
        list: 嵌入向量
        """
        # 使用LSI模型对词袋表示进行转换
        result = self.lsi_bow[bow]

        # 如果结果的长度等于表示大小，则直接提取值作为向量
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            # 否则，初始化一个全零向量，并将结果中的值填充到相应位置
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        # 断言检查向量的长度是否与表示大小一致
        assert len(vector) == self.representation_size

        return vector



class PlanEmbedderLSIBOWWithoutIndexes(PlanEmbedderLSIBOW):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedderLSIBOW.__init__(
            self, query_texts, representation_size, database_connector, columns, without_indexes=True
        )


class PlanEmbedderLSITFIDF(PlanEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, query_texts, representation_size, database_connector, columns)

    def _create_model(self):
        self.tfidf = gensim.models.TfidfModel(self.bow_corpus, normalize=True)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        self.lsi_tfidf = gensim.models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
            len(self.lsi_tfidf.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_tfidf.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, boo):
        result = self.lsi_tfidf[self.tfidf[bow]]

        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector
