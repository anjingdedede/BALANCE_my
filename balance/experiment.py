import copy
import datetime
import gzip
import importlib
import json
import logging
import os
import pickle
import random
import subprocess

import gym
import numpy as np

from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from index_selection_evaluation.selection.algorithms.extend_algorithm import ExtendAlgorithm
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector

from . import utils
from .configuration_parser import ConfigurationParser
from .schema import Schema
from .workload_generator import WorkloadGenerator

class Experiment(object):
    def __init__(self, configuration_file,aa=None,id=None):
        self._init_times()

        cp = ConfigurationParser(configuration_file)
        self.config = cp.config
        if aa!=None:
            self.config["id"] = "TPCDS_depart_unknow_"+aa
            self.config["workload"]["unknown_queries"] = int(aa)
        if id!=None:
            self.config["id"] = id
        self._set_sb_version_specific_methods()

        self.id = self.config["id"]
        self.cmp_runtime = datetime.timedelta(0)
        self.dataset_size = None
        self.model = None
        self.Smodel_1 = None
        self.Smodel_2 = None
        self.Smodel_3 = None
        self.Smodel_4 = None
        self.Smodel_5 = None
        self.Smodel_6 = None
        self.rnd = random.Random()
        self.rnd.seed(self.config["random_seed"])

        self.comparison_performances = {
            "test": {"Extend": [], "DB2Adv": []},
            "validation": {"Extend": [], "DB2Adv": []},
        }
        self.comparison_indexes = {"Extend": set(), "DB2Adv": set()}

        self.number_of_features = None
        self.number_of_actions = None
        self.evaluated_workloads_strs = []

        self.EXPERIMENT_RESULT_PATH = self.config["result_path"]
        self._create_experiment_folder()
        

    def prepare(self):
        """
        准备实验所需的各项资源，包括模式、工作负载生成器等。
        """
        # 初始化 Schema 对象，用于描述数据库的模式
        self.schema = Schema(
            # 工作负载的基准测试名称
            self.config["workload"]["benchmark"],
            # 工作负载的缩放因子
            self.config["workload"]["scale_factor"],
            # 数据库名称
            self.config["database"],
            # self.config["used_tables"]["names"],
            # 列过滤器配置
            self.config["column_filters"],

        )

        # 初始化工作负载生成器，用于生成实验所需的工作负载
        self.workload_generator = WorkloadGenerator(
            # 工作负载相关配置
            self.config["workload"],
            # 工作负载文件路径
            spath =  self.config["workload"]["path"],
            # 工作负载涉及的列
            workload_columns=self.schema.columns,
            # 随机种子，用于保证结果的可重复性
            random_seed=self.config["random_seed"],
            # 数据库名称
            database_name=self.schema.database_name,
            # 实验 ID
            experiment_id=self.id,
            # 是否过滤使用的列
            filter_utilized_columns=self.config["filter_utilized_columns"],
            # 实验文件夹路径
            experiment_folder_path =self.experiment_folder_path
        )
        # 为工作负载分配预算
        self._assign_budgets_to_workloads()
        # 将工作负载数据进行序列化保存
        self._pickle_workloads()

        # 获取全局可索引的列
        self.globally_indexable_columns = self.workload_generator.globally_indexable_columns

        # 根据最大索引宽度创建列的排列索引
        self.globally_indexable_columns = utils.create_column_permutation_indexes(
            # 全局可索引的列
            self.globally_indexable_columns,
            # 最大索引宽度
            self.config["max_index_width"]
        )

        # 创建单列的扁平集合
        self.single_column_flat_set = set(map(lambda x: x[0], self.globally_indexable_columns[0]))

        # 将全局可索引列的嵌套列表展平为一维列表
        self.globally_indexable_columns_flat = [item for sublist in self.globally_indexable_columns for item in sublist]
        # 记录日志，显示将多少候选索引输入到环境中
        logging.info(f"Feeding {len(self.globally_indexable_columns_flat)} candidates into the environments.")

        # 预测全局可索引列的索引大小
        self.action_storage_consumptions = utils.predict_index_sizes(
            # 全局可索引列的扁平列表
            self.globally_indexable_columns_flat,
            # 数据库名称
            self.schema.database_name
        )

        # 检查配置中是否包含工作负载嵌入器的配置
        if "workload_embedder" in self.config:
            # 动态导入工作负载嵌入器类
            workload_embedder_class = getattr(
                # 导入指定模块
                importlib.import_module("balance.workload_embedder"),
                # 获取指定类型的工作负载嵌入器类
                self.config["workload_embedder"]["type"]
            )
            # 初始化 Postgres 数据库连接器
            workload_embedder_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
            # 初始化工作负载嵌入器对象
            self.workload_embedder = workload_embedder_class(
                # 工作负载的查询文本
                self.workload_generator.available_query_texts,
                # 表示大小(50) - 值大小(10)
                40,
                # 数据库连接器
                workload_embedder_connector,
                # 全局可索引列
                self.globally_indexable_columns,
            )

        # 初始化多验证工作负载列表
        self.multi_validation_wl = []
        # 检查验证工作负载的数量是否大于 1
        if len(self.workload_generator.wl_validation) > 1:
            # 遍历每个验证工作负载列表
            for workloads in self.workload_generator.wl_validation:
                # 从每个验证工作负载列表中随机选择最多 7 个工作负载添加到多验证工作负载列表中
                self.multi_validation_wl.extend(self.rnd.sample(workloads, min(7, len(workloads))))



       

    def _assign_budgets_to_workloads(self):
        """
        为工作负载生成器中的测试和验证工作负载分配预算。

        此方法会遍历工作负载生成器中的测试和验证工作负载列表，
        为每个工作负载随机分配一个预算值，预算值从配置文件中指定的列表中选择。
        """
        # 遍历测试工作负载列表
        for workload_list in self.workload_generator.wl_testing:
            # 遍历每个测试工作负载
            for workload in workload_list:
                # 为每个测试工作负载随机分配一个预算值，预算值从配置文件中的 validation_and_testing 列表中选择
                workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])

        # # 遍历验证工作负载列表
        # for workload_list in self.workload_generator.wl_validation:
        #     # 遍历每个验证工作负载
        #     for workload in workload_list:
        #         # 为每个验证工作负载随机分配一个预算值，预算值从配置文件中的 validation_and_testing 列表中选择
        #         workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])
        # 为验证工作负载分配不同大小的预算
        result_workloads = []
        for workload_list in self.workload_generator.wl_validation:
            for workload in workload_list:
                _tmp_workloads = []
                for _budget in self.config["budgets"]["validation_and_testing"]:
                    # 复制当前工作负载
                    _workload = copy.deepcopy(workload)
                    # 为复制的工作负载分配当前预算
                    _workload.budget = _budget
                    _tmp_workloads.append(_workload)
                # 将临时工作负载列表添加到结果工作负载列表中
                result_workloads.extend(_tmp_workloads)
        # 更新验证工作负载列表
        self.workload_generator.wl_validation = [copy.deepcopy(result_workloads)]




    def _pickle_workloads(self):
        """
        将工作负载生成器中的测试、验证和训练工作负载数据进行序列化，并保存为 pickle 文件。
        """
        # 定义一个字符串变量，用于构建文件名
        st = "1"
        # 打开测试工作负载的 pickle 文件，以二进制写入模式
        with open(f"{self.experiment_folder_path}/testing_workloads{st}.pickle", "wb") as handle:
            # 使用 pickle 模块将测试工作负载数据以最高协议序列化并写入文件
            pickle.dump(self.workload_generator.wl_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # 打开验证工作负载的 pickle 文件，以二进制写入模式
        with open(f"{self.experiment_folder_path}/validation_workloads{st}.pickle", "wb") as handle:
            # 使用 pickle 模块将验证工作负载数据以最高协议序列化并写入文件
            pickle.dump(self.workload_generator.wl_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # 打开训练工作负载的 pickle 文件，以二进制写入模式
        with open(f"{self.experiment_folder_path}/train_workloads{st}.pickle", "wb") as handle:
            # 使用 pickle 模块将训练工作负载数据以最高协议序列化并写入文件
            pickle.dump(self.workload_generator.wl_training, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def finishmy(self):
        """
        完成实验，进行模型评估、报告生成和日志记录。
        """
        # 记录实验结束时间
        self.end_time = datetime.datetime.now()

        # 设置模型为非训练模式
        self.model.training = False
        # 禁用模型环境的奖励归一化
        self.model.env.norm_reward = False
        # 设置模型环境为非训练模式
        self.model.env.training = False

        # 设置数据库名称
        self.schema.database_name = self.config["database"]

        # 测试当前模型并获取性能
        self.test_fm = self.test_model(self.model)[0]
        # 验证当前模型并获取性能
        self.vali_fm = self.validate_model(self.model)[0]

        # 加载移动平均模型
        self.moving_average_model = self.model_type.load(f"{self.experiment_folder_path}/moving_average_model.zip")
        # 设置移动平均模型为非训练模式
        self.moving_average_model.training = False
        # 测试移动平均模型并获取性能
        self.test_ma = self.test_model(self.moving_average_model)[0]
        # 验证移动平均模型并获取性能
        self.vali_ma = self.validate_model(self.moving_average_model)[0]

        # 如果存在多验证工作负载
        if len(self.multi_validation_wl) > 0:
            # 加载多验证移动平均模型
            self.moving_average_model_mv = self.model_type.load(
                f"{self.experiment_folder_path}/moving_average_model_mv.zip"
            )
            # 设置多验证移动平均模型为非训练模式
            self.moving_average_model_mv.training = False
            # 测试多验证移动平均模型并获取性能
            self.test_ma_mv = self.test_model(self.moving_average_model_mv)[0]
            # 验证多验证移动平均模型并获取性能
            self.vali_ma_mv = self.validate_model(self.moving_average_model_mv)[0]

        # 加载另一个移动平均模型
        self.moving_average_model_3 = self.model_type.load(f"{self.experiment_folder_path}/moving_average_model_3.zip")
        # 设置另一个移动平均模型为非训练模式
        self.moving_average_model_3.training = False
        # 测试另一个移动平均模型并获取性能
        self.test_ma_3 = self.test_model(self.moving_average_model_3)[0]
        # 验证另一个移动平均模型并获取性能
        self.vali_ma_3 = self.validate_model(self.moving_average_model_3)[0]

        # 如果存在多验证工作负载
        if len(self.multi_validation_wl) > 0:
            # 加载多验证另一个移动平均模型
            self.moving_average_model_3_mv = self.model_type.load(
                f"{self.experiment_folder_path}/moving_average_model_3_mv.zip"
            )
            # 设置多验证另一个移动平均模型为非训练模式
            self.moving_average_model_3_mv.training = False
            # 测试多验证另一个移动平均模型并获取性能
            self.test_ma_3_mv = self.test_model(self.moving_average_model_3_mv)[0]
            # 验证多验证另一个移动平均模型并获取性能
            self.vali_ma_3_mv = self.validate_model(self.moving_average_model_3_mv)[0]

        # 加载最佳平均奖励模型
        self.best_mean_reward_model = self.model_type.load(f"{self.experiment_folder_path}/best_mean_reward_model.zip")
        # 设置最佳平均奖励模型为非训练模式
        self.best_mean_reward_model.training = False
        # 测试最佳平均奖励模型并获取性能
        self.test_bm = self.test_model(self.best_mean_reward_model)[0]
        # 验证最佳平均奖励模型并获取性能
        self.vali_bm = self.validate_model(self.best_mean_reward_model)[0]

        # 如果存在多验证工作负载
        if len(self.multi_validation_wl) > 0:
            # 加载多验证最佳平均奖励模型
            self.best_mean_reward_model_mv = self.model_type.load(
                f"{self.experiment_folder_path}/best_mean_reward_model_mv.zip"
            )
            # 设置多验证最佳平均奖励模型为非训练模式
            self.best_mean_reward_model_mv.training = False
            # 测试多验证最佳平均奖励模型并获取性能
            self.test_bm_mv = self.test_model(self.best_mean_reward_model_mv)[0]
            # 验证多验证最佳平均奖励模型并获取性能
            self.vali_bm_mv = self.validate_model(self.best_mean_reward_model_mv)[0]

        # 生成实验报告
        self._write_report( self.config["database"])

        # 记录关键日志信息，包含实验 ID 和报告路径
        logging.critical(
            (
                f"Finished training of ID {self.id}. Report can be found at "
                f"./{self.experiment_folder_path}/report_ID_{self.id}.txt"
            )
        )


    def _get_wl_budgets_from_model_perfs(self, perfs):
        """
        从模型性能数据中提取每个工作负载的预算。

        参数:
        perfs (list): 包含模型性能信息的列表，每个元素是一个字典，
                      其中包含 'evaluated_workload' 和 'available_budget' 键。

        返回:
        list: 包含每个工作负载预算的列表。
        """
        # 初始化一个空列表，用于存储每个工作负载的预算
        wl_budgets = []
        # 遍历每个模型性能数据
        for perf in perfs:
            # 断言评估工作负载的预算和可用预算相等，如果不相等则抛出异常
            assert perf["evaluated_workload"].budget == perf["available_budget"], "Budget mismatch!"
            # 将评估工作负载的预算添加到列表中
            wl_budgets.append(perf["evaluated_workload"].budget)
        # 返回包含所有工作负载预算的列表
        return wl_budgets


    def start_learning(self):
        self.training_start_time = datetime.datetime.now()

    def set_model(self, model):
        self.model = model

    def finish_learning(self, training_env, moving_average_model_step, best_mean_model_step):
        """
        完成学习过程，记录训练结束时间、保存模型和环境，统计评估的回合数、总步数、缓存命中信息等。

        参数:
        training_env (object): 训练环境对象。
        moving_average_model_step (int): 移动平均模型的步数。
        best_mean_model_step (int): 最佳平均模型的步数。
        """
        # 记录训练结束时间
        self.training_end_time = datetime.datetime.now()

        # 记录移动平均模型的步数
        self.moving_average_validation_model_at_step = moving_average_model_step
        # 记录最佳平均模型的步数
        self.best_mean_model_step = best_mean_model_step

        # 保存最终模型到实验文件夹
        self.model.save(f"{self.experiment_folder_path}/final_model")
        # 保存训练环境的归一化参数到实验文件夹
        training_env.save(f"{self.experiment_folder_path}/vec_normalize.pkl")

        # 初始化评估的回合数为 0
        self.evaluated_episodes = 0
        # 遍历训练环境中每个环境的重置次数
        for number_of_resets in training_env.get_attr("number_of_resets"):
            # 累加每个环境的重置次数到评估的回合数
            self.evaluated_episodes += number_of_resets

        # 初始化总步数为 0
        self.total_steps_taken = 0
        # 遍历训练环境中每个环境的总步数
        for total_number_of_steps in training_env.get_attr("total_number_of_steps"):
            # 累加每个环境的总步数到总步数
            self.total_steps_taken += total_number_of_steps

        # 初始化缓存命中次数为 0
        self.cache_hits = 0
        # 初始化成本请求次数为 0
        self.cost_requests = 0
        # 初始化成本计算时间为 0
        self.costing_time = datetime.timedelta(0)
        # 遍历训练环境中每个环境的成本评估缓存信息
        for cache_info in training_env.env_method("get_cost_eval_cache_info"):
            # 累加每个环境的缓存命中次数到缓存命中次数
            self.cache_hits += cache_info[1]
            # 累加每个环境的成本请求次数到成本请求次数
            self.cost_requests += cache_info[0]
            # 累加每个环境的成本计算时间到成本计算时间
            self.costing_time += cache_info[2]
        # 计算平均成本计算时间
        self.costing_time /= self.config["parallel_environments"]

        # 计算缓存命中率
        self.cache_hit_ratio = self.cache_hits / self.cost_requests * 100

        # 如果配置中要求对成本估计缓存进行序列化
        if self.config["pickle_cost_estimation_caches"]:
            # 初始化缓存列表
            caches = []
            # 遍历训练环境中每个环境的成本评估缓存
            for cache in training_env.env_method("get_cost_eval_cache"):
                # 将每个环境的成本评估缓存添加到缓存列表
                caches.append(cache)
            # 初始化合并后的缓存字典
            combined_caches = {}
            # 遍历缓存列表中的每个缓存
            for cache in caches:
                # 合并缓存到合并后的缓存字典
                combined_caches = {**combined_caches, **cache}
            # 以二进制写入模式打开实验文件夹中的缓存文件
            with gzip.open(f"{self.experiment_folder_path}/caches.pickle.gzip", "wb") as handle:
                # 将合并后的缓存字典以最高协议序列化到文件
                pickle.dump(combined_caches, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def _init_times(self):
        self.start_time = datetime.datetime.now()

        self.end_time = None
        self.training_start_time = None
        self.training_end_time = None

    def _create_experiment_folder(self):
        # 断言实验结果文件夹存在，若不存在则抛出异常
        assert os.path.isdir(
            self.EXPERIMENT_RESULT_PATH
        ), f"Folder for experiment results should exist at: ./{self.EXPERIMENT_RESULT_PATH}"

        # 构建实验文件夹的路径
        self.experiment_folder_path = f"{self.EXPERIMENT_RESULT_PATH}/ID_{self.id}"
        # 导入shutil模块，用于文件和文件夹操作
        import shutil
        # 检查实验文件夹是否已经存在
        if(os.path.isdir(self.experiment_folder_path) == True):
            # 若存在，则递归删除该文件夹，忽略可能出现的错误
            shutil.rmtree(self.experiment_folder_path, ignore_errors=True)
        # 创建新的实验文件夹
        os.mkdir(self.experiment_folder_path)


    def _write_report(self,dbname):
        with open(f"{self.experiment_folder_path}/report_ID_{self.id}_{dbname}.txt", "w") as f:
            f.write(f"##### Report for Experiment with ID: {self.id} #####\n")
            f.write(f"Description: {self.config['description']}\n")
            f.write("\n")

            f.write(f"Start:                         {self.start_time}\n")
            f.write(f"End:                           {self.start_time}\n")
            f.write(f"Duration:                      {self.end_time - self.start_time}\n")
            f.write("\n")
            f.write(f"Start Training:                {self.training_start_time}\n")
            f.write(f"End Training:                  {self.training_end_time}\n")
            f.write(f"Duration Training:             {self.training_end_time - self.training_start_time}\n")
            f.write(f"Moving Average model at step:  {self.moving_average_validation_model_at_step}\n")
            f.write(f"Mean reward model at step:     {self.best_mean_model_step}\n")
            f.write(f"Git Hash:                      {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}\n")
            f.write(f"Number of features:            {self.number_of_features}\n")
            f.write(f"Number of actions:             {self.number_of_actions}\n")
            f.write("\n")
            if self.config["workload"]["unknown_queries"] > 0:
                f.write(f"Unknown Query Classes {sorted(self.workload_generator.unknown_query_classes)}\n")
                f.write(f"Known Queries: {self.workload_generator.known_query_classes}\n")
                f.write("\n")
            probabilities = len(self.config["workload"]["validation_testing"]["unknown_query_probabilities"])
            for idx, unknown_query_probability in enumerate(
                self.config["workload"]["validation_testing"]["unknown_query_probabilities"]
            ):
                f.write(f"Unknown query probability: {unknown_query_probability}:\n")
                f.write("    Final mean performance test:\n")
                test_fm_perfs, self.performance_test_final_model, self.test_fm_details = self.test_fm[idx]
                vali_fm_perfs, self.performance_vali_final_model, self.vali_fm_details = self.vali_fm[idx]

                _, self.performance_test_moving_average_model, self.test_ma_details = self.test_ma[idx]
                _, self.performance_vali_moving_average_model, self.vali_ma_details = self.vali_ma[idx]
                _, self.performance_test_moving_average_model_3, self.test_ma_details_3 = self.test_ma_3[idx]
                _, self.performance_vali_moving_average_model_3, self.vali_ma_details_3 = self.vali_ma_3[idx]
                _, self.performance_test_best_mean_reward_model, self.test_bm_details = self.test_bm[idx]
                _, self.performance_vali_best_mean_reward_model, self.vali_bm_details = self.vali_bm[idx]

                if len(self.multi_validation_wl) > 0:
                    _, self.performance_test_moving_average_model_mv, self.test_ma_details_mv = self.test_ma_mv[idx]
                    _, self.performance_vali_moving_average_model_mv, self.vali_ma_details_mv = self.vali_ma_mv[idx]
                    _, self.performance_test_moving_average_model_3_mv, self.test_ma_details_3_mv = self.test_ma_3_mv[
                        idx
                    ]
                    _, self.performance_vali_moving_average_model_3_mv, self.vali_ma_details_3_mv = self.vali_ma_3_mv[
                        idx
                    ]
                    _, self.performance_test_best_mean_reward_model_mv, self.test_bm_details_mv = self.test_bm_mv[idx]
                    _, self.performance_vali_best_mean_reward_model_mv, self.vali_bm_details_mv = self.vali_bm_mv[idx]

                self.test_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(test_fm_perfs)
                self.vali_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(vali_fm_perfs)

                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_test_final_model:.2f} ({self.test_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Moving Average model:      "
                        f"{self.performance_test_moving_average_model:.2f} ({self.test_ma_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average model (MV): "
                            f"{self.performance_test_moving_average_model_mv:.2f} ({self.test_ma_details_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Moving Average 3 model:    "
                        f"{self.performance_test_moving_average_model_3:.2f} ({self.test_ma_details_3})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average 3 mod (MV): "
                            f"{self.performance_test_moving_average_model_3_mv:.2f} ({self.test_ma_details_3_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Best mean reward model:    "
                        f"{self.performance_test_best_mean_reward_model:.2f} ({self.test_bm_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Best mean reward mod (MV): "
                            f"{self.performance_test_best_mean_reward_model_mv:.2f} ({self.test_bm_details_mv})\n"
                        )
                    )
                for key, value in self.comparison_performances["test"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value):.2f} ({value})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.test_fm_wl_budgets}\n")
                f.write("\n")
                f.write("    Final mean performance validation:\n")
                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_vali_final_model:.2f} ({self.vali_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Moving Average model:      "
                        f"{self.performance_vali_moving_average_model:.2f} ({self.vali_ma_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average model (MV): "
                            f"{self.performance_vali_moving_average_model_mv:.2f} ({self.vali_ma_details_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Moving Average 3 model:    "
                        f"{self.performance_vali_moving_average_model_3:.2f} ({self.vali_ma_details_3})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average 3 mod (MV): "
                            f"{self.performance_vali_moving_average_model_3_mv:.2f} ({self.vali_ma_details_3_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Best mean reward model:    "
                        f"{self.performance_vali_best_mean_reward_model:.2f} ({self.vali_bm_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Best mean reward mod (MV): "
                            f"{self.performance_vali_best_mean_reward_model_mv:.2f} ({self.vali_bm_details_mv})\n"
                        )
                    )
                for key, value in self.comparison_performances["validation"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value):.2f} ({value})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.vali_fm_wl_budgets}\n")
                f.write("\n")
                f.write("\n")
            f.write("Overall Test:\n")

            def final_avg(values, probabilities):
                val = 0
                for res in values:
                    val += res[1]
                return val / probabilities

            f.write(("        Final model:               " f"{final_avg(self.test_fm, probabilities):.2f}\n"))
            f.write(("        Moving Average model:      " f"{final_avg(self.test_ma, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average model (MV): " f"{final_avg(self.test_ma_mv, probabilities):.2f}\n"))
            f.write(("        Moving Average 3 model:    " f"{final_avg(self.test_ma_3, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.test_ma_3_mv, probabilities):.2f}\n"))
            f.write(("        Best mean reward model:    " f"{final_avg(self.test_bm, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Best mean reward mod (MV): " f"{final_avg(self.test_bm_mv, probabilities):.2f}\n"))
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['test']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['test']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("Overall Validation:\n")
            f.write(("        Final model:               " f"{final_avg(self.vali_fm, probabilities):.2f}\n"))
            f.write(("        Moving Average model:      " f"{final_avg(self.vali_ma, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average model (MV): " f"{final_avg(self.vali_ma_mv, probabilities):.2f}\n"))
            f.write(("        Moving Average 3 model:    " f"{final_avg(self.vali_ma_3, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.vali_ma_3_mv, probabilities):.2f}\n"))
            f.write(("        Best mean reward model:    " f"{final_avg(self.vali_bm, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Best mean reward mod (MV): " f"{final_avg(self.vali_bm_mv, probabilities):.2f}\n"))
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['validation']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['validation']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("\n")
            f.write(f"Evaluated episodes:            {self.evaluated_episodes}\n")
            f.write(f"Total steps taken:             {self.total_steps_taken}\n")
            f.write(
                (
                    f"CostEval cache hit ratio:      "
                    f"{self.cache_hit_ratio:.2f} ({self.cache_hits} of {self.cost_requests})\n"
                )
            )
            training_time = self.training_end_time - self.training_start_time
            f.write(
                f"Cost eval time (% of total):   {self.costing_time} ({self.costing_time / training_time * 100:.2f}%)\n"
            )
            # f.write(f"Cost eval time:                {self.costing_time:.2f}\n")

            f.write("\n\n")
            f.write("Used configuration:\n")
            json.dump(self.config, f)
            f.write("\n\n")
            f.write("Evaluated test workloads:\n")
            for evaluated_workload in self.evaluated_workloads_strs[: (len(self.evaluated_workloads_strs) // 2)]:
                f.write(f"{evaluated_workload}\n")
            f.write("Evaluated validation workloads:\n")
            # fmt: off
            for evaluated_workload in self.evaluated_workloads_strs[(len(self.evaluated_workloads_strs) // 2) :]:  # noqa: E203, E501
                f.write(f"{evaluated_workload}\n")
            # fmt: on
            f.write("\n\n")


    # todo: code duplication with validate_model
    def test_model(self, model):
        """
        测试模型在测试工作负载上的性能。

        参数:
        model (object): 要测试的模型。

        返回:
        tuple: 包含每个测试工作负载的模型性能和测试标识的元组。
        """
        # 初始化一个空列表，用于存储每个测试工作负载的模型性能
        model_performances = []
        # 遍历所有测试工作负载
        for test_wl in self.workload_generator.wl_testing:
            # 创建一个虚拟向量环境，用于评估模型在测试工作负载上的性能
            test_env = self.DummyVecEnv([self.make_env(0, EnvironmentType.TESTING, test_wl)])
            # 对测试环境进行向量归一化处理，仅对观测值进行归一化，不处理奖励
            test_env = self.VecNormalize(
                test_env, norm_obs=True, norm_reward=False, gamma=self.config["rl_algorithm"]["gamma"], training=False
            )

            # 如果当前模型不是默认模型，则将其环境设置为默认模型的环境
            if model != self.model:
                model.set_env(self.model.env)

            # 评估模型在测试环境上的性能
            model_performance = self._evaluate_model(model, test_env, len(test_wl))
            # 将评估结果添加到模型性能列表中
            model_performances.append(model_performance)

        # 返回模型在所有测试工作负载上的性能和测试标识
        return model_performances, "test"


    def validate_model(self, model):
        """
        验证模型在验证工作负载上的性能。

        参数:
        model (object): 要验证的模型。

        返回:
        tuple: 包含每个验证工作负载的模型性能和验证标识的元组。
        """
        # 初始化一个空列表，用于存储每个验证工作负载的模型性能
        model_performances = []
        # 遍历所有验证工作负载
        for validation_wl in self.workload_generator.wl_validation:
            # 创建一个虚拟向量环境，用于评估模型在验证工作负载上的性能
            validation_env = self.DummyVecEnv([self.make_env(0, EnvironmentType.VALIDATION, validation_wl)])
            # 对验证环境进行向量归一化处理，仅对观测值进行归一化，不处理奖励
            validation_env = self.VecNormalize(
                validation_env,
                norm_obs=True,
                norm_reward=False,
                gamma=self.config["rl_algorithm"]["gamma"],
                training=False,
            )

            # 如果当前模型不是默认模型，则将其环境设置为默认模型的环境
            if model != self.model:
                model.set_env(self.model.env)

            # 评估模型在验证环境上的性能
            model_performance = self._evaluate_model(model, validation_env, len(validation_wl))
            # 将评估结果添加到模型性能列表中
            model_performances.append(model_performance)

        # 返回模型在所有验证工作负载上的性能和验证标识
        return model_performances, "validation"


    def _evaluate_model(self, model, evaluation_env, n_eval_episodes):
        """
        评估模型在给定环境中的性能。

        参数:
        model (object): 要评估的模型。
        evaluation_env (object): 用于评估的环境。
        n_eval_episodes (int): 评估的回合数。

        返回:
        tuple: 包含每个回合的性能、平均性能和每个回合的性能列表的元组。
        """
        # 获取模型的向量归一化环境
        training_env = model.get_vec_normalize_env()
        # 同步训练环境和评估环境的归一化参数
        self.sync_envs_normalization(training_env, evaluation_env)

        # 记录评估开始时间
        flag1 = datetime.datetime.now()
        # 使用评估环境对模型进行指定回合数的评估
        self.evaluate_policy(model, evaluation_env, n_eval_episodes)
        # 计算评估所花费的时间
        flag11 = datetime.datetime.now() - flag1
        print("eval time:")
        print(flag11)

        # 从评估环境中获取每个回合的性能数据
        episode_performances = evaluation_env.get_attr("episode_performances")[0]
        # 初始化一个空列表，用于存储每个回合的性能得分
        perfs = []
        # 遍历每个回合的性能数据
        for perf in episode_performances:
            # 提取每个回合的实际成本，并四舍五入到小数点后两位
            perfs.append(round(perf["achieved_cost"], 2))

        # 计算所有回合的平均性能得分
        mean_performance = np.mean(perfs)
        print(f"Mean performance: {mean_performance:.2f} ({perfs})")

        # 返回每个回合的性能数据、平均性能得分和每个回合的性能得分列表
        return episode_performances, mean_performance, perfs


    def make_env(self, env_id, environment_type=EnvironmentType.TRAINING, workloads_in=None):
        """
        创建一个用于训练、测试或验证的环境初始化函数。

        参数:
        env_id (int): 环境的唯一标识符。
        environment_type (EnvironmentType, 可选): 环境的类型，默认为训练环境。
        workloads_in (list, 可选): 传入的工作负载列表，默认为None。

        返回:
        function: 一个初始化环境的函数。
        """
        def _init():
            # 动态导入动作管理器类
            action_manager_class = getattr(
                importlib.import_module("balance.action_manager"), self.config["action_manager"]
            )
            # 初始化动作管理器
            action_manager = action_manager_class(
                indexable_column_combinations=self.globally_indexable_columns,
                action_storage_consumptions=self.action_storage_consumptions,
                sb_version=self.config["rl_algorithm"]["stable_baselines_version"],
                max_index_width=self.config["max_index_width"],
                reenable_indexes=self.config["reenable_indexes"],
            )

            # 如果动作数量未设置，则进行设置
            if self.number_of_actions is None:
                self.number_of_actions = action_manager.number_of_actions

            # 配置观测管理器
            observation_manager_config = {
                # 工作负载的查询类数量
                "number_of_query_classes": self.workload_generator.number_of_query_classes,
                # 工作负载嵌入器
                "workload_embedder": self.workload_embedder if "workload_embedder" in self.config else None,
                # 工作负载的大小
                "workload_size": self.config["workload"]["size"],
            }
            # 动态导入观测管理器类
            observation_manager_class = getattr(
                importlib.import_module("balance.observation_manager"), self.config["observation_manager"]
            )
            # 初始化观测管理器
            observation_manager = observation_manager_class(
                action_manager.number_of_columns, observation_manager_config
            )

            # 如果特征数量未设置，则进行设置
            if self.number_of_features is None:
                self.number_of_features = observation_manager.number_of_features

            # 动态导入奖励计算器类
            reward_calculator_class = getattr(
                importlib.import_module("balance.reward_calculator"), self.config["reward_calculator"]
            )
            # 初始化奖励计算器
            reward_calculator = reward_calculator_class()

            # 根据环境类型选择工作负载
            if environment_type == EnvironmentType.TRAINING:
                # 如果没有传入工作负载，则使用训练工作负载
                workloads = self.workload_generator.wl_training if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.TESTING:
                # 如果没有传入工作负载，则使用最后一个测试工作负载
                workloads = self.workload_generator.wl_testing[-1] if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.VALIDATION:
                # 如果没有传入工作负载，则使用最后一个验证工作负载
                workloads = self.workload_generator.wl_validation[-1] if workloads_in is None else workloads_in
            else:
                # 不支持的环境类型，抛出错误
                raise ValueError

            # 创建OpenAI Gym环境
            env = gym.make(
                f"DB-v{self.config['gym_version']}",
                environment_type=environment_type,
                config={
                    # 数据库名称
                    "database_name": self.schema.database_name,
                    # 全局可索引的列组合
                    "globally_indexable_columns": self.globally_indexable_columns_flat,
                    # 工作负载列表
                    "workloads": workloads,
                    # 随机种子
                    "random_seed": self.config["random_seed"] + env_id,
                    # 每个回合的最大步数
                    "max_steps_per_episode": self.config["max_steps_per_episode"],
                    # 动作管理器
                    "action_manager": action_manager,
                    # 观测管理器
                    "observation_manager": observation_manager,
                    # 奖励计算器
                    "reward_calculator": reward_calculator,
                    # 环境ID
                    "env_id": env_id,
                    # 相似工作负载标志
                    "similar_workloads": self.config["workload"]["similar_workloads"],
                    # 实验ID
                    "ids": self.config["id"],
                },
            )
            return env

        # 设置随机种子
        self.set_random_seed(self.config["random_seed"])

        return _init


 

    def _set_sb_version_specific_methods(self):
        if self.config["rl_algorithm"]["stable_baselines_version"] == 2:
            from stable_baselines.common import set_global_seeds as set_global_seeds_sb2
            from stable_baselines.common.evaluation import evaluate_policy as evaluate_policy_sb2
            from stable_baselines.common.vec_env import DummyVecEnv as DummyVecEnv_sb2
            from stable_baselines.common.vec_env import VecNormalize as VecNormalize_sb2
            from stable_baselines.common.vec_env import sync_envs_normalization as sync_envs_normalization_sb2

            self.set_random_seed = set_global_seeds_sb2
            self.evaluate_policy = evaluate_policy_sb2
            self.DummyVecEnv = DummyVecEnv_sb2
            self.VecNormalize = VecNormalize_sb2
            self.sync_envs_normalization = sync_envs_normalization_sb2
        elif self.config["rl_algorithm"]["stable_baselines_version"] == 3:
            raise ValueError("Currently, only StableBaselines 2 is supported.")

            from stable_baselines3.common.evaluation import evaluate_policy as evaluate_policy_sb3
            from stable_baselines3.common.utils import set_random_seed as set_random_seed_sb3
            from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnv_sb3
            from stable_baselines3.common.vec_env import VecNormalize as VecNormalize_sb3
            from stable_baselines3.common.vec_env import sync_envs_normalization as sync_envs_normalization_sb3

            self.set_random_seed = set_random_seed_sb3
            self.evaluate_policy = evaluate_policy_sb3
            self.DummyVecEnv = DummyVecEnv_sb3
            self.VecNormalize = VecNormalize_sb3
            self.sync_envs_normalization = sync_envs_normalization_sb3
        else:
            raise ValueError("There are only versions 2 and 3 of StableBaselines.")


    def compare(self):
        # 检查配置文件中指定的比较算法列表是否为空，如果为空则直接返回
        if len(self.config["comparison_algorithms"]) < 1:
            return

        # 如果配置文件中指定了 "extend" 算法，则调用 _compare_extend 方法进行比较
        if "extend" in self.config["comparison_algorithms"]:
            self._compare_extend()
        if "extend_partition" in self.config["comparison_algorithms"]:
            self._compare_extend_partition()
        # 如果配置文件中指定了 "slalom" 算法，则调用 _compare_slalom 方法进行比较
        if "slalom" in self.config["comparison_algorithms"]:
            self._compare_slalom()
        # 如果配置文件中指定了 "db2advis" 算法，则调用 _compare_db2advis 方法进行比较
        if "db2advis" in self.config["comparison_algorithms"]:
            self._compare_db2advis()
        # 遍历比较性能字典，打印每个比较的结果
        for key, comparison_performance in self.comparison_performances.items():
            print(f"Comparison for {key}:")
            # 遍历每个比较的性能指标，打印指标名称、平均值和具体值
            for key, value in comparison_performance.items():
                print(f"    {key}: {np.mean(value):.2f} ({value})")

        # 调用 _evaluate_comparison 方法对比较结果进行评估
        self._evaluate_comparison()


    def _evaluate_comparison(self):
        """
        评估比较结果，检查比较算法找到的索引是否包含不可索引的列。

        该方法遍历所有比较算法找到的索引，提取出所有索引涉及的列，
        并与可索引的单列集合进行比较。如果发现有不可索引的列，
        则记录严重错误信息，并抛出断言错误。

        Returns:
            None
        """
        # 遍历比较索引字典中的每个键值对
        for key, comparison_indexes in self.comparison_indexes.items():
            # 初始化一个空集合，用于存储从索引中提取的所有列
            columns_from_indexes = set()
            # 遍历当前比较算法找到的所有索引
            for index in comparison_indexes:
                # 遍历当前索引中的所有列
                for column in index.columns:
                    # 将当前列添加到 columns_from_indexes 集合中
                    columns_from_indexes |= set([column])

            # 找出 columns_from_indexes 集合中不在 self.single_column_flat_set 集合中的列，即不可索引的列
            impossible_index_columns = columns_from_indexes - self.single_column_flat_set
            # 记录严重错误信息，显示当前比较算法找到的索引中包含不可索引的列
            logging.critical(f"{key} finds indexes on these not indexable columns:\n    {impossible_index_columns}")

            # 断言不可索引的列数量为 0，如果不为 0 则抛出断言错误
            assert len(impossible_index_columns) == 0, "Found indexes on not indexable columns."

    def _compare_extend(self):
        self.evaluated_workloads = set()
        extend_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
        extend_connector.drop_indexes()
        extend_algorithm = ExtendAlgorithm(extend_connector)

        run_type = "test"
        for test_wl in self.workload_generator.wl_testing[0]:
            self.comparison_performances[run_type]["Extend"].append([])
            # self.evaluated_workloads.add(test_wl)

            parameters = {
                "budget_MB": test_wl.budget,
                "max_index_width": self.config["max_index_width"],
                "min_cost_improvement": 1.0003,
            }
            extend_algorithm.reset(parameters)
            indexes = extend_algorithm.calculate_best_indexes(test_wl)
            self.comparison_indexes["Extend"] |= frozenset(indexes)

            self.comparison_performances[run_type]["Extend"][-1].append(extend_algorithm.final_cost_proportion)

        run_type = "validation"
        for validation_wl in self.workload_generator.wl_validation[0]:
            self.comparison_performances[run_type]["Extend"].append([])

            parameters = {
                "budget_MB": validation_wl.budget,
                "max_index_width": self.config["max_index_width"],
                "min_cost_improvement": 1.0003,
            }
            extend_algorithm.reset(parameters)
            indexes = extend_algorithm.calculate_best_indexes(validation_wl)
            self.comparison_indexes["Extend"] |= frozenset(indexes)

            self.comparison_performances[run_type]["Extend"][-1].append(extend_algorithm.final_cost_proportion)

    

