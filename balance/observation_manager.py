import logging

import numpy as np
from gym import spaces

VERY_HIGH_BUDGET = 100_000_000_000


class ObservationManager(object):
    def __init__(self, number_of_actions):
        """
        初始化 ObservationManager 类的实例。

        参数:
        number_of_actions (int): 动作的数量。
        """
        # 存储动作的数量
        self.number_of_actions = number_of_actions

    def _init_episode(self, state_fix_for_episode):
        """
        初始化每个回合的状态。

        参数:
        state_fix_for_episode (dict): 每个回合固定的状态信息，包含预算和初始成本。
        """
        # 从状态信息中获取本回合的预算
        self.episode_budget = state_fix_for_episode["budget"]
        # 如果预算为 None，则使用预设的非常高的预算
        if self.episode_budget is None:
            self.episode_budget = VERY_HIGH_BUDGET

        # 从状态信息中获取初始成本
        self.initial_cost = state_fix_for_episode["initial_cost"]

    def init_episode(self, state_fix_for_episode):
        """
        初始化每个回合的状态，此方法需要在子类中实现。

        参数:
        state_fix_for_episode (dict): 每个回合固定的状态信息。

        抛出:
        NotImplementedError: 如果在子类中未实现此方法。
        """
        raise NotImplementedError

    def get_observation(self, environment_state):
        """
        获取当前环境状态的观测值，此方法需要在子类中实现。

        参数:
        environment_state (dict): 当前环境的状态信息。

        抛出:
        NotImplementedError: 如果在子类中未实现此方法。
        """
        raise NotImplementedError

    def get_observation_space(self):
        """
        获取观测空间。

        返回:
        gym.spaces.Box: 表示观测空间的 Box 对象。
        """
        # 创建观测空间，指定上下界和形状
        observation_space = spaces.Box(
            low=self._create_low_boundaries(), high=self._create_high_boundaries(), shape=self._create_shape()
        )

        # 记录日志，显示创建的观测空间的特征数量
        logging.info(f"Creating ObservationSpace with {self.number_of_features} features.")

        return observation_space

    def _create_shape(self):
        """
        创建观测空间的形状。

        返回:
        tuple: 观测空间的形状，为一个包含特征数量的元组。
        """
        return (self.number_of_features,)

    def _create_low_boundaries(self):
        """
        创建观测空间的下界。

        返回:
        np.ndarray: 包含所有特征下界的 numpy 数组，每个特征的下界为负无穷。
        """
        # 为每个特征创建负无穷的下界
        low = [-np.inf for feature in range(self.number_of_features)]

        return np.array(low)

    def _create_high_boundaries(self):
        """
        创建观测空间的上界。

        返回:
        np.ndarray: 包含所有特征上界的 numpy 数组，每个特征的上界为正无穷。
        """
        # 为每个特征创建正无穷的上界
        high = [np.inf for feature in range(self.number_of_features)]

        return np.array(high)

class EmbeddingObservationManager(ObservationManager):
    def __init__(self, number_of_actions, config):
        """
        初始化 EmbeddingObservationManager 类的实例。

        参数:
        number_of_actions (int): 动作的数量。
        config (dict): 配置信息，包含工作负载嵌入器和工作负载大小。
        """
        # 调用父类的构造函数
        ObservationManager.__init__(self, number_of_actions)

        # 从配置中获取工作负载嵌入器
        self.workload_embedder = config["workload_embedder"]
        # 获取嵌入表示的真实大小
        self.representation_size = self.workload_embedder.true_representation_size
        # 从配置中获取工作负载的大小
        self.workload_size = config["workload_size"]

        # 计算特征的总数
        self.number_of_features = (
            self.number_of_actions  # 指示每个动作是否被执行
            + (
                self.representation_size * self.workload_size
            )  # 工作负载中每个查询的嵌入表示
            + self.workload_size  # 工作负载中每个查询的频率
            + 1  # 回合的预算
            + 1  # 当前的存储消耗
            + 1  # 初始工作负载成本
            + 1  # 当前工作负载成本
        )

    def _init_episode(self, state_fix_for_episode):
        """
        初始化每个回合的状态。

        参数:
        state_fix_for_episode (dict): 每个回合固定的状态信息，包含工作负载。
        """
        # 从状态信息中获取本回合的工作负载
        episode_workload = state_fix_for_episode["workload"]
        # 从工作负载中获取每个查询的频率
        self.frequencies = np.array(EmbeddingObservationManager._get_frequencies_from_workload(episode_workload))

        # 调用父类的 _init_episode 方法
        super()._init_episode(state_fix_for_episode)

    def init_episode(self, state_fix_for_episode):
        """
        初始化每个回合的状态，此方法需要在子类中实现。

        参数:
        state_fix_for_episode (dict): 每个回合固定的状态信息。

        抛出:
        NotImplementedError: 如果在子类中未实现此方法。
        """
        raise NotImplementedError

    def get_observation(self, environment_state):
        """
        获取当前环境状态的观测值。

        参数:
        environment_state (dict): 当前环境的状态信息，包含动作状态、查询计划和成本等。

        返回:
        np.ndarray: 包含观测值的 numpy 数组。
        """
        # 根据配置决定是否每步更新工作负载嵌入
        if self.UPDATE_EMBEDDING_PER_OBSERVATION:
            # 获取当前查询计划的嵌入表示
            workload_embedding = np.array(self.workload_embedder.get_embeddings(environment_state["plans_per_query"]))
        else:
            # 如果工作负载嵌入未初始化，则获取嵌入表示
            if self.workload_embedding is None:
                self.workload_embedding = np.array(
                    self.workload_embedder.get_embeddings(environment_state["plans_per_query"])
                )
            # 使用已有的工作负载嵌入
            workload_embedding = self.workload_embedding

        # 获取动作状态
        observation = np.array(environment_state["action_status"])
        # 将工作负载嵌入添加到观测值中
        observation = np.append(observation, workload_embedding)
        # 将查询频率添加到观测值中
        observation = np.append(observation, self.frequencies)
        # 将回合预算添加到观测值中
        observation = np.append(observation, self.episode_budget)
        # 将当前存储消耗添加到观测值中
        observation = np.append(observation, environment_state["current_storage_consumption"])
        # 将初始工作负载成本添加到观测值中
        observation = np.append(observation, self.initial_cost)
        # 将当前工作负载成本添加到观测值中
        observation = np.append(observation, environment_state["current_cost"])

        return observation

    @staticmethod
    def _get_frequencies_from_workload(workload):
        """
        从工作负载中获取每个查询的频率。

        参数:
        workload (Workload): 工作负载对象，包含多个查询。

        返回:
        list: 包含每个查询频率的列表。
        """
        frequencies = []
        # 遍历工作负载中的每个查询
        for query in workload.queries:
            # 将查询的频率添加到列表中
            frequencies.append(query.frequency)
        return frequencies


# Todo: Rename. Single/multi-column is not handled by the ObservationManager anymore.
# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexWorkloadEmbeddingObservationManager(EmbeddingObservationManager):
    def __init__(self, number_of_actions, config):
        super().__init__(number_of_actions, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = False

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

        self.workload_embedding = np.array(self.workload_embedder.get_embeddings(state_fix_for_episode["workload"]))


# Todo: Rename. Single/multi-column is not handled by the ObservationManager anymore.
# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexPlanEmbeddingObservationManager(EmbeddingObservationManager):
    def __init__(self, number_of_actions, config):
        super().__init__(number_of_actions, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)


# Todo: Rename. Single/multi-column is not handled by the ObservationManager anymore.
# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexPlanEmbeddingObservationManagerWithoutPlanUpdates(EmbeddingObservationManager):
    def __init__(self, number_of_actions, config):
        super().__init__(number_of_actions, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = False

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

        self.workload_embedding = None
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from src.feature_extraction.extract_features import *

from src.parameters import *
from src.plan_encoding.meta_info import *
from balance.utils import *

# Todo: Rename. Single/multi-column is not handled by the ObservationManager anymore.
# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexPlanEmbeddingObservationManagerWithCost(EmbeddingObservationManager):
    def __init__(self, number_of_actions, config):
        """
        初始化 SingleColumnIndexPlanEmbeddingObservationManagerWithCost 类的实例。

        参数:
        number_of_actions (int): 动作的数量。
        config (dict): 配置信息，包含工作负载嵌入器和工作负载大小等。
        """
        # 调用父类的构造函数
        super().__init__(number_of_actions, config)

        # 每步更新工作负载嵌入
        self.UPDATE_EMBEDDING_PER_OBSERVATION = True
        ########################################
        # 从工作负载嵌入器中获取数据库连接器
        self.db_connector = self.workload_embedder.database_connector
        # 获取参数
        self.parameters = self.getParameters()
        
        print("use boo")
        # 重新计算特征数量，覆盖父类的特征数量
        self.number_of_features = (
            self.number_of_actions  # 指示每个动作是否被执行
            + (
                self.representation_size * self.workload_size
            )  # 工作负载的嵌入表示
            + self.workload_size  # 工作负载中每个查询的成本
            + self.workload_size  # 工作负载中每个查询的频率
            + 1  # 回合的预算
            + 1  # 当前的存储消耗
            + 1  # 初始工作负载成本
            + 1  # 当前工作负载成本
        )

    def getParameters(self):
        """
        获取参数。

        返回:
        Parameters: 参数对象。
        """
        # 准备数据集，获取相关信息
        column2pos, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, tables, columnTypeisNum, box_lines = prepare_dataset()
        # 表的总数
        table_total_num = len(tables_id)
        # 列的总数
        column_total_num = len(columns_id)
        # 物理操作的总数
        physic_op_total_num = len(physic_ops_id)
        # 比较操作的总数
        compare_ops_total_num = len(compare_ops_id)
        # 布尔操作的总数
        bool_ops_total_num = len(bool_ops_id)
        # 盒子数量
        box_num = 10
        # 条件操作的维度
        condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num + box_num

        # 创建参数对象
        parameters = Parameters(tables_id, columns_id, physic_ops_id, column_total_num,
                    table_total_num, physic_op_total_num, condition_op_dim, compare_ops_id, bool_ops_id,
                    bool_ops_total_num, compare_ops_total_num, box_num, columnTypeisNum, box_lines)
        return parameters    

    def init_episode(self, state_fix_for_episode):
        """
        初始化每个回合的状态。

        参数:
        state_fix_for_episode (dict): 每个回合固定的状态信息，包含工作负载。
        """
        # 调用父类的 _init_episode 方法
        super()._init_episode(state_fix_for_episode)

    def w_get_detail_repr(self,w):
        """
        获取工作负载的详细表示。

        参数:
        w (Workload): 工作负载对象。

        返回:
        str: 工作负载的详细表示字符串。
        """
        # 存储查询 ID 的列表
        ids = []
        # 存储查询频率的列表
        fr = []
        # 存储查询文本的列表
        texts = []
        # 遍历工作负载中的每个查询
        for query in w.queries:
            # 添加查询 ID
            ids.append(query.nr)
            # 添加查询频率
            fr.append(query.frequency)
            # 添加查询文本
            texts.append(query.text)
        return f"Query IDs: {ids} with {fr}. {w.description} Budget: None Detail: {texts}"

    def get_observation(self, environment_state,dn = None):
        """
        获取当前环境状态的观测值。

        参数:
        environment_state (dict): 当前环境的状态信息，包含动作状态、查询计划和成本等。
        dn (Optional): 可选参数，未使用。

        返回:
        np.ndarray: 包含观测值的 numpy 数组。
        """
        # 获取当前查询计划的嵌入表示
        workload_embedding = np.array(self.workload_embedder.get_embeddings(environment_state["plans_per_query"]))
        
        # 获取动作状态
        observation = np.array(environment_state["action_status"])
        # 将工作负载嵌入添加到观测值中
        observation = np.append(observation, workload_embedding)
        # 将每个查询的成本添加到观测值中
        observation = np.append(observation, environment_state["costs_per_query"])
        # 将查询频率添加到观测值中
        observation = np.append(observation, self.frequencies)
        # 将回合预算添加到观测值中
        observation = np.append(observation, self.episode_budget)
        # 将当前存储消耗添加到观测值中
        observation = np.append(observation, environment_state["current_storage_consumption"])
        # 将初始工作负载成本添加到观测值中
        observation = np.append(observation, self.initial_cost)
        # 将当前工作负载成本添加到观测值中
        observation = np.append(observation, environment_state["current_cost"])

        return observation



class SingleColumnIndexObservationManager(ObservationManager):
    def __init__(self, number_of_actions, config):
        ObservationManager.__init__(self, number_of_actions)

        self.number_of_query_classes = config["number_of_query_classes"]

        self.number_of_features = (
            self.number_of_actions  # Indicates for each action whether it was taken or not
            + self.number_of_query_classes  # The frequencies for every query class
            + 1  # The episode's budget
            + 1  # The current storage consumption
            + 1  # The initial workload cost
            + 1  # The current workload cost
        )

    def init_episode(self, state_fix_for_episode):
        episode_workload = state_fix_for_episode["workload"]
        super()._init_episode(state_fix_for_episode)
        self.frequencies = np.array(self._get_frequencies_from_workload_wide(episode_workload))

    def get_observation(self, environment_state):
        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, self.frequencies)
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation

    def _get_frequencies_from_workload_wide(self, workload):
        frequencies = [0 for query in range(self.number_of_query_classes)]

        for query in workload.queries:
            # query numbers stat at 1
            frequencies[query.nr - 1] = query.frequency

        return frequencies
