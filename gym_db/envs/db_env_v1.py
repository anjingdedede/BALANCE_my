import collections
import copy
import logging
import random

import os
import gym
import pickle
from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.utils import b_to_mb


class DBEnvV1(gym.Env):
    def __init__(self, environment_type=EnvironmentType.TRAINING, config=None):
        """
        初始化 DBEnvV1 环境。

        参数:
        environment_type (EnvironmentType): 环境类型，默认为训练环境。
        config (dict): 配置字典，包含随机种子、数据库名等信息。
        """
        # 调用父类 gym.Env 的构造函数
        super(DBEnvV1, self).__init__()

        # 创建随机数生成器并设置种子
        self.rnd = random.Random()
        self.rnd.seed(config["random_seed"])
        # 环境 ID
        self.env_id = config["env_id"]
        # 环境类型
        self.environment_type = environment_type
        # 配置字典
        self.config = config

        # 重置次数
        self.number_of_resets = 0
        # 总步数
        self.total_number_of_steps = 0

        # 创建 Postgres 数据库连接器并删除现有索引
        self.connector = PostgresDatabaseConnector(config["database_name"], autocommit=True)
        self.connector.drop_indexes()
        # 创建成本评估器
        self.cost_evaluation = CostEvaluation(self.connector)

        # 全局可索引的列
        self.globally_indexable_columns = config["globally_indexable_columns"]
        # 复制工作负载列表，避免修改原始配置
        self.workloads = copy.copy(config["workloads"])
        # 当前工作负载的索引
        self.current_workload_idx = 0
        # 是否使用相似工作负载
        self.similar_workloads = config["similar_workloads"]
        # 每回合的最大步数
        self.max_steps_per_episode = config["max_steps_per_episode"]

        # 动作管理器
        self.action_manager = config["action_manager"]
        # 设置动作管理器的测试变量
        self.action_manager.test_variable = self.env_id
        # 获取动作空间
        self.action_space = self.action_manager.get_action_space()

        # 观测管理器
        self.observation_manager = config["observation_manager"]
        # 获取观测空间
        self.observation_space = self.observation_manager.get_observation_space()

        # 奖励计算器
        self.reward_calculator = config["reward_calculator"]
        # 一个空字典
        self.dic = {}
        # 初始化可修改状态
        self._init_modifiable_state()

        # 如果不是训练环境，初始化一个固定长度的双端队列来存储每回合的性能
        if self.environment_type != environment_type.TRAINING:
            self.episode_performances = collections.deque(maxlen=len(config["workloads"]))

    def reset(self):
        """
        重置环境，将环境状态恢复到初始状态，并返回初始观测值。

        返回:
            初始化后的观测值。
        """
        # 增加重置次数
        self.number_of_resets += 1
        # 累计总步数
        self.total_number_of_steps += self.steps_taken

        # 初始化可修改的环境状态，并获取初始观测值
        initial_observation = self._init_modifiable_state()

        return initial_observation

    def _step_asserts(self, action):
        """
        检查动作的有效性。

        参数:
        action (int): 智能体选择的动作。
        """
        # 确保动作在动作空间内
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        # 确保动作是有效的动作
        assert (
            self.valid_actions[action] == self.action_manager.ALLOWED_ACTION
        ), f"Agent has chosen invalid action: {action}"
        # 确保选择的索引不在当前索引集合中
        assert (
            Index(self.globally_indexable_columns[action]) not in self.current_indexes
        ), f"{Index(self.globally_indexable_columns[action])} already in self.current_indexes"


    def step(self, action, start=False):
        """
        执行一个环境步骤。

        参数:
        action (int): 智能体选择的动作。
        start (bool): 表示是否为起始步骤，默认为 False。

        返回:
        tuple: 包含当前观测值、奖励、回合是否结束以及有效动作掩码的元组。
        """
        # 检查动作的有效性
        self._step_asserts(action)

        # 增加已采取的步数
        self.steps_taken += 1
        # 旧索引的大小，初始化为 0
        old_index_size = 0

        # 根据动作创建新的索引
        new_index = Index(self.globally_indexable_columns[action])
        # 将新索引添加到当前索引集合中
        self.current_indexes.add(new_index)

        # 初始化路径和测试标志
        path = ""
        testflag = False
        # 根据环境类型设置路径和测试标志
        if self.environment_type == EnvironmentType.TRAINING:
            path = path + "train" + "_"
            testflag = False
        elif self.environment_type == EnvironmentType.TESTING:
            path = path + "test" + "_"
            testflag = True
        elif self.environment_type == EnvironmentType.VALIDATION:
            path = path + "validation" + "_"
            testflag = False
        # 这里 path = path 可以去掉，属于多余代码
        path = path
        # 获取当前工作负载的 ID
        path2 = str(self.current_workload.idxx)

        # 起始标志，这里 startup 计算有误，False & start 恒为 False
        startup = False & start

        # 如果新索引不是单列表索引
        if not new_index.is_single_column():
            # 创建父索引
            parent_index = Index(new_index.columns[:-1])
            # 查找父索引并获取其大小
            for index in self.current_indexes:
                if index == parent_index:
                    old_index_size = index.estimated_size

            # 从当前索引集合中移除父索引
            self.current_indexes.remove(parent_index)

            # 确保父索引的大小大于 0
            assert old_index_size > 0, "Parent index size must have been found if not single column index."

        # 判断是否需要打印
        print_flag = (self.steps_taken >= self.max_steps_per_episode) and testflag

        # 更新并返回环境状态
        environment_state = self._update_return_env_state(
            init=False, new_index=new_index, old_index_size=old_index_size, print_flag=print_flag
        )
        # 获取当前的观测值
        current_observation = self.observation_manager.get_observation(environment_state, self.config["database_name"])

        # 更新有效动作列表
        self.valid_actions, is_valid_action_left = self.action_manager.update_valid_actions(
            action, self.current_budget, self.current_storage_consumption
        )
        # 判断回合是否结束
        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left

        # 计算奖励
        reward = self.reward_calculator.calculate_reward(environment_state)

        # 如果回合结束且不是训练环境
        if episode_done and self.environment_type != EnvironmentType.TRAINING:
            # 报告本回合的性能
            self._report_episode_performance(environment_state)
            # 更新当前工作负载的索引
            self.current_workload_idx += 1
            print(f"Indexes: {len(self.current_indexes)}")
        else:
            # 这里 ace = 1 可能是临时调试代码，可以考虑移除
            ace = 1

        return current_observation, reward, episode_done, {"action_mask": self.valid_actions}

    def _report_episode_performance(self, environment_state):
        """
        报告当前回合的性能指标，包括成本、内存消耗、可用预算、评估的工作负载和索引等信息。

        参数:
        environment_state (dict): 包含当前环境状态的字典。
        """
        # 计算并存储本回合的性能指标
        episode_performance = {
            # 计算当前成本相对于初始成本的百分比
            "achieved_cost": self.current_costs / self.initial_costs * 100,
            # 当前内存消耗
            "memory_consumption": self.current_storage_consumption,
            # 当前可用预算
            "available_budget": self.current_budget,
            # 当前评估的工作负载
            "evaluated_workload": self.current_workload,
            # 当前使用的索引集合
            "indexes": self.current_indexes,
        }

        # 构建输出信息，包含工作负载类型、成本变化、奖励、内存使用和索引数量等信息
        output = (
            # 输出评估的工作负载和环境类型
            f"Evaluated Workload ({self.environment_type}): {self.current_workload}\n    "
            # 输出初始成本、当前成本和成本百分比
            f"Initial cost: {self.initial_costs:,.2f}, now: {self.current_costs:,.2f} "
            f"({episode_performance['achieved_cost']:.2f}). Reward: {self.reward_calculator.accumulated_reward}.\n    "
            # 输出内存使用和索引数量
            f"Size: {b_to_mb(self.current_storage_consumption):.2f} with {len(self.current_indexes)} indexes:\n    "
            #f"{self.current_indexes}\n    "
        )
        # 使用日志记录输出信息
        logging.warning(output)

        # 将本回合的性能指标添加到性能队列中
        self.episode_performances.append(episode_performance)


    def _init_modifiable_state(self):
        """
        初始化可修改的环境状态，包括索引、步数、存储消耗、奖励计算器等。

        返回:
            初始化后的观测值。
        """
        # 初始化当前索引集合为空
        self.current_indexes = set()
        # 初始化已采取的步数为 0
        self.steps_taken = 0
        # 初始化当前存储消耗为 0
        self.current_storage_consumption = 0
        # 重置奖励计算器
        self.reward_calculator.reset()

        # 如果工作负载列表为空，则重新复制配置中的工作负载列表
        if len(self.workloads) == 0:
            self.workloads = copy.copy(self.config["workloads"])

        # 根据环境类型选择当前工作负载
        if self.environment_type == EnvironmentType.TRAINING:
            if self.similar_workloads:
                # 200 是一个任意值，用于从工作负载列表中选择特定的工作负载
                self.current_workload = self.workloads.pop(0 + self.env_id * 200)
            else:
                # 随机选择一个工作负载
                self.current_workload = self.rnd.choice(self.workloads)
        else:
            # 按顺序选择工作负载
            self.current_workload = self.workloads[self.current_workload_idx % len(self.workloads)]

        # 设置当前工作负载的预算
        self.current_budget = self.current_workload.budget
        # 初始化上一次的成本为 None
        self.previous_cost = None

        # 获取初始的有效动作列表
        self.valid_actions = self.action_manager.get_initial_valid_actions(self.current_workload, self.current_budget)

        # 更新并返回环境状态
        environment_state = self._update_return_env_state(init=True)

        # 固定本回合的状态信息
        state_fix_for_episode = {
            "budget": self.current_budget,
            "workload": self.current_workload,
            "initial_cost": self.initial_costs,
        }
        # 初始化观测管理器本回合的状态
        self.observation_manager.init_episode(state_fix_for_episode)
        # 获取初始的观测值
        initial_observation = self.observation_manager.get_observation(environment_state,self.config["database_name"])

        return initial_observation


    def _update_return_env_state(self, init, new_index=None, old_index_size=None , print_flag = False):
        """
        更新并返回环境状态。

        参数:
        init (bool): 是否为初始化状态。
        new_index (Index, optional): 新的索引，默认为 None。
        old_index_size (int, optional): 旧索引的大小，默认为 None。
        print_flag (bool, optional): 是否打印标志，默认为 False。

        返回:
        dict: 包含当前环境状态的字典。
        """
        # 计算当前工作负载在当前索引下的总成本、每个查询的执行计划和成本
        total_costs, plans_per_query, costs_per_query = self.cost_evaluation.calculate_cost_and_plans(
            self.current_workload, self.current_indexes, store_size=True
        )

        # 如果不是初始化状态，更新上一次的成本和存储消耗
        if not init:
            self.previous_cost = self.current_costs
            self.previous_storage_consumption = self.current_storage_consumption

        # 更新当前的总成本
        self.current_costs = total_costs

        # 如果是初始化状态，记录初始成本
        if init:
            self.initial_costs = total_costs

        # 新索引的大小
        new_index_size = None

        # 如果有新索引，更新存储消耗并计算新索引的大小
        if new_index is not None:
            # 更新当前存储消耗
            self.current_storage_consumption += new_index.estimated_size
            self.current_storage_consumption -= old_index_size

            # 确保新索引的估计大小不小于旧索引的大小
            assert new_index.estimated_size >= old_index_size

            # 计算新索引相对于旧索引的大小变化
            new_index_size = new_index.estimated_size - old_index_size
            if new_index_size == 0:
                new_index_size = 1

            # 确保存储消耗不超过预算
            if self.current_budget:
                assert b_to_mb(self.current_storage_consumption) <= self.current_budget, (
                    "Storage consumption exceeds budget: "
                    f"{b_to_mb(self.current_storage_consumption)} "
                    f" > {self.current_budget}"
                )

        # 构建环境状态字典
        environment_state = {
            "action_status": self.action_manager.current_action_status,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            "new_index_size": new_index_size,
            "plans_per_query": plans_per_query,
            "costs_per_query": costs_per_query,
            "workload":self.current_workload,
        }
        return environment_state


    def get_cost_eval_cache_info(self):
        """
        获取成本评估缓存的相关信息。

        返回:
        tuple: 包含成本请求次数、缓存命中次数和成本计算时间的元组。
        """
        return self.cost_evaluation.cost_requests, self.cost_evaluation.cache_hits, self.cost_evaluation.costing_time

    def get_cost_eval_cache(self):
        """
        获取成本评估的缓存。

        返回:
        dict: 成本评估的缓存字典。
        """
        return self.cost_evaluation.cache

    # BEGIN OF NOT IMPLEMENTED ##########
    def render(self, mode="human"):
        """
        渲染环境，目前仅打印调用信息。

        参数:
        mode (str): 渲染模式，默认为 "human"。
        """
        print("render() was called")
        pass

    def close(self):
        """
        关闭环境，目前仅打印调用信息。
        """
        print("close() was called")

    # END OF NOT IMPLEMENTED ##########
