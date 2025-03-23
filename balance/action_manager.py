import copy
import logging

import numpy as np
from gym import spaces

from index_selection_evaluation.selection.utils import b_to_mb

FORBIDDEN_ACTION_SB3 = -np.inf
ALLOWED_ACTION_SB3 = 0

FORBIDDEN_ACTION_SB2 = 0
ALLOWED_ACTION_SB2 = 1


class ActionManager(object):
    """
    动作管理器的基类，用于管理动作的有效性和状态。

    参数:
    sb_version (int): Stable Baselines的版本号，用于确定动作的有效性表示。
    max_index_width (int): 最大索引宽度。
    """
    def __init__(self, sb_version, max_index_width):
        # 存储当前所有动作的有效性状态，初始为None
        self.valid_actions = None
        # 存储剩余的有效动作索引，初始为None
        self._remaining_valid_actions = None
        # 存储动作的总数，初始为None
        self.number_of_actions = None
        # 存储当前动作的状态，初始为None
        self.current_action_status = None

        # 测试变量，用于调试目的
        self.test_variable = None

        # 存储Stable Baselines的版本号
        self.sb_version = sb_version
        # 存储最大索引宽度
        self.MAX_INDEX_WIDTH = max_index_width

        # 根据Stable Baselines版本设置禁止和允许的动作值
        if self.sb_version == 2:
            self.FORBIDDEN_ACTION = FORBIDDEN_ACTION_SB2
            self.ALLOWED_ACTION = ALLOWED_ACTION_SB2
        else:
            self.FORBIDDEN_ACTION = FORBIDDEN_ACTION_SB3
            self.ALLOWED_ACTION = ALLOWED_ACTION_SB3

    def get_action_space(self):
        """
        获取动作空间，使用离散空间表示。

        返回:
        spaces.Discrete: 离散动作空间。
        """
        return spaces.Discrete(self.number_of_actions)

    def get_initial_valid_actions(self, workload, budget):
        """
        获取初始的有效动作列表。

        参数:
        workload: 工作负载信息。
        budget: 存储预算。

        返回:
        np.array: 初始有效动作列表。
        """
        # 0 表示动作尚未执行，1 表示单列索引已存在，0.5 表示两列索引已存在，0.33 表示三列索引已存在，依此类推
        self.current_action_status = [0 for action in range(self.number_of_columns)]

        # 初始化所有动作为禁止动作
        self.valid_actions = [self.FORBIDDEN_ACTION for action in range(self.number_of_actions)]
        # 初始化剩余有效动作列表为空
        self._remaining_valid_actions = []

        # 根据工作负载更新有效动作
        self._valid_actions_based_on_workload(workload)
        # 根据预算更新有效动作
        self._valid_actions_based_on_budget(budget, current_storage_consumption=0)

        # 初始化当前组合集合
        self.current_combinations = set()

        return np.array(self.valid_actions)

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        """
        根据最后一个动作更新有效动作列表。

        参数:
        last_action (int): 最后一个执行的动作索引。
        budget: 存储预算。
        current_storage_consumption: 当前存储消耗。

        返回:
        tuple: 包含更新后的有效动作列表和是否还有有效动作的布尔值。
        """
        # 确保最后一个动作的组合不在当前组合集合中
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        # 获取最后一个动作的索引宽度
        actions_index_width = len(self.indexable_column_combinations_flat[last_action])
        if actions_index_width == 1:
            # 如果是单列索引，将当前动作状态加 1
            self.current_action_status[last_action] += 1
        else:
            # 获取要扩展的组合
            combination_to_be_extended = self.indexable_column_combinations_flat[last_action][:-1]
            # 确保要扩展的组合已存在于当前组合集合中
            assert combination_to_be_extended in self.current_combinations

            # 计算状态值
            status_value = 1 / actions_index_width

            # 获取最后一个动作的最后一列
            last_action_back_column = self.indexable_column_combinations_flat[last_action][-1]
            # 获取最后一列的索引
            last_action_back_columns_idx = self.column_to_idx[last_action_back_column]
            # 更新当前动作状态
            self.current_action_status[last_action_back_columns_idx] += status_value

            # 从当前组合集合中移除要扩展的组合
            self.current_combinations.remove(combination_to_be_extended)

        # 将最后一个动作的组合添加到当前组合集合中
        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        # 将最后一个动作标记为禁止动作
        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        # 如果最后一个动作在剩余有效动作列表中，将其移除
        if last_action in self._remaining_valid_actions:
            self._remaining_valid_actions.remove(last_action)

        # 根据最后一个动作更新有效动作
        self._valid_actions_based_on_last_action(last_action)
        # 根据预算更新有效动作
        self._valid_actions_based_on_budget(budget, current_storage_consumption)

        # 检查是否还有有效动作
        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        """
        根据预算更新有效动作列表。

        参数:
        budget: 存储预算。
        current_storage_consumption: 当前存储消耗。
        """
        if budget is None:
            # 如果没有预算限制，不做任何处理
            return
        else:
            # 初始化新的剩余有效动作列表
            new_remaining_actions = []
            for action_idx in self._remaining_valid_actions:
                # 检查执行该动作后是否会超出预算
                if b_to_mb(current_storage_consumption + self.action_storage_consumptions[action_idx]) > budget:
                    # 如果超出预算，将该动作标记为禁止动作
                    self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
                else:
                    # 如果未超出预算，将该动作添加到新的剩余有效动作列表中
                    new_remaining_actions.append(action_idx)

            # 更新剩余有效动作列表
            self._remaining_valid_actions = new_remaining_actions

    def _valid_actions_based_on_workload(self, workload):
        """
        根据工作负载更新有效动作列表，此方法需要在子类中实现。

        参数:
        workload: 工作负载信息。
        """
        raise NotImplementedError

    def _valid_actions_based_on_last_action(self, last_action):
        """
        根据最后一个动作更新有效动作列表，此方法需要在子类中实现。

        参数:
        last_action (int): 最后一个执行的动作索引。
        """
        raise NotImplementedError



class DRLindaActionManager(ActionManager):
    def __init__(
        self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width, reenable_indexes
    ):
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width)

        self.indexable_column_combinations = indexable_column_combinations
        # This is the same as the Expdriment's object globally_indexable_columns_flat attribute
        self.indexable_column_combinations_flat = [
            item for sublist in self.indexable_column_combinations for item in sublist
        ]
        self.number_of_actions = len(self.indexable_column_combinations_flat)
        self.number_of_columns = len(self.indexable_column_combinations[0])

        self.action_storage_consumptions = action_storage_consumptions

        self.indexable_columns = list(
            map(lambda one_column_combination: one_column_combination[0], self.indexable_column_combinations[0])
        )

        self.column_to_idx = {}
        for idx, column in enumerate(self.indexable_column_combinations[0]):
            c = column[0]
            self.column_to_idx[c] = idx

    def get_action_space(self):
        return spaces.Discrete(self.number_of_actions)

    def get_initial_valid_actions(self, workload, budget):
        # 0 for actions not taken yet, 1 for single column index present
        self.current_action_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.ALLOWED_ACTION for action in range(self.number_of_actions)]
        self._remaining_valid_actions = list(range(self.number_of_columns))

        self.current_combinations = set()

        return np.array(self.valid_actions)

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        # actions_index_width = len(self.indexable_column_combinations_flat[last_action])
        # if actions_index_width == 1:
        self.current_action_status[last_action] = 1

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        pass

    def _valid_actions_based_on_workload(self, workload):
        pass

    def _valid_actions_based_on_last_action(self, last_action):
        pass

class MultiColumnIndexActionManager(ActionManager):
    def __init__(
        self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width, reenable_indexes
    ):
        """
        初始化 MultiColumnIndexActionManager 类的实例。

        参数:
        indexable_column_combinations (list): 可索引的列组合列表。
        action_storage_consumptions (list): 每个动作的存储消耗列表。
        sb_version (int): Stable Baselines 的版本号。
        max_index_width (int): 最大索引宽度。
        reenable_indexes (bool): 是否重新启用索引的标志。
        """
        # 调用父类 ActionManager 的构造函数
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width)

        # 存储可索引的列组合
        self.indexable_column_combinations = indexable_column_combinations
        # 扁平化可索引的列组合，将嵌套列表展开为一维列表
        # 这与 Expdriment 对象的 globally_indexable_columns_flat 属性相同
        self.indexable_column_combinations_flat = [
            item for sublist in self.indexable_column_combinations for item in sublist
        ]
        # 计算动作的总数
        self.number_of_actions = len(self.indexable_column_combinations_flat)
        # 计算列的数量
        self.number_of_columns = len(self.indexable_column_combinations[0])
        # 存储每个动作的存储消耗
        self.action_storage_consumptions = action_storage_consumptions

        # 提取可索引的列
        self.indexable_columns = list(
            map(lambda one_column_combination: one_column_combination[0], self.indexable_column_combinations[0])
        )

        # 存储是否重新启用索引的标志
        self.REENABLE_INDEXES = reenable_indexes

        # 初始化列到索引的映射字典
        self.column_to_idx = {}
        # 遍历可索引的列组合，为每列创建索引映射
        for idx, column in enumerate(self.indexable_column_combinations[0]):
            # 取列组合的第一个元素作为列名
            c = column[0]
            # 将列名映射到索引
            self.column_to_idx[c] = idx

        # 初始化列组合到索引的映射字典
        self.column_combination_to_idx = {}
        # 遍历扁平化后的可索引列组合，为每个组合创建索引映射
        for idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            # 将列组合转换为字符串
            cc = str(column_combination)
            # 将列组合字符串映射到索引
            self.column_combination_to_idx[cc] = idx

        # 初始化候选依赖映射字典
        self.candidate_dependent_map = {}
        # 遍历扁平化后的可索引列组合
        for indexable_column_combination in self.indexable_column_combinations_flat:
            # 如果列组合的长度大于最大索引宽度减1，则跳过
            if len(indexable_column_combination) > max_index_width - 1:
                continue
            # 初始化该列组合的候选依赖列表为空
            self.candidate_dependent_map[indexable_column_combination] = []

        # 遍历扁平化后的可索引列组合及其索引
        for column_combination_idx, indexable_column_combination in enumerate(self.indexable_column_combinations_flat):
            # 如果列组合的长度小于2，则跳过
            if len(indexable_column_combination) < 2:
                continue
            # 获取列组合的前缀作为依赖项
            dependent_of = indexable_column_combination[:-1]
            # 将当前列组合的索引添加到依赖项的候选依赖列表中
            self.candidate_dependent_map[dependent_of].append(column_combination_idx)

    def _valid_actions_based_on_last_action(self, last_action):
        """
        根据最后一个执行的动作更新有效动作列表。

        参数:
        last_action (int): 最后一个执行的动作的索引。
        """
        # 获取最后一个动作对应的列组合
        last_combination = self.indexable_column_combinations_flat[last_action]
        # 获取最后一个动作对应列组合的长度
        last_combination_length = len(last_combination)

        # 如果最后一个组合的长度不等于最大索引宽度，尝试扩展组合
        if last_combination_length != self.MAX_INDEX_WIDTH:
            # 遍历最后一个组合的候选依赖组合索引
            for column_combination_idx in self.candidate_dependent_map[last_combination]:
                # 获取候选依赖组合
                indexable_column_combination = self.indexable_column_combinations_flat[column_combination_idx]
                # 获取候选依赖组合可能扩展的列
                possible_extended_column = indexable_column_combination[-1]

                # 如果可能扩展的列不在工作负载可索引列中，跳过该组合
                if possible_extended_column not in self.wl_indexable_columns:
                    continue
                # 如果候选依赖组合已经在当前组合集合中，跳过该组合
                if indexable_column_combination in self.current_combinations:
                    continue

                # 将该组合的索引添加到剩余有效动作列表中
                self._remaining_valid_actions.append(column_combination_idx)
                # 将该组合标记为允许的动作
                self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

        # 禁用在最后一个动作之后无效的组合
        for column_combination_idx in copy.copy(self._remaining_valid_actions):
            # 获取当前组合
            indexable_column_combination = self.indexable_column_combinations_flat[column_combination_idx]
            # 获取当前组合的长度
            indexable_column_combination_length = len(indexable_column_combination)
            # 如果当前组合是单列组合，跳过
            if indexable_column_combination_length == 1:
                continue

            # 如果当前组合的长度不等于最后一个组合的长度，跳过
            if indexable_column_combination_length != last_combination_length:
                continue

            # 如果当前组合的前缀与最后一个组合的前缀不同，跳过
            if last_combination[:-1] != indexable_column_combination[:-1]:
                continue

            # 如果该组合的索引在剩余有效动作列表中，移除它
            if column_combination_idx in self._remaining_valid_actions:
                self._remaining_valid_actions.remove(column_combination_idx)
            # 将该组合标记为禁止的动作
            self.valid_actions[column_combination_idx] = self.FORBIDDEN_ACTION

        # 如果允许重新启用索引且最后一个组合长度大于1
        if self.REENABLE_INDEXES and last_combination_length > 1:
            # 获取最后一个组合去掉最后一列的组合
            last_combination_without_extension = last_combination[:-1]

            # 如果去掉最后一列的组合长度大于1
            if len(last_combination_without_extension) > 1:
                # 获取去掉最后一列组合的父组合
                last_combination_without_extension_parent = last_combination_without_extension[:-1]
                # 如果父组合不在当前组合集合中，不做任何操作
                if last_combination_without_extension_parent not in self.current_combinations:
                    return

            # 获取去掉最后一列组合的索引
            column_combination_idx = self.column_combination_to_idx[str(last_combination_without_extension)]
            # 将该组合的索引添加到剩余有效动作列表中
            self._remaining_valid_actions.append(column_combination_idx)
            # 将该组合标记为允许的动作
            self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

            # 记录重新启用索引的日志
            logging.debug(f"REENABLE_INDEXES: {last_combination_without_extension} after {last_combination}")

    def _valid_actions_based_on_workload(self, workload):
        """
        根据工作负载更新有效动作列表。

        参数:
        workload: 工作负载信息。
        """
        # 从工作负载中获取可索引的列，不进行排序
        indexable_columns = workload.indexable_columns(return_sorted=False)
        # 取工作负载中可索引列与当前类可索引列的交集，更新可索引列
        indexable_columns = indexable_columns & frozenset(self.indexable_columns)
        # 将交集结果存储到类属性中，方便后续使用
        self.wl_indexable_columns = indexable_columns

        # 遍历可索引列
        for indexable_column in indexable_columns:
            # 仅处理单列索引，遍历第一组列组合及其索引
            for column_combination_idx, indexable_column_combination in enumerate(
                self.indexable_column_combinations[0]
            ):
                # 如果当前列组合的第一列等于可索引列
                if indexable_column == indexable_column_combination[0]:
                    # 将该列组合对应的动作标记为允许执行
                    self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION
                    # 将该列组合的索引添加到剩余有效动作列表中
                    self._remaining_valid_actions.append(column_combination_idx)




class MultiColumnIndexActionManagerNonMasking(ActionManager):
    def __init__(
        self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width, reenable_indexes
    ):
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width)

        self.indexable_column_combinations = indexable_column_combinations
        # This is the same as the Expdriment's object globally_indexable_columns_flat attribute
        self.indexable_column_combinations_flat = [
            item for sublist in self.indexable_column_combinations for item in sublist
        ]
        self.number_of_actions = len(self.indexable_column_combinations_flat)
        self.number_of_columns = len(self.indexable_column_combinations[0])
        self.action_storage_consumptions = action_storage_consumptions

        self.indexable_columns = list(
            map(lambda one_column_combination: one_column_combination[0], self.indexable_column_combinations[0])
        )

        self.REENABLE_INDEXES = reenable_indexes

        self.column_to_idx = {}
        for idx, column in enumerate(self.indexable_column_combinations[0]):
            c = column[0]
            self.column_to_idx[c] = idx

        self.column_combination_to_idx = {}
        for idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            cc = str(column_combination)
            self.column_combination_to_idx[cc] = idx

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        last_action_column_combination = self.indexable_column_combinations_flat[last_action]

        for idx, column in enumerate(last_action_column_combination):
            status_value = 1 / (idx + 1)
            last_action_columns_idx = self.column_to_idx[column]
            self.current_action_status[last_action_columns_idx] += status_value

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self._valid_actions_based_on_last_action(last_action)
        self._valid_actions_based_on_budget(budget, current_storage_consumption)

        return np.array(self.valid_actions), True

    def _valid_actions_based_on_last_action(self, last_action):
        pass

    def _valid_actions_based_on_workload(self, workload):
        self.valid_actions = [self.ALLOWED_ACTION for action in range(self.number_of_actions)]

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        pass



