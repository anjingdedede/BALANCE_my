from index_selection_evaluation.selection.utils import b_to_mb


class RewardCalculator(object):
    def __init__(self):
        """
        初始化 RewardCalculator 类的实例。
        调用 reset 方法来重置累积奖励。
        """
        self.reset()

    def reset(self):
        """
        重置累积奖励为 0。
        """
        self.accumulated_reward = 0

    def calculate_reward(self, environment_state):
        """
        根据环境状态计算奖励，并更新累积奖励。

        参数:
        environment_state (dict): 包含当前成本、先前成本、初始成本和新索引大小的环境状态。

        返回:
        float: 计算得到的奖励。
        """
        # 从环境状态中提取所需信息
        current_cost = environment_state["current_cost"]
        previous_cost = environment_state["previous_cost"]
        initial_cost = environment_state["initial_cost"]
        new_index_size = environment_state["new_index_size"]

        # 确保新索引大小不为 None
        assert new_index_size is not None

        # 调用 _calculate_reward 方法计算奖励
        reward = self._calculate_reward(current_cost, previous_cost, initial_cost, new_index_size)

        # 更新累积奖励
        self.accumulated_reward += reward

        return reward

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        """
        抽象方法，用于计算奖励。
        该方法需要在子类中实现。

        参数:
        current_cost (float): 当前成本。
        previous_cost (float): 先前成本。
        initial_cost (float): 初始成本。
        new_index_size (float): 新索引大小。

        抛出:
        NotImplementedError: 如果该方法未在子类中实现。
        """
        raise NotImplementedError


class AbsoluteDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (previous_cost - current_cost) / new_index_size

        return reward


class AbsoluteDifferenceToPreviousReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = previous_cost - current_cost

        return reward


class RelativeDifferenceToPreviousReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (previous_cost - current_cost) / initial_cost

        return reward


class RelativeDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self):
        """
        初始化 RelativeDifferenceRelativeToStorageReward 类的实例。
        调用父类的构造函数，并初始化缩放因子 SCALER。
        """
        RewardCalculator.__init__(self)
        # 初始化缩放因子，用于调整奖励的比例
        self.SCALER = 1

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        """
        计算相对于存储的相对差异奖励。

        参数:
        current_cost (float): 当前成本。
        previous_cost (float): 先前成本。
        initial_cost (float): 初始成本。
        new_index_size (float): 新索引大小。

        返回:
        float: 计算得到的奖励。
        """
        # 确保新索引大小大于 0，避免除零错误
        assert new_index_size > 0

        # 计算奖励，先计算成本的相对差异，再除以新索引大小（转换为 MB），最后乘以缩放因子
        reward = ((previous_cost - current_cost) / initial_cost) / b_to_mb(new_index_size) * self.SCALER

        return reward


class DRLindaReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = ((initial_cost - current_cost) / initial_cost) * 100

        return reward
