import copy
import importlib
import logging
import pickle
import sys
import gym_db  # noqa: F401
from gym_db.common import EnvironmentType
from balance.experiment import Experiment
import os


use_gpu = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

if __name__ == "__main__":
    # 配置日志记录，设置日志级别为INFO
    logging.basicConfig(level=logging.INFO)

    # 定义配置文件路径
    CONFIGURATION_FILE = "experiments/tpch.json"

    # 记录警告信息，提示使用的GPU编号
    logging.warning("use gpu:" + use_gpu)
    # 创建实验对象，传入配置文件路径
    experiment = Experiment(CONFIGURATION_FILE)

    # 根据稳定基线库的版本导入不同的模块和类
    if experiment.config["rl_algorithm"]["stable_baselines_version"] == 2:
        # 从stable_baselines.common中导入回调函数和向量环境类
        from stable_baselines.common.callbacks import EvalCallbackWithTBRunningAverage
        from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
        # 从stable_baselines.ppo2中导入PPO2算法相关类
        from stable_baselines.ppo2 import ppo2, ppo2_BALANCE
        # 设置算法类为PPO2_BALANCE的PPO2类
        algorithm_class = ppo2_BALANCE.PPO2
        source_algorithm_class = ppo2_BALANCE.PPO2
    elif experiment.config["rl_algorithm"]["stable_baselines_version"] == 3:
        # 从stable_baselines3.common中导入回调函数和向量环境类
        from stable_baselines3.common.callbacks import EvalCallbackWithTBRunningAverage
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
        # 根据配置文件中的算法名称动态导入算法类
        algorithm_class = getattr(
            importlib.import_module("stable_baselines3"), experiment.config["rl_algorithm"]["algorithm"]
        )
    else:
        # 如果版本号不合法，抛出值错误异常
        raise ValueError

    # 准备实验
    experiment.prepare()
    # 将实验对象保存到pickle文件中
    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # 根据并行环境数量选择并行环境类
    ParallelEnv = SubprocVecEnv if experiment.config["parallel_environments"] > 1 else DummyVecEnv

    # 创建训练环境
    training_env = ParallelEnv(
        [experiment.make_env(env_id) for env_id in range(experiment.config["parallel_environments"])]
    )
    # 对训练环境进行归一化处理
    training_env = VecNormalize(
        training_env, norm_obs=True, norm_reward=True, gamma=experiment.config["rl_algorithm"]["gamma"], training=True
    )
    # 初始化一个空列表，用于存储源模型
    temac = []
    # 设置实验的源模型类型
    experiment.source_model_type = source_algorithm_class
    # 设置实验的模型类型
    experiment.model_type = algorithm_class

    # 以下代码被注释掉，可能是用于加载源模型的部分
    # path1 = "./experiment_results/source"
    # path2 = "./experiment_results/source"
    # path3 = "./experiment_results/source"
    #
    #
    #
    # experiment.Smodel_1 = experiment.source_model_type.load(path1+"/f_s1.zip")
    # experiment.Smodel_1.training = False
    # experiment.Smodel_2 = experiment.source_model_type.load(path2+"/f_s2.zip")
    # experiment.Smodel_2.training = False
    # experiment.Smodel_3 = experiment.source_model_type.load(path3+"/f_s3.zip")
    # experiment.Smodel_3.training = False
    #
    #
    # temac.append(experiment.Smodel_1)
    # temac.append(experiment.Smodel_2)
    # temac.append(experiment.Smodel_3)

    # 创建模型实例
    model = algorithm_class(
        # 设置策略类型
        policy=experiment.config["rl_algorithm"]["policy"],
        # 设置训练环境
        env=training_env,
        # 设置日志详细程度
        verbose=2,
        # 设置随机种子
        seed=experiment.config["random_seed"],
        # 设置折扣因子
        gamma=experiment.config["rl_algorithm"]["gamma"],
        # 设置TensorBoard日志路径
        tensorboard_log="tensor_log",
        # 注释掉的参数，可能用于传递源模型列表
        # acc = temac,
        # 复制模型架构配置，避免被修改
        policy_kwargs=copy.copy(
            experiment.config["rl_algorithm"]["model_architecture"]
        ),  # This is necessary because SB modifies the passed dict.
        # 传递其他算法参数
        **experiment.config["rl_algorithm"]["args"],
    )
    # 记录警告信息，提示创建的模型的神经网络架构
    logging.warning(f"Creating model with NN architecture: {experiment.config['rl_algorithm']['model_architecture']}")
    # 设置实验的模型
    experiment.set_model(model)

    # 创建测试环境的归一化向量环境
    callback_test_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.config["rl_algorithm"]["gamma"],
        training=False,
    )
    # 创建测试回调函数
    test_callback = EvalCallbackWithTBRunningAverage(
        # 设置评估的回合数
        n_eval_episodes=experiment.config["workload"]["validation_testing"]["number_of_workloads"],
        # 设置评估频率
        eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
        # 设置评估环境
        eval_env=callback_test_env,
        # 设置日志详细程度
        verbose=1,
        # 设置回调函数名称
        name="test",
        # 设置评估是否为确定性的
        deterministic=True,
        # 设置比较性能指标
        comparison_performances=experiment.comparison_performances["test"],
    )

    # 创建验证环境的归一化向量环境
    callback_validation_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.config["rl_algorithm"]["gamma"],
        training=False,
    )
    # 创建验证回调函数
    validation_callback = EvalCallbackWithTBRunningAverage(
        # 设置评估的回合数
        n_eval_episodes=experiment.config["workload"]["validation_testing"]["number_of_workloads"],
        # 设置评估频率
        eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
        # 设置评估环境
        eval_env=callback_validation_env,
        # 设置最佳模型保存路径
        best_model_save_path=experiment.experiment_folder_path,
        # 设置日志详细程度
        verbose=1,
        # 设置回调函数名称
        name="validation",
        # 设置评估是否为确定性的
        deterministic=True,
        # 设置比较性能指标
        comparison_performances=experiment.comparison_performances["validation"],
    )
    # 将验证和测试回调函数添加到回调列表中
    callbacks = [validation_callback, test_callback]

    # 如果存在多个验证工作负载
    if len(experiment.multi_validation_wl) > 0:
        # 创建多验证环境的归一化向量环境
        callback_multi_validation_env = VecNormalize(
            DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION, experiment.multi_validation_wl)]),
            norm_obs=True,
            norm_reward=False,
            gamma=experiment.config["rl_algorithm"]["gamma"],
            training=False,
        )
        # 创建多验证回调函数
        multi_validation_callback = EvalCallbackWithTBRunningAverage(
            # 设置评估的回合数
            n_eval_episodes=len(experiment.multi_validation_wl),
            # 设置评估频率
            eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
            # 设置评估环境
            eval_env=callback_multi_validation_env,
            # 设置最佳模型保存路径
            best_model_save_path=experiment.experiment_folder_path,
            # 设置日志详细程度
            verbose=1,
            # 设置回调函数名称
            name="multi_validation",
            # 设置评估是否为确定性的
            deterministic=True,
            # 设置比较性能指标
            comparison_performances={},
        )
        # 将多验证回调函数添加到回调列表中
        callbacks.append(multi_validation_callback)

    # 开始实验学习
    experiment.start_learning()

    # 模型开始学习
    model.learn(
        # 设置总时间步数
        total_timesteps=experiment.config["timesteps"],
        # 设置回调函数列表
        callback=callbacks,
        # 设置TensorBoard日志名称
        tb_log_name=experiment.id,
        # 设置实验ID
        ids=experiment.config["id"]
    )
    # 结束实验学习
    experiment.finish_learning(
        # 传入训练环境
        training_env,
        # 传入验证回调函数的移动平均步数乘以并行环境数量
        validation_callback.moving_average_step * experiment.config["parallel_environments"],
        # 传入验证回调函数的最佳模型步数乘以并行环境数量
        validation_callback.best_model_step * experiment.config["parallel_environments"],
    )

    # 将工作负载字典保存到pickle文件中
    with open(f"{experiment.experiment_folder_path}/workload_dic.pickle", "wb") as handle:
        pickle.dump([training_env.venv.envs[0].dic, callbacks[0].eval_env.venv.envs[0].dic, callbacks[1].eval_env.venv.envs[0].dic], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # 完成自定义结束操作
    experiment.finishmy()

    # 打印结束信息
    print("结束")
