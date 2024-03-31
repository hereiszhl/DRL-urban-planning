import logging
import math
import copy
from pprint import pprint
from typing import Tuple, Dict, List, Text, Callable
from functools import partial

import numpy as np
from geopandas import GeoDataFrame
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from urban_planning.envs.plan_client import PlanClient
from urban_planning.envs.observation_extractor import ObservationExtractor
from urban_planning.envs import city_config
from urban_planning.utils.config import Config


class InfeasibleActionError(ValueError):  # 构建一个报错函数，出错时会返回报错相关信息
    """An infeasible action were passed to the env."""

    def __init__(self, action, mask):
        """Initialize an infeasible action error.

        Args:
          action: Infeasible action that was performed.
          mask: The mask associated with the current observation. mask[action] is
            `0` for infeasible actions.
        """
        super().__init__(self, action, mask)
        self.action = action
        self.mask = mask

    def __str__(self):
        return 'Infeasible action ({}) when the mask is ({})'.format(self.action, self.mask)


def reward_info_function( # 根据name的不同设置各项奖励的权重
    plc: PlanClient,
    name: Text, 
    road_network_weight: float = 1.0,
    life_circle_weight: float = 1.0,
    greenness_weight: float = 1.0,
    concept_weight: float = 0.0,
    weight_by_area: bool = False) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greenness_weight: Weight of greenness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        weight_by_area: Whether to weight the life circle reward by the area of residential zones.

    Returns:
        The RL reward.
        Info dictionary.
    """
    # 函数根据指定的权重和组成部分计算一个代理奖励。
    # 初始化一个代理奖励 proxy_reward，默认为 CityEnv.INTERMEDIATE_REWARD
    proxy_reward = CityEnv.INTERMEDIATE_REWARD

    # 根据 name 的不同，计算不同类型的奖励
    if name == 'intermediate':  #Intermediate 奖励：每个奖励都考虑。返回一个固定的代理奖励。
        return proxy_reward, {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greenness': -1.0,
            'concept': -1.0,
        }
    elif name == 'road':  # 只考虑道路奖励
        proxy_reward = 0.0 # 初始化 proxy_reward 为 0.0，表示初始代理奖励
        road_network = -1.0 # 初始化 road_network 为 -1.0，表示初始道路网络奖励
        road_network_info = dict() # 创建一个空的字典 road_network_info，用于存储道路网络信息
        if road_network_weight > 0.0:
            # 调用 plc.get_road_network_reward()，计算道路网络奖励，并将结果赋给 road_network
            road_network, road_network_info = plc.get_road_network_reward()
            # 更新代理奖励：proxy_reward += road_network_weight * road_network
            proxy_reward += road_network_weight * road_network
        return proxy_reward, {  # 返回一个包含以下信息的元组
            'road_network': road_network,
            'life_circle': -1.0, # 'life_circle'：生活圈（设置为 -1.0，因为这里不涉及生活圈奖励，下列同理）
            'greenness': -1.0, # 确保在计算代理奖励时，特定奖励不会对其产生积极影响
            'concept': -1.0,
            'road_network_info': road_network_info # 道路网络信息
        }
    elif name == 'land_use': # 只考虑土地利用奖励
        proxy_reward = 0.0 # 初始化 proxy_reward 为 0.0，表示初始代理奖励
        life_circle = -1.0 # 初始化 life_circle 为 -1.0，表示初始道路网络奖励
        greenness = -1.0 # 初始化 greenness 为 -1.0，表示初始道路网络奖励
        concept = -1.0 # 初始化 concept 为 -1.0，表示初始道路网络奖励

        life_circle_info = dict() # 创建一个空的字典 life_circle_info，用于存储生活圈配置
        if life_circle_weight > 0.0:
            # 调用 plc.get_life_circle_reward（），计算生活圈配置奖励，并将结果赋给 life_circle
            life_circle, life_circle_info = plc.get_life_circle_reward(weight_by_area=weight_by_area)
            # 更新代理奖励：proxy_reward += life_circle_weight * life_circle
            proxy_reward += life_circle_weight * life_circle

        if greenness_weight > 0.0:
            # 调用 plc.get_greenness_reward()，计算绿地覆盖率奖励，并将结果赋给 greenness
            greenness = plc.get_greenness_reward()
            proxy_reward += greenness_weight * greenness

        concept_info = dict()  # 创建一个空的字典 concept_info，用于存储与规划概念相关的信息
        if concept_weight > 0.0:
            # 调用 plc.get_concept_reward()，计算规划概念奖励，并将结果赋给 concept
            concept, concept_info = plc.get_concept_reward()
            # 更新代理奖励：proxy_reward += concept_weight * concept
            proxy_reward += concept_weight * concept

        return proxy_reward, {
            'road_network': -1.0, # 道路网络奖励（设置为 -1.0，因为在这里不涉及道路网络奖励
            'life_circle': life_circle,
            'greenness': greenness,
            'concept': concept,
            'life_circle_info': life_circle_info,
            'concept_info': concept_info
        }
    else:
        raise ValueError('Invalid state.')


class CityEnv: # 定义了一个名为 CityEnv 的类，用于模拟城市规划的环境
    """ Environment for urban planning."""
    # 在城市规划环境中的某些情况下，失败的奖励值。在这里，它被设置为 -1.0。如果在规划过程中出现错误或不良决策，可以使用这个奖励值
    FAILURE_REWARD = -1.0
    # 表示中间状态下的奖励值。在这里，它被设置为 0.0。中间状态可能指的是规划过程中的某些中间步骤，而不是最终结果
    INTERMEDIATE_REWARD = 0.0

    # 用于初始化 CityEnv 类的实例
    def __init__(self,
                 cfg: Config, # 一个名为 Config 的对象，用于配置城市规划环境
                 is_eval: bool = False, # 一个布尔值，表示是否处于评估模式。默认为 False
                 # 一个可调用对象，用于计算奖励和提供信息。默认为 reward_info_function 函数。
                 reward_info_fn:
                 Callable[[PlanClient, Text, float, float, float, bool], Tuple[float, Dict]] = reward_info_function):
        self.cfg = cfg # 将传入的配置对象赋值给实例变量cfg
        self._is_eval = is_eval # 将评估模式标志赋值给实例变量 _is_eval
        self._frozen = False # 初始化冻结状态标志为 False
        self._action_history = [] # 初始化动作历史为空列表
        self._plc = PlanClient(cfg.objectives_plan, cfg.init_plan) # 创建一个 PlanClient 对象，用于与规划环境交互。

        # 初始化奖励信息函数，使用 partial 函数将 reward_info_fn 与配置中的权重参数绑定
        self._reward_info_fn = partial(reward_info_fn,
                                       road_network_weight=cfg.reward_specs.get('road_network_weight', 1.0),
                                       life_circle_weight=cfg.reward_specs.get('life_circle_weight', 1.0),
                                       greenness_weight=cfg.reward_specs.get('greenness_weight', 1.0),
                                       concept_weight=cfg.reward_specs.get('concept_weight', 0.0),
                                       weight_by_area=cfg.reward_specs.get('weight_by_area', False))

        self._all_stages = ['land_use', 'road', 'done'] # ['land_use', 'road', 'done']：定义所有阶段的列表
        self._set_stage() # 设置当前阶段
        self._done = False # 初始化完成标志为 False
        self._set_cached_reward_info() # 设置缓存的奖励信息
        self._observation_extractor = ObservationExtractor(self._plc,
                                                           self.cfg.state_encoder_specs['max_num_nodes'],
                                                           self.cfg.state_encoder_specs['max_num_edges'],
                                                           len(self._all_stages)) #创建一个观测数据提取器，用于从规划环境中提取状态信息

    def _set_stage(self): # 用于设置环境的阶段
        """
        Set the stage.
        """
        self._land_use_steps = 0 # 初始化土地利用步数为 0
        self._road_steps = 0 # 初始化道路步数为 0
        if not self.cfg.skip_land_use: # 如果不跳过土地利用阶段
            self._stage = 'land_use' # 设置当前阶段为 'land_use'
            self._land_use_done = False # 将土地利用完成标志 _land_use_done 设置为 False
            self._road_done = False # 将道路完成标志 _road_done 设置为 False
        elif not self.cfg.skip_road: # 如果不跳过道路阶段
            self._stage = 'road' # 设置当前阶段为 'road'
            self._land_use_done = True # 将土地利用完成标志 _land_use_done 设置为 True
            self._road_done = False # 将道路完成标志 _road_done 设置为 False
        else:
            raise ValueError('Invalid stage. Land_use step and road step both reached max steps.')

    def _compute_total_road_steps(self) -> None: # 用于根据当前阶段和道路掩码计算总的道路步数
        """
        Compute the total number of road steps.
        """
        if self._stage == 'road' and self._road_steps == 0: # 如果当前阶段是 'road'，且道路步数为 0
            self._total_road_steps = math.floor(np.count_nonzero(self._current_road_mask)*self.cfg.road_ratio) # 进行步数的计算
        else:
            raise ValueError('Invalid stage.')

    def _set_cached_reward_info(self): # 用于缓存奖励信息
        """
        Set the cached reward.
        """
        if not self._frozen: # 如果当前状态不是冻结状态（not self._frozen）
            self._cached_life_circle_reward = -1.0 # 初始化缓存的生活圈奖励为 -1.0（下同）
            self._cached_greenness_reward = -1.0
            self._cached_concept_reward = -1.0

            self._cached_life_circle_info = dict() # 创建字典用于储存信息
            self._cached_concept_info = dict()

            self._cached_land_use_reward = -1.0 # 初始化缓存的土地利用奖励为 -1.0
            self._cached_land_use_gdf = self.snapshot_land_use() # 获取当前土地利用情况

    def freeze_land_use(self, info: Dict): # 用于将土地利用状态设置为“冻结”，并更新相应的奖励值和信息
        """
        Freeze the land use.
        """
        land_use_gdf = info['land_use_gdf'] # 从传入的字典info获取以下信息
        self._plc.freeze_land_use(land_use_gdf) # 将 land_use_gdf 设置为冻结的土地利用地理数据库
        self._cached_land_use_gdf = land_use_gdf # 更新土地利用数据
        self._cached_land_use_reward = info['land_use_reward'] # 更新土地利用奖励值
        self._cached_life_circle_reward = info['life_circle'] # 更新生活圈奖励值
        self._cached_greenness_reward = info['greenness'] # 更新绿地奖励值
        self._cached_concept_reward = info['concept'] # 更新规划概念奖励值
        self._cached_life_circle_info = info['life_circle_info'] # 更新生活圈信息
        self._cached_concept_info = info['concept_info'] # 更新规划概念信息
        self._frozen = True

    def get_reward_info(self) -> Tuple[float, Dict]: # 用于根据当前阶段和配置返回 RL 奖励和信息
        """
        Returns the RL reward and info.

        Returns:
            The RL reward.
            Info dictionary.
        """
        if self.cfg.skip_road: # 如果跳过了道路阶段
            if self._stage == 'land_use': # 如果当前阶段是 'land_use'，则返回中间状态的奖励信息
                return self._reward_info_fn(self._plc, 'intermediate')
            elif self._stage == 'done': # 如果当前阶段是 'done'，则返回土地利用阶段的奖励信息
                return self._reward_info_fn(self._plc, 'land_use')
            else:
                raise ValueError('Invalid stage.')
        elif self.cfg.skip_land_use: # 如果跳过了土地利用阶段
            if self._stage == 'road': # 如果当前阶段是 'road'，则返回中间状态的奖励信息
                return self._reward_info_fn(self._plc, 'intermediate')
            elif self._stage == 'done': # 如果当前阶段是 'done'，则返回道路阶段的奖励信息
                return self._reward_info_fn(self._plc, 'road')
            else:
                raise ValueError('Invalid stage.')
        else: # 如果都没跳过，则进行以下操作
            # 如果当前阶段是 'land_use' 或者是 'road' 且道路步数大于 0，则返回中间状态的奖励信息
            if (self._stage == 'land_use') or (self._stage == 'road' and self._road_steps > 0):
                return self._reward_info_fn(self._plc, 'intermediate')
            # 如果当前阶段是 'road' 且道路步数为 0，则返回土地利用阶段的奖励信息
            elif self._stage == 'road' and self._road_steps == 0:
                return self._reward_info_fn(self._plc, 'land_use')
            # 如果当前阶段是 'done'，则返回道路阶段的奖励信息
            elif self._stage == 'done':
                return self._reward_info_fn(self._plc, 'road')
            else:
                raise ValueError('Invalid stage.')

    def _get_all_reward_info(self) -> Tuple[float, Dict]: # 用于获取完整的奖励和信息，通常在加载计划时使用
        """
        Returns the entire reward and info. Used for loaded plans.
        """
        # 调用 self._reward_info_fn(self._plc, 'land_use')，计算土地利用奖励和信息，赋值给 land_use_reward 和 land_use_info
        land_use_reward, land_use_info = self._reward_info_fn(self._plc, 'land_use')
        # 调用 self._reward_info_fn(self._plc, 'road')，计算道路奖励和信息，给 road_reward 和 road_info
        road_reward, road_info = self._reward_info_fn(self._plc, 'road')
        reward = land_use_reward + road_reward # 计算总奖励 reward，将土地利用奖励和道路奖励相加
        info = {
            'road_network': road_info['road_network'],
            'life_circle': land_use_info['life_circle'],
            'greenness': land_use_info['greenness'],
            'road_network_info': road_info['road_network_info'],
            'life_circle_info': land_use_info['life_circle_info']
        } # 创建一个字典去记录以上的奖励信息
        return reward, info

    def eval(self): # 用于将环境设置为评估模式
        """
        Set the environment to eval mode.
        """
        self._is_eval = True

    def train(self): # 用于将环境设置为训练模式
        """
        Set the environment to training mode.
        """
        self._is_eval = False

    def get_numerical_feature_size(self): # 用于返回数值特征的大小
        """
        Returns the numerical feature size.

        Returns:
            feature_size (int): the feature size.
        """
        return self._observation_extractor.get_numerical_feature_size()

    def get_node_dim(self): # 用于返回节点维度
        """
        Returns the node dimension.

        Returns:
            node_dim (int): the node dimension.
        """
        dummy_land_use = self._get_dummy_land_use() # 创建一个名为 dummy_land_use 的虚拟土地利用数据
        return self._observation_extractor.get_node_dim(dummy_land_use)

    def _get_dummy_land_use(self): # 用于获取虚拟的土地利用信息
        """
        Returns the dummy land use.

        Returns:
            land_use (dictionary): the dummy land use.
        """
        dummy_land_use = dict() # 创建一个名为 dummy_land_use 的空字典
        dummy_land_use['type'] = city_config.FEASIBLE # 土地利用类型
        dummy_land_use['x'] = 0.5 #x和y表示土地利用的坐标位置
        dummy_land_use['y'] = 0.5
        dummy_land_use['area'] = 0.0 #土地利用的面积（下同）
        dummy_land_use['length'] = 0.0
        dummy_land_use['width'] = 0.0
        dummy_land_use['height'] = 0.0
        dummy_land_use['rect'] = 0.5
        dummy_land_use['eqi'] = 0.5
        dummy_land_use['sc'] = 0.5
        return dummy_land_use

    def _get_land_use_and_mask(self): # 用于获取当前土地利用信息和掩码
        """
        Returns the current land use and mask.

        Returns:
            land_use (dictionary): the current land use.
            mask (np.ndarray): the current mask.
        """
        if self._stage != 'land_use': # 如果当前阶段不是土地利用阶段
            land_use = self._get_dummy_land_use() # 创建一个名为 dummy_land_use 的虚拟土地利用数据，作为占位符
            # 创建一个全零数组 mask，大小为配置中的最大边数，数据类型为布尔型
            mask = np.zeros(self.cfg.state_encoder_specs['max_num_edges'], dtype=bool) 
        else: # 如果当前阶段是土地利用阶段
            # 调用实例变量 self._plc 的方法 get_current_land_use_and_mask()，获取当前土地利用信息和掩码
            land_use, mask = self._plc.get_current_land_use_and_mask()
        return land_use, mask

    def _get_road_mask(self): # 用于获取当前的道路掩码
        """
        Returns the current road mask.

        Returns:
            mask (np.ndarray): the current mask.
        """
        if self._stage == 'land_use': # 如果当前阶段是土地利用阶段
            # 创建一个全零数组 mask，大小为配置中的最大节点数，数据类型为布尔型
            mask = np.zeros(self.cfg.state_encoder_specs['max_num_nodes'], dtype=bool)
        else: # 如果当前阶段不是土地利用阶段
            # 调用实例变量 self._plc 的方法 get_current_road_mask()，获取当前的道路掩码
            mask = self._plc.get_current_road_mask()
        return mask

    def _get_stage_obs(self) -> int: # 获取当前阶段的观察值
        """
        Returns the current stage observation.

        Returns:
            obs (int): the current stage index.
        """
        return self._all_stages.index(self._stage)

    def _get_obs(self) -> List: # 用于获取观察值
        """
        Returns the observation.

        Returns:
            observation (object): the observation
        """
        # 调用实例变量 self._observation_extractor 的方法 get_obs()，传入当前土地利用、土地利用掩码、道路掩码和当前阶段的观察值。
        return self._observation_extractor.get_obs(self._current_land_use,
                                                   self._current_land_use_mask,
                                                   self._current_road_mask,
                                                   self._get_stage_obs())

    def place_land_use(self, land_use: Dict, action: int): # 用于放置土地利用
        """
        Places the land use.

        Args:
            land_use (dictionary): the land use.
            action (int): the action.
        """
        # 调用实例变量 self._plc 的方法 place_land_use(land_use, action)，传入土地利用信息和动作
        self._plc.place_land_use(land_use, action)

    def build_road(self, action: int): # 用于建造道路
        """
        Builds the road.

        Args:
            action (int): the action.
        """
        self._plc.build_road(action)

   # # 用于填充剩余的空间。具体来说，它可能会在城市规划中的某些情况下，将剩余的土地利用空间填充起来，以确保有效利用整个区域
    def fill_leftover(self): 
        """
        Fill the leftover space.
        """
        self._plc.fill_leftover()

   # 用于获取土地利用的快照。在城市规划过程中，可以使用这个方法来记录当前的土地利用状态，以备后续参考或分析
    def snapshot_land_use(self):
        """
        Snapshot the land use.
        """
        return self._plc.snapshot()

   # 用于建造所有的道路。在城市规划中，如果需要一次性建造所有道路，可以调用这个方法
    def build_all_road(self):
        """
        Build all the road.
        """
        self._plc.build_all_road()

    def transition_stage(self): # 用于在城市规划环境中进行阶段的过渡
        """
        Transition to the next stage.
        """
        if self._stage == 'land_use': # 如果当前阶段是土地利用阶段
            self._land_use_done = True # 将土地利用完成标志 _land_use_done 设置为 True
            if not self.cfg.skip_road: # 如果不跳过道路阶段
                self._stage = 'road' # 设置当前阶段为 'road'
            else: # 如果跳过了道路阶段
                self._road_done = True # 将道路完成标志 _road_done 设置为 True
                self._done = True # 将完成标志 _done 设置为 True
                self._stage = 'done' # 设置当前阶段为 'done'
        elif self._stage == 'road': # 如果当前阶段是道路阶段
            self._road_done = True # 将道路完成标志 _road_done 设置为 True
            self._done = True # 将完成标志 _done 设置为 True
            self._stage = 'done' # 设置当前阶段为 'done'
        else:
            raise ValueError('Unknown stage: {}'.format(self._stage))

    def failure_step(self, logging_str, logger): # 用于在发生失败步骤后进行日志记录和重置
        """
        Logging and reset after a failure step.
        """
        # # 记录日志，包括传入的日志字符串 logging_str 和动作历史 self._action_history
        logger.info('{}: {}'.format(logging_str, self._action_history)) 
        info = {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greenness': -1.0,
        }
        return self._get_obs(), self.FAILURE_REWARD, True, info

    # 用于执行一个时间步，根据动作更新环境状态，并返回观察值、奖励、完成标志和信息
    def step(self, action: np.ndarray, logger: logging.Logger) -> Tuple[List, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, you are responsible for calling `reset()` to reset
        the environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (np.ndarray of size 2): The action to take.
                                           1 is the land_use placement action.
                                           1 is the building road action.
            logger (Logger): The logger.

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # 检查是否已经完成了整个回合（self._done）。如果是，则抛出运行时错误，表示在回合结束后执行了动作。
        if self._done:
            raise RuntimeError('Action taken after episode is done.')

        if self._stage == 'land_use': # 如果当前阶段是土地利用阶段
            land_use = self._current_land_use # 获取当前土地利用信息 land_use
            action = int(action[0]) #  将动作转换为整数
            self._action_history.append((land_use, action)) # 土地利用信息和动作添加到动作历史中
            if self._current_land_use_mask[action] == 0: # 如果当前土地利用掩码中对应的动作不可行
                raise InfeasibleActionError(action, self._current_land_use_mask) # 抛出不可行动作错误

            try:
                self.place_land_use(land_use, action) # 尝试将土地利用放置在指定位置（self.place_land_use(land_use, action)）
            except ValueError as err: # 如果在放置土地利用时出现了 ValueError 异常
                logger.error(err) # 用于在发生失败步骤后进行日志记录和重置
                return self.failure_step('Actions took before failing to place land use', logger)
            except Exception as err: # 如果在放置土地利用时出现了其他类型的异常
                logger.error(err) # 会记录错误信息
                return self.failure_step('Actions took before failing to place land use', logger)

            self._land_use_steps += 1 # 将土地利用步数加一，用于跟踪在土地利用阶段执行的步数
            land_use_done = self._plc.is_land_use_done()
            if land_use_done: # 如果土地利用已完成
                self.fill_leftover() # 填充剩余空间
                self._cached_land_use_gdf = self.snapshot_land_use() # 获取土地利用的快照
                self.transition_stage() # 进行阶段过渡
            reward, info = self.get_reward_info() # 获取奖励和信息
            self._current_land_use, self._current_land_use_mask = self._get_land_use_and_mask() # 更新当前土地利用信息和掩码
            # 如果土地利用未完成且没有可行的土地利用掩码
            if not self._land_use_done and not np.any(self._current_land_use_mask): 
                return self.failure_step('Actions took before becoming infeasible', logger) # 返回失败步骤，记录日志
            self._current_road_mask = self._get_road_mask() # 更新当前道路掩码
            if self._stage != 'land_use': # 如果当前阶段不再是土地利用阶段
                self._cached_land_use_reward = reward # 更新缓存的土地利用奖励
                if self._stage == 'road': # 如果当前阶段是道路阶段
                    if not np.any(self._current_road_mask): # 如果没有可行的道路掩码
                        return self.failure_step('Actions took before becoming infeasible', logger) # 返回失败步骤，记录日志
                    # 更新缓存的生活圈奖励、绿地奖励和规划概念奖励
                    self._cached_life_circle_reward = info['life_circle']
                    self._cached_greenness_reward = info['greenness']
                    self._cached_concept_reward = info['concept']

                    # 更新缓存的生活圈信息和规划概念信息
                    self._cached_life_circle_info = info['life_circle_info']
                    self._cached_concept_info = info['concept_info']

                    # 计算总的道路步数
                    self._compute_total_road_steps()
        elif self._stage == 'road':
            action = int(action[1])
            self._action_history.append(('road', action))
            if self._current_road_mask[action] == 0:
                raise InfeasibleActionError(action, self._current_road_mask)

            try:
                self.build_road(action)
            except ValueError as err:
                logger.error(err)
                return self.failure_step('Actions took before failing to place land use', logger)
            except Exception as err:
                logger.error(err)
                return self.failure_step('Actions took before failing to place land use', logger)

            self._road_steps += 1
            if self._road_steps >= self._total_road_steps:
                self.transition_stage()
            reward, info = self.get_reward_info()
            self._current_land_use, self._current_land_use_mask = self._get_land_use_and_mask()
            self._current_road_mask = self._get_road_mask()
        else:
            raise ValueError('Cannot step in stage: {}.'.format(self._stage))

        if self._done:
            info['land_use_reward'] = self._cached_land_use_reward
            if not self.cfg.skip_road:
                info['life_circle'] = self._cached_life_circle_reward
                info['greenness'] = self._cached_greenness_reward
                info['concept'] = self._cached_concept_reward

                info['life_circle_info'] = self._cached_life_circle_info
                info['concept_info'] = self._cached_concept_info
            else:
                self.build_all_road()
            if self._is_eval:
                info['gdf'] = self._plc.get_gdf()
                info['land_use_gdf'] = self._cached_land_use_gdf

        return self._get_obs(), reward, self._done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation from the reset
        """
        self._plc.unplan_all_land_use()
        self._action_history = []
        self._set_stage()
        self._done = False
        self._set_cached_reward_info()
        self._current_land_use, self._current_land_use_mask = self._get_land_use_and_mask()
        self._current_road_mask = self._get_road_mask()
        if self.cfg.skip_land_use:
            self._compute_total_road_steps()
        return self._get_obs()

    @staticmethod
    def filter_land_use_road(gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Filter out the land use and road features.
        """
        land_use_road_gdf = copy.deepcopy(gdf[(gdf['existence'] == True) &
                                              (gdf['type'] != city_config.OUTSIDE) &
                                              (gdf['type'] != city_config.BOUNDARY) &
                                              (gdf['type'] != city_config.INTERSECTION)])
        return land_use_road_gdf

    @staticmethod
    def filter_road_boundary(gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Filter out the road and boundary features.
        """
        road_boundary_gdf = copy.deepcopy(gdf[(gdf['existence'] == True) &
                                              ((gdf['type'] == city_config.ROAD) |
                                               (gdf['type'] == city_config.BOUNDARY))])
        return road_boundary_gdf

    @staticmethod
    def _add_legend_to_gdf(gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Add legend to the gdf.
        """
        gdf['legend'] = gdf['type'].apply(lambda x: city_config.LAND_USE_ID_MAP_INV[x])
        return gdf

    @staticmethod
    def plot_and_save_gdf(gdf: GeoDataFrame, cmap: ListedColormap,
                          save_fig: bool = False, path: Text = None, legend: bool = False,
                          ticks: bool = True, bbox: bool = True) -> None:
        """
        Plot and save the gdf.
        """
        gdf = CityEnv._add_legend_to_gdf(gdf)
        gdf.plot(
            'legend',
            cmap=cmap,
            categorical=True,
            legend=legend,
            legend_kwds={'bbox_to_anchor': (1.8, 1)}
        )
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        if not bbox:
            plt.axis('off')
        if save_fig:
            assert path is not None
            plt.savefig(path, format='svg', transparent=True)
        plt.show()
        plt.close()

    def visualize(self, save_fig: bool = False, path: Text = None, legend: bool = True,
                  ticks: bool = True, bbox: bool = True) -> None:
        """
        Visualize the city plan.
        """
        gdf = self._plc.get_gdf()
        land_use_road_gdf = self.filter_land_use_road(gdf)
        existing_types = sorted([city_config.LAND_USE_ID_MAP_INV[var] for var in land_use_road_gdf['type'].unique()])
        cmap = ListedColormap(
            [city_config.TYPE_COLOR_MAP[var] for var in existing_types])
        self.plot_and_save_gdf(land_use_road_gdf, cmap, save_fig, path, legend, ticks, bbox)

    def visualize_road_and_boundary(self, save_fig: bool = False, path: Text = None, legend: bool = True,
                                    ticks: bool = True, bbox: bool = True) -> None:
        """
        Visualize the roads and boundaries.
        """
        gdf = self._plc.get_gdf()
        road_boundary_gdf = self.filter_road_boundary(gdf)
        existing_types = sorted([city_config.LAND_USE_ID_MAP_INV[var] for var in road_boundary_gdf['type'].unique()])
        cmap = ListedColormap(
            [city_config.TYPE_COLOR_MAP[var] for var in existing_types])
        self.plot_and_save_gdf(road_boundary_gdf, cmap, save_fig, path, legend, ticks, bbox)

    def load_plan(self, gdf: GeoDataFrame) -> None:
        """
        Load a city plan.
        """
        self._plc.load_plan(gdf)

    def score_plan(self, verbose=True) -> Tuple[float, Dict]:
        """
        Score the city plan.
        """
        reward, info = self._get_all_reward_info()
        if verbose:
            print(f'reward: {reward}')
            pprint(info, indent=4, sort_dicts=False)
        return reward, info

    def get_init_plan(self) -> Dict:
        """
        Get the gdf of the city plan.
        """
        return self._plc.get_init_plan()
