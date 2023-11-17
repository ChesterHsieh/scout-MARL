import random

from tensordict import tensordict
from collections import defaultdict
from typing import Optional
import numpy as np
import torch
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.data import TensorSpec, CompositeSpec, BinaryDiscreteTensorSpec
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)

from environment.ScoutConst import ActionType
from itertools import combinations


class ScoutGameEnv(EnvBase):
    def __init__(self, device='cpu'):
        super(ScoutGameEnv, self).__init__()

        # 动作类型
        action_type_spec = BinaryDiscreteTensorSpec()

        # 出牌参数
        play_cards_spec = TensorSpec(shape=(11,), dtype=torch.int64, minimum=0, maximum=1)

        # 換牌参数
        exchange_card_spec = CompositeSpec(
            exchange_choice=TensorSpec(shape=(), dtype=torch.int64, minimum=0, maximum=1),
            play_after_exchange=TensorSpec(shape=(12,), dtype=torch.int64, minimum=0, maximum=1)
        )

        # 整个动作空间
        action_space_spec = CompositeSpec(
            action_type=action_type_spec,
            play_action=play_cards_spec,
            exchange_action=exchange_card_spec
        )



    def _step(self, td: tensordict):
        """
        torchdict is a god's structure


        Parameters:
        action (tuple): 描述玩家動作的元組。例如，(action_type, cards)。

        Returns:
        observation, reward, done, info: 四元組，包含新的遊戲狀態、獎勵、遊戲是否結束及其他信息。
        """
        # 首先，檢查動作是否有效

        action = td['action']
        if not self.is_valid_action(action):
            raise ValueError("無效的動作")

        # 提取動作類型和卡片
        action_type, cards = action

        # 根據動作類型更新遊戲狀態
        if action_type == ActionType.PLAY:
            # 玩家打出一組卡片
            self.play_cards(cards)
        elif action_type == ActionType.SCOUT:
            # 玩家執行 "Scout" 動作
            self.scout_action(cards)

        # 檢查遊戲是否結束
        done = self.check_if_game_ends()

        # 計算獎勵
        reward = self.calculate_reward()

        # 返回新的遊戲狀態、獎勵、是否結束，以及其他信息
        return self.get_observation(), reward, done, {}

    def _reset(self, td, **kwargs):
        cards = list(combinations(range(1, 11), 2))
        cards.remove((9, 10))  # 移除 (9, 10) 組合

        random.shuffle(cards)
        player_hands = [[] for i in range(4)]
        for i in range(4):  # 四名玩家
            player_hands[i] = list(zip(*cards[i * 11:(i + 1) * 11]))

        init_td = TensorDict({}, batch_size=torch.Size())
        for agent in self

        init_td.set('observation',torch.tensor(self.state.flatten()), device = self.device))
        return td

    def _set_seed(self):
        pass
