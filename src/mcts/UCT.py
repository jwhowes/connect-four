from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from ..gym import Board

from .base import AbstractMCTSNode, AbstractMCTS

EXPLORATION_COEFF = sqrt(2.0)


class UCT(AbstractMCTS):
    class Node(AbstractMCTSNode):
        pass

    @staticmethod
    def heuristic(board: Board) -> 0 | 1 | 2:
        while board.winner is None:
            board = board.step(
                np.random.choice(torch.where(board.legal_moves)[0])
            )

        return board.winner

    @staticmethod
    def expansion_value(node: Node, node_visits: int) -> Tensor:
        return (
            (node.total_value / node.num_visits) +
            EXPLORATION_COEFF * torch.sqrt(np.log(node_visits) / node.num_visits)
        )

    @staticmethod
    def backtrack_value(node: Node, heuristic_value: 0 | 1 | 2, action: int):
        node.total_value[action] += 0 if heuristic_value == 0 else 2 * (node.player == heuristic_value) - 1

    def policy(self, temperature: float = 1.0) -> Tensor:
        return F.normalize(self.root.num_visits.pow(1.0 / temperature), dim=-1, p=1)
