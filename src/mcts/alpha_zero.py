from math import sqrt

from torch import Tensor

from ..model import ModelConfig, Agent

from .UCT import UCT
from ..gym import Board

EXPLORATION_COEFF = sqrt(2.0)


class AlphaZero(UCT):
    model: Agent

    def __init__(self, config: ModelConfig):
        super(AlphaZero, self).__init__()

        self.model = config.build()

    def heuristic(self, board: Board) -> Tensor:
        return self.model(
            board.tensor().unsqueeze(0)
        ).squeeze(0)

    @staticmethod
    def backtrack_value(node: UCT.Node, heuristic_value: Tensor, action: int):
        if node.player == 1:
            node.total_value[action] += heuristic_value
        else:
            node.total_value[action] -= heuristic_value
