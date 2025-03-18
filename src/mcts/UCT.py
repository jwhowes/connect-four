from math import sqrt
from typing import List, Optional

import torch
import numpy as np
from tqdm import tqdm

from ..gym import Board

from .base import AbstractMCTS

EXPLORATION_COEFF = sqrt(2.0)


class UCTNode:
    def __init__(self, board: Board):
        self.board = board

        self.total_value = torch.zeros(7, dtype=torch.float32)
        self.num_simulations = torch.zeros(7, dtype=torch.long)

        self.children: List[Optional[UCTNode]] = [
            None for _ in range(7)
        ]
        self.empty = torch.ones(7, dtype=torch.bool)

    @property
    def is_leaf(self):
        return self.empty.any().bool()

    @staticmethod
    def rollout(board: Board) -> 0 | 1 | 2:
        if board.winner is not None:
            return board.winner

        action = np.random.choice(torch.where(board.legal_moves)[0])
        return UCTNode.rollout(
            board.step(action)
        )

    def run_simulation(self, parent_sims: int = 1):
        if self.is_leaf:
            child_idx = np.random.choice(torch.where(self.empty)[0])

            self.empty[child_idx] = False
            self.children[child_idx] = UCTNode(
                board=self.board.step(child_idx)
            )

            rollout_winner = self.rollout(self.children[child_idx].board)

            self.total_value[child_idx] += 0 if rollout_winner == 0 else 2 * (rollout_winner == self.board.player) - 1
            self.num_simulations[child_idx] += 1

            return rollout_winner

        value = (
            (self.total_value / self.num_simulations) +
            EXPLORATION_COEFF * torch.sqrt(np.log(parent_sims) / self.num_simulations)
        )
        value[~self.board.legal_moves] = float('-inf')

        child_idx = value.argmax()

        simulation_winner = self.children[child_idx].run_simulation(self.num_simulations[child_idx].item())
        self.num_simulations[child_idx] += 1
        self.total_value[child_idx] += 0 if simulation_winner == 0 else 2 * (simulation_winner == self.board.player) - 1

        return simulation_winner


class UCTMCTS(AbstractMCTS):
    def __init__(self):
        self.parent_sims: int = 1
        self.root: UCTNode = UCTNode(board=Board.initial())

    @property
    def board(self) -> Board:
        return self.root.board

    def run_sim(self):
        self.root.run_simulation(self.parent_sims)

    def get_move(self) -> int:
        return self.root.num_simulations.argmax().item()

    def step(self, action: int):
        if self.root.children[action] is not None:
            self.parent_sims = self.root.num_simulations[action]
            self.root = self.root.children[action]
        else:
            self.parent_sims = 1
            self.root = UCTNode(board=self.root.board.step(action))
