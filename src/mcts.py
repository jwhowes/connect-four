import torch
import numpy as np

from torch import LongTensor
from typing import Optional, List

from . import NUM_COLS
from .gym import State


EXPLORE_COEFF = np.sqrt(2)


class Node:
    def __init__(self, state: State):
        self.leaf: bool = True

        self.state: State = state

        self.winner: Optional[0 | 1 | 2] = self.state.winner()

        self.children: List[Optional[Node]] = [None for _ in range(NUM_COLS)]

        self.num_visits: LongTensor = torch.zeros(NUM_COLS, dtype=torch.long)
        self.num_wins: LongTensor = torch.zeros(NUM_COLS, dtype=torch.long)

    def rollout(self) -> 0 | 1 | 2:
        self.leaf = False
        if self.winner is not None:
            return self.winner

        child_idx = np.random.choice(torch.where(~self.state.illegal_moves())[0])

        self.num_visits[child_idx] = 1
        self.children[child_idx] = Node(
            self.state.step(child_idx)
        )

        winner = self.children[child_idx].rollout()

        self.num_wins[child_idx] += int(winner == self.state.player)
        return winner

    def search(self, parent_visits: int = 1) -> 0 | 1 | 2:
        if self.leaf:
            return self.rollout()

        quality = (
            torch.nan_to_num(self.num_wins / self.num_visits, nan=0.0) +
            EXPLORE_COEFF * torch.sqrt(np.log(parent_visits) / (1 + self.num_visits))
        )
        quality[self.state.illegal_moves()] = float('-inf')

        child_idx = quality.argmax()

        self.num_visits[child_idx] += 1

        if self.children[child_idx] is None:
            self.children[child_idx] = Node(
                self.state.step(child_idx)
            )

        winner = self.children[child_idx].search(self.num_visits[child_idx])

        self.num_wins[child_idx] += int(winner == self.state.player)

        if winner == 0 or self.state.winner is not None:
            return winner

        return 3 - winner


class MCTS:
    def __init__(self, sims_per_move: int = 1000):
        self.sims_per_move: int = sims_per_move

        self.root: Node = Node(State.initial())
        self.root_visits: int = 1

    def get_move(self) -> int:
        self.run_simulations()

        return self.root.num_visits.argmax()

    def run_simulations(self):
        for _ in range(self.sims_per_move):
            self.root.search(self.root_visits)

    def step(self, action: int):
        if self.root.children[action] is None:
            self.root_visits = 1
            self.root = Node(self.root.state.step(action))
        else:
            self.root_visits = self.root.num_visits[action]
            self.root = self.root.children[action]
