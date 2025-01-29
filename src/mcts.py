from __future__ import annotations

import torch
import numpy as np

from torch import LongTensor
from typing import Optional, List

from . import NUM_COLS
from .gym import State


EXPLORE_COEFF = np.sqrt(2)


class Node:
    def __init__(self, state: State, parent: Optional[Node] = None, action: int = -1):
        self.leaf: bool = True

        self.state: State = state
        self.action: int = action
        self.parent: Optional[Node] = parent

        self.winner: Optional[0 | 1 | 2] = self.state.winner()

        self.children: List[Optional[Node]] = [None for _ in range(NUM_COLS)]

        self.num_visits: LongTensor = torch.zeros(NUM_COLS, dtype=torch.long)
        self.value: LongTensor = torch.zeros(NUM_COLS, dtype=torch.long)


class MCTS:
    def __init__(self, sims_per_move: int = 1000):
        self.sims_per_move: int = sims_per_move

        self.root: Node = Node(State.initial())
        self.root_visits: int = 1

    def get_move(self) -> int:
        self.run_simulations()

        return self.root.num_visits.argmax()

    @staticmethod
    def rollout(node: Node) -> 0 | 1 | 2:
        node.leaf = False
        if node.winner is not None:
            return node.winner

        state = node.state
        winner = state.winner()
        while winner is None:
            child_idx = np.random.choice(torch.where(~state.illegal_moves())[0])
            state = state.step(child_idx)
            winner = state.winner()

        return winner

    def search(self):
        parent_visits = self.root_visits
        node = self.root

        while not node.leaf:
            quality = (
                    torch.nan_to_num(node.value / node.num_visits, nan=0.0) +
                    EXPLORE_COEFF * torch.sqrt(np.log(parent_visits) / (1 + node.num_visits))
            )
            quality[node.state.illegal_moves()] = float('-inf')

            child_idx = quality.argmax()
            node.num_visits[child_idx] += 1

            if node.children[child_idx] is None:
                node.children[child_idx] = Node(
                    state=node.state.step(child_idx),
                    parent=node,
                    action=child_idx
                )

            parent_visits = node.num_visits[child_idx]
            node = node.children[child_idx]

        winner = MCTS.rollout(node)

        while node.parent is not None:
            action = node.action
            node = node.parent

            node.value[action] += 2 * int(winner == node.state.player) - 1

    def run_simulations(self):
        for _ in range(self.sims_per_move):
            self.search()
            # self.root.search(self.root_visits)

    def step(self, action: int):
        if self.root.children[action] is None:
            self.root_visits = 1
            self.root = Node(self.root.state.step(action))
        else:
            self.root_visits = self.root.num_visits[action]
            self.root = self.root.children[action]
            self.root.parent = None
