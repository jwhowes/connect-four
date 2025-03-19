from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Type, List

import torch
import numpy as np
from torch import Tensor

from ..gym import Board

# TODO
#   - Run multiple asynchronous workers with virtual loss
#   - Heuristic runs in a separate, asynchronous process
#   - For alpha zero we have two processes: model and sim
#   - The model process continuously takes a batch from the queue, processes it and returns the result to the sim thread
#   - The sim thread consists of asynchronous workers. Each worker spawns and executes a simulation (using virtual loss)
#     until it reaches the forward pass. At this point it adds its board to the queue (awaiting heuristic) and another
#     thread runs while it's waiting. Importantly, we resume threads in a FIFO manner.


class AbstractMCTSNode(ABC):
    def __init__(self, player: 1 | 2, parent: Optional[AbstractMCTSNode] = None, action: Optional[int] = None):
        self.player = player
        self.parent = parent
        self.action = action

        self.total_value = torch.zeros(7, dtype=torch.float32)
        self.num_visits = torch.zeros(7, dtype=torch.long)

        self.children: List[Optional[AbstractMCTSNode]] = [
            None for _ in range(7)
        ]
        self.expanded = torch.zeros(7, dtype=torch.bool)


class AbstractMCTS(ABC):
    Node: Type[AbstractMCTSNode]

    def __init__(self):
        self.node_visits: int = 1
        self.board: Board = Board.initial()
        self.root = self.Node(player=1)

    @staticmethod
    @abstractmethod
    def expansion_value(node: Node, node_visits: int) -> Tensor:
        ...

    @abstractmethod
    def heuristic(self, board: Board):
        ...

    @staticmethod
    @abstractmethod
    def backtrack_value(node: Node, heuristic_value, action: int):
        ...

    @abstractmethod
    def policy(self, temperature: float = 1.0) -> Tensor:
        ...

    def run_sim(self):
        board = self.board.clone()

        node_visits = self.node_visits
        node = self.root

        while node.expanded.all():
            value = self.expansion_value(node, node_visits)
            value[~board.legal_moves] = float('-inf')

            child_idx = value.argmax().item()
            board = board.step(child_idx)

            node_visits = node.num_visits[child_idx]
            node = node.children[child_idx]

        child_idx = np.random.choice(torch.where(~node.expanded)[0])
        board = board.step(child_idx)
        node.expanded[child_idx] = True
        node.children[child_idx] = self.Node(player=board.player, parent=node, action=child_idx)

        node.num_visits[child_idx] += 1
        heuristic_value = self.heuristic(board)
        self.backtrack_value(node, heuristic_value, child_idx)

        node = node.parent
        while node is not None:
            self.backtrack_value(node, heuristic_value, child_idx)
            node.num_visits[child_idx] += 1

            child_idx = node.action
            node = node.parent
