from __future__ import annotations

from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from torch import FloatTensor, LongTensor, BoolTensor

from . import NUM_COLS
from .gym import State
from .model import BaseModel

EXPLORE_COEFF = np.sqrt(2)


@dataclass
class GameHistory:
    boards: LongTensor
    players: LongTensor
    values: FloatTensor
    legals: BoolTensor

    def save(self, path: str):
        torch.save(
            self.__dict__, path
        )


class Node:
    def __init__(self, state: State, parent: Optional[Node] = None, action: int = -1):
        self.state = state
        self.parent = parent
        self.action = action

        self.winner: Optional[0 | 1 | 2] = state.winner()

        self.leaf = True
        self.value: FloatTensor = torch.zeros(NUM_COLS, dtype=torch.long)
        self.num_visits: LongTensor = torch.zeros(NUM_COLS, dtype=torch.long)
        self.children: List[Optional[Node]] = [None for _ in range(NUM_COLS)]

    @property
    def quality(self):
        if self.state.player == 1:
            return self.value

        return -self.value


class Search:
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

        self.root = Node(State.initial())
        self.root_visits: int = 1

    def policy(self, temperature: float = 1.0):
        likelihood = self.root.num_visits ** (1 / temperature)

        return likelihood / likelihood.sum()

    @staticmethod
    def self_play(sims_per_move: int, model: BaseModel, temperature: float = 1.0, gamma: float = 0.99):
        model.eval()
        model.requires_grad_(False)

        search = Search(gamma)

        boards: List[LongTensor] = []
        players: List[int] = []
        values: List[FloatTensor] = []
        legals: List[BoolTensor] = []
        while search.root.winner is None:
            for _ in range(sims_per_move):
                search.search(model)

            boards.append(search.root.state.board)
            players.append(search.root.state.player)
            values.append(search.root.value)
            legals.append(~search.root.state.illegal_moves())

            action = torch.multinomial(search.policy(temperature), 1)[0]
            search.step(action)

        return GameHistory(
            boards=torch.stack(boards),
            players=torch.tensor(players, dtype=torch.long),
            values=torch.stack(values),
            legals=torch.stack(legals)
        )

    def step(self, action: int):
        if self.root.children[action] is None:
            self.root_visits = 1
            self.root = Node(self.root.state.step(action))
        else:
            self.root_visits = self.root.num_visits[action]
            self.root = self.root.children[action]
            self.root.parent = None

    @torch.inference_mode()
    def search(self, model: BaseModel):
        parent_visits = self.root_visits
        node = self.root

        while not node.leaf:
            search_objective = (
                node.quality +
                EXPLORE_COEFF * np.log(parent_visits) / (1 + node.num_visits)
            )
            search_objective[node.state.illegal_moves()] = float('-inf')

            child_idx = search_objective.argmax()

            node.num_visits[child_idx] += 1
            parent_visits = node.num_visits[child_idx]

            if node.children[child_idx] is None:
                node.children[child_idx] = Node(node.state.step(child_idx), node, child_idx)

            node = node.children[child_idx]

        if node.winner is not None:
            if node.winner == 0:
                node.parent.value[node.action] = 0
            elif node.winner == 1:
                node.parent.value[node.action] = 1
            else:
                node.parent.value[node.action] = -1

            node = node.parent
        else:
            node.leaf = False
            board = F.one_hot(node.state.board.unsqueeze(0), num_classes=3).to(torch.float32)
            if node.state.player != 1:
                board = board[:, :, :, [0, 2, 1]]

            node.value = model(board).squeeze(0)
            if node.state.player != 1:
                node.value = -node.value

        while node.parent is not None:
            action = node.action
            node = node.parent

            if node.state.player != 1:
                node.value[action] = self.gamma * node.children[action].value.max()
            else:
                node.value[action] = self.gamma * node.children[action].value.min()
