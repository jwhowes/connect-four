from __future__ import annotations

from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from torch import FloatTensor, LongTensor

from . import NUM_COLS
from .gym import State
from .model import BaseModel

EXPLORE_COEFF = 5.0


@dataclass
class GameHistory:
    boards: LongTensor
    players: LongTensor
    priors: FloatTensor
    winner: int

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
        self.value: FloatTensor = torch.zeros(NUM_COLS, dtype=torch.float32)
        self.prior: FloatTensor = torch.zeros(NUM_COLS, dtype=torch.float32)
        self.num_visits: LongTensor = torch.zeros(NUM_COLS, dtype=torch.long)
        self.children: List[Optional[Node]] = [None for _ in range(NUM_COLS)]


class MCTS:
    def __init__(self):
        self.root: Node = Node(State.initial())
        self.root_visits: int = 1

    def policy(self, temperature: float = 1.0):
        likelihood = self.root.num_visits ** (1 / temperature)

        return likelihood / likelihood.sum()

    @staticmethod
    def self_play(sims_per_move: int, model: BaseModel, temperature: float = 1.0) -> GameHistory:
        model.eval()
        model.requires_grad_(False)

        mcts = MCTS()

        boards: List[LongTensor] = []
        players: List[int] = []
        priors: List[FloatTensor] = []
        while mcts.root.winner is None:
            for _ in range(sims_per_move):
                mcts.search(model)

            prior = mcts.policy(temperature)

            boards.append(mcts.root.state.board)
            players.append(mcts.root.state.player)
            priors.append(prior)

            action = torch.multinomial(prior, 1)[0]
            mcts.step(action)

        return GameHistory(
            boards=torch.stack(boards),
            players=torch.tensor(players, dtype=torch.long),
            priors=torch.stack(priors),
            winner=mcts.root.winner
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
                torch.nan_to_num(node.value / node.num_visits, nan=0.0) +
                EXPLORE_COEFF * node.prior * np.sqrt(parent_visits) / (1 + node.num_visits)
            )
            search_objective[node.state.illegal_moves()] = float('-inf')

            child_idx = search_objective.argmax()

            node.num_visits[child_idx] += 1
            parent_visits = node.num_visits[child_idx]

            if node.children[child_idx] is None:
                node.children[child_idx] = Node(node.state.step(child_idx), node, child_idx)

            node = node.children[child_idx]

        if node.winner is not None:
            winner = torch.zeros(3, dtype=torch.float32)
            winner[node.winner] = 1.0
        else:
            node.leaf = False
            board = F.one_hot(node.state.board.unsqueeze(0), num_classes=3).to(torch.float32)
            if node.state.player != 1:
                board = board[:, :, :, [0, 2, 1]]

            winner, prior = model(board)
            winner = winner.squeeze(0)
            if node.state.player != 1:
                winner = winner[[0, 2, 1]]

            winner = F.softmax(winner, dim=-1)
            node.prior = F.softmax(prior.squeeze(0), dim=-1)

        while node.parent is not None:
            action = node.action
            node = node.parent

            node.value[action] += (winner[node.state.player] - winner[3 - node.state.player])
