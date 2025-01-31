from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor, FloatTensor

from . import NUM_COLS
from .gym import State
from .model import BaseModel

EXPLORE_COEFF: float = 5.0


class Node:
    def __init__(self, state: State, parent: Optional[Node] = None, action: int = -1):
        self.state: State = state
        self.action: int = action
        self.parent: Optional[Node] = parent

        self.winner: Optional[0 | 1 | 2] = self.state.winner()

        self.leaf = True

        self.children: List[Optional[Node]] = [None for _ in range(NUM_COLS)]
        self.prior: Optional[FloatTensor] = None

        self.num_visits: LongTensor = torch.zeros(NUM_COLS, dtype=torch.long)
        self.value: FloatTensor = torch.zeros(NUM_COLS, dtype=torch.float32)


@dataclass
class GameHistory:
    boards: LongTensor
    players: LongTensor

    mcts_probs: FloatTensor
    winner: int

    def save(self, path: str):
        torch.save(
            self.__dict__, path
        )


class MCTS:
    def __init__(self):
        self.root: Node = Node(State.initial())
        self.root_visits: int = 1

    def policy(self, temperature: float) -> FloatTensor:
        likelihood = self.root.num_visits ** (1.0 / temperature)

        return F.normalize(likelihood, dim=-1, p=1)

    @staticmethod
    def rollout(node: Node) -> 0 | 1 | 2:
        if node.winner is not None:
            return node.winner

        node.leaf = False

        state = node.state
        winner = state.winner()
        while winner is None:
            child_idx = np.random.choice(torch.where(~state.illegal_moves())[0])
            state = state.step(child_idx)
            winner = state.winner()

        return winner

    @torch.inference_mode()
    def search(self, model: BaseModel):
        node = self.root

        while not node.leaf:
            quality = (
                    torch.nan_to_num(node.value / node.num_visits, nan=0.0) +
                    EXPLORE_COEFF * node.prior * torch.sqrt(node.num_visits.sum()) / (1 + node.num_visits)
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

            node = node.children[child_idx]

        board = F.one_hot(node.state.board.unsqueeze(0), num_classes=3).to(torch.float32)
        if node.state.player != 1:
            board = board[:, :, :, [0, 2, 1]]

        prior = model(board)
        prior = F.softmax(prior, dim=-1).squeeze(0)

        node.prior = prior
        rollout_winner = MCTS.rollout(node)

        while node.parent is not None:
            action = node.action
            node = node.parent

            node.value[action] += int(rollout_winner == node.state.player) - int(rollout_winner == 3 - node.state.player)

    @staticmethod
    def self_play(sims_per_move: int, model: BaseModel, temperature: float = 1.0) -> GameHistory:
        model.eval()
        model.requires_grad_(False)

        mcts = MCTS()

        players: List[int] = []
        boards: List[LongTensor] = []
        mcts_probs: List[FloatTensor] = []
        while mcts.root.winner is None:
            for _ in range(sims_per_move):
                mcts.search(model)

            mcts_prob = mcts.policy(temperature)

            mcts_probs.append(mcts_prob)
            boards.append(mcts.root.state.board)
            players.append(mcts.root.state.player)

            action = torch.multinomial(mcts_prob, 1)[0]
            mcts.step(action)

        return GameHistory(
            boards=torch.stack(boards),
            players=torch.tensor(players, dtype=torch.long),
            mcts_probs=torch.stack(mcts_probs),
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
