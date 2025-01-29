from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import LongTensor, BoolTensor
from dataclasses import dataclass
from typing import Optional

from . import NUM_COLS, NUM_ROWS

H_kernel = torch.tensor([1, 1, 1, 1]).view(1, 1, 1, 4).expand(2, -1, -1, -1).contiguous()
V_kernel = H_kernel.transpose(2, 3).contiguous()

DL_kernel = torch.eye(4, dtype=torch.long).view(1, 1, 4, 4).expand(2, -1, -1, -1).contiguous()
DR_kernel = DL_kernel.flip(3).contiguous()


@dataclass
class State:
    board: LongTensor
    top: LongTensor
    player: 1 | 2

    def display(self):
        print("\n".join([
            "\t".join([
                str(p.item()) for p in row
            ]) for row in self.board.flip(0)
        ]))

    def step(self, action: int) -> State:
        board = self.board.clone()
        top = self.top.clone()

        board[top[action], action] = self.player

        top[action] += 1

        return State(
            board=board,
            player=3 - self.player,
            top=top
        )

    def winner(self) -> Optional[0 | 1 | 2]:
        if self.illegal_moves().all():
            return 0

        board = F.one_hot(self.board.unsqueeze(0), num_classes=3).permute(0, 3, 1, 2)[:, 1:]

        for kernel in [H_kernel, V_kernel, DL_kernel, DR_kernel]:
            m = F.conv2d(board, kernel, groups=2).amax((0, 2, 3))

            if m.max() >= 4:
                return int(m.argmax() + 1)

        return None

    @staticmethod
    def initial() -> State:
        return State(
            board=torch.zeros(NUM_ROWS, NUM_COLS, dtype=torch.long),
            top=torch.zeros(NUM_COLS, dtype=torch.long),
            player=1
        )

    def illegal_moves(self) -> BoolTensor:
        return self.top >= NUM_ROWS
