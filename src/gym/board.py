# Note:
#   A board is 7 columns x 6 rows. DO NOT CHANGE!

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass
class Board:
    player: 0 | 1
    board_p1: Tensor
    board_p2: Tensor
    top: Tensor
    winner: Optional[0 | 1]

    @staticmethod
    def initial() -> Board:
        return Board(
            player=0,
            board_p1=torch.zeros(size=(), dtype=torch.long),
            board_p2=torch.zeros(size=(), dtype=torch.long),
            top=torch.zeros(7, dtype=torch.long),
            winner=None
        )

    @staticmethod
    def horizontal_winner(board, action) -> bool:
        shift = action - 3

        if shift < 0:
            masked = (board << -shift) & 0x7f
        else:
            masked = (board >> shift) & 0x7f

        return (
                masked & 0x78 == 0x78 or
                masked & 0x3c == 0x3c or
                masked & 0x1e == 0x1e or
                masked & 0xf == 0xf
        )

    @staticmethod
    def vertical_winner(board, action) -> bool:
        shift = action - 21

        if shift < 0:
            masked = (board << -shift) & 0x40810204081
        else:
            masked = (board >> shift) & 0x40810204081

        return (
            masked & 0x40810200000 == 0x40810200000 or
            masked & 0x810204000 == 0x810204000 or
            masked & 0x10204080 == 0x10204080 or
            masked & 0x204081 == 0x204081
        )

    @staticmethod
    def up_right_winner(board, action) -> bool:
        return False  # TODO

    @staticmethod
    def up_left_winner(board, action) -> bool:
        return False  # TODO

    @staticmethod
    def check_winner(board, action) -> bool:
        return (
                Board.horizontal_winner(board, action) or
                Board.vertical_winner(board, action) or
                Board.up_right_winner(board, action) or
                Board.up_left_winner(board, action)
        )

    def step(self, action: int) -> Board:
        clone_p1 = self.board_p1.clone()
        clone_p2 = self.board_p2.clone()
        clone_top = self.top.clone()
        if self.player == 0:
            clone_p1 |= 1 << (action + clone_top[action] * 7)

            winner = self.check_winner(clone_p1, action)
        else:
            clone_p2 |= 1 << (action + clone_top[action] * 7)
            winner = self.check_winner(clone_p2, action)

        return Board(
            player=1 - self.player if not winner else self.player,
            board_p1=clone_p1,
            board_p2=clone_p2,
            top=clone_top,
            winner=self.player if winner else None
        )

    @property
    def legal_moves(self) -> Tensor:
        return self.top < 6
