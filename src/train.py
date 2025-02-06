from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional
from multiprocessing import Value, Lock, Process

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import GameHistoryDataset
from .mcts import MCTS
from .model import BaseModelConfig, BaseModel
from .util import Config, timestamp


@dataclass
class TrainConfig(Config):
    queue_size: int = 16

    sims_per_move: int = 250
    temperature: float = 1.0
    p_random_choice: float = 0.0

    p_flip: float = 0.5

    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01


class Trainer:
    def __init__(
            self, train_config: TrainConfig,
            data_dir: str, model_dir: str, model_config: BaseModelConfig, resume: bool = False, data_workers: int = 4
    ):
        self.queue_size = train_config.queue_size

        self.train_config = train_config

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_config = model_config

        self.data_workers = data_workers

        self.model_lock = Lock()
        self.data_lock = Lock()

        self.timestamp = None
        if resume:
            for file in os.listdir(self.model_dir):
                if os.path.splitext(file)[1] == ".pt":
                    self.timestamp = Value('i', int(os.path.splitext(file)[0]))
                    break

            assert self.timestamp is not None, "No model found."
        else:
            model: BaseModel = self.model_config.build_model()

            to_remove = []
            for file in os.listdir(self.model_dir):
                if os.path.splitext(file)[1] == ".pt":
                    to_remove.append(os.path.join(self.model_dir, file))

            for file in to_remove:
                os.remove(file)

            for file in os.listdir(self.data_dir):
                os.remove(os.path.join(self.data_dir, file))

            self.timestamp = Value('i', timestamp())
            torch.save({
                "model": model.state_dict()
            },
                os.path.join(self.model_dir, f"{self.timestamp.value}.pt")
            )

    def load_model(self, model: BaseModel, opt: Optional[torch.optim.Optimizer] = None):
        state_dict = torch.load(os.path.join(self.model_dir, f"{self.timestamp.value}.pt"), weights_only=True)
        model.load_state_dict(state_dict["model"])
        if opt is not None and "opt" in state_dict:
            opt.load_state_dict(state_dict["opt"])

    def save_model(self, model: BaseModel, opt: torch.optim.Optimizer):
        os.remove(os.path.join(self.model_dir, f"{self.timestamp.value}.pt"))

        self.timestamp.value = timestamp()
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict()
        }, os.path.join(self.model_dir, f"{self.timestamp.value}.pt"))

    def train(self):
        train_worker = Process(target=self.train_worker)
        train_worker.start()

        data_workers = [
            Process(target=self.data_worker) for _ in range(self.data_workers)
        ]

        for worker in data_workers:
            worker.start()

        train_worker.join()

    def data_worker(self):
        model: BaseModel = self.model_config.build_model()
        model.eval()

        data_timestamp = 0
        while True:
            with self.model_lock:
                if data_timestamp != self.timestamp.value:
                    self.load_model(model)
                    data_timestamp = self.timestamp.value

            history = MCTS.self_play(
                self.train_config.sims_per_move, model, self.train_config.temperature,
                p_random_choice=self.train_config.p_random_choice,
            )

            with self.data_lock:
                files = sorted([
                    os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)
                ])
                if len(files) >= self.queue_size:
                    os.remove(files[0])

                history.save(os.path.join(self.data_dir, f"{timestamp()}.hist"))

    def train_worker(self):
        model: BaseModel = self.model_config.build_model()
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=self.train_config.lr, weight_decay=self.train_config.weight_decay)

        with self.model_lock:
            self.load_model(model, opt)

        while True:
            if len(os.listdir(self.data_dir)) == 0:
                time.sleep(1)
                continue

            with self.data_lock:
                dataset = GameHistoryDataset(self.data_dir, p_flip=self.train_config.p_flip)

            dataloader = DataLoader(
                dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                pin_memory=True
            )

            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            total_loss = 0
            for i, (board, winner) in pbar:
                opt.zero_grad()

                pred_winner = model(board)
                loss = F.cross_entropy(pred_winner, winner)

                total_loss += loss.item()

                loss.backward()
                opt.step()

                pbar.set_description(f"Average Loss: {total_loss / (i + 1):.4f}")

            with self.model_lock:
                self.save_model(model, opt)

            time.sleep(0.5)
