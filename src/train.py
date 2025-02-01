from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import Value, Lock, Process

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import GameHistoryDataset
from .search import Search
from .model import BaseModelConfig, BaseModel
from .util import Config, timestamp


@dataclass
class TrainConfig(Config):
    queue_size: int = 16

    sims_per_move: int = 250
    temperature: float = 1.0
    gamma: float = 0.99

    batch_size: int = 16
    lr: float = 5e-5
    steps_per_cycle: int = 500


class Trainer:
    def __init__(
            self, queue_size: int, batch_size: int, lr: float, steps_per_cycle: int, sims_per_move: int,
            temperature: float, gamma: float,
            data_dir: str, model_dir: str, model_config: BaseModelConfig, resume: bool = False, data_workers: int = 4
    ):
        self.queue_size = queue_size

        self.sims_per_move = sims_per_move
        self.temperature = temperature
        self.gamma = gamma

        self.batch_size = batch_size
        self.lr = lr
        self.steps_per_cycle = steps_per_cycle

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
            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, f"{self.timestamp.value}.pt")
            )

    def load_model(self, model: BaseModel):
        model.load_state_dict(torch.load(os.path.join(self.model_dir, f"{self.timestamp.value}.pt"), weights_only=True))

    def save_model(self, model: BaseModel):
        os.remove(os.path.join(self.model_dir, f"{self.timestamp.value}.pt"))

        self.timestamp.value = timestamp()
        torch.save(model.state_dict(), os.path.join(self.model_dir, f"{self.timestamp.value}.pt"))

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

            history = Search.self_play(self.sims_per_move, model, self.temperature, self.gamma)

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

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, self.steps_per_cycle)

        with self.model_lock:
            self.load_model(model)

        while True:
            if len(os.listdir(self.data_dir)) == 0:
                time.sleep(1)
                continue

            with self.data_lock:
                dataset = GameHistoryDataset(self.data_dir)

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )

            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            total_loss = 0
            for i, (board, quality, legal) in pbar:
                opt.zero_grad()

                pred = model(board)
                loss = F.mse_loss(pred, quality, reduction='none')
                loss = loss[legal].mean()

                total_loss += loss.item()

                loss.backward()
                opt.step()
                lr_scheduler.step()

                pbar.set_description(f"Average Loss: {total_loss / (i + 1):.4f}")

            with self.model_lock:
                self.save_model(model)

            time.sleep(0.5)
