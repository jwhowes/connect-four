import os
import warnings
from typing import Optional

import click

from src.model import BaseModelConfig
from src.play import Player
from src.train import Trainer, TrainConfig


@click.group()
@click.argument("exp_dir", type=click.Path(file_okay=False, exists=True))
@click.pass_context
def cli(ctx: click.Context, exp_dir: str):
    warnings.simplefilter("ignore")
    ctx.ensure_object(dict)

    ctx.obj["model_config"] = BaseModelConfig.from_yaml(os.path.join(exp_dir, "model.yaml"))
    ctx.obj["model_dir"] = exp_dir

    if not os.path.isdir(ctx.obj["model_dir"]):
        os.makedirs(ctx.obj["model_dir"])


@cli.command()
@click.option("--resume", is_flag=True)
@click.option("--data-dir", type=click.Path(file_okay=False, exists=True), default="data")
@click.option("--data-workers", type=int, default=4)
@click.pass_context
def train(ctx: click.Context, resume: bool, data_dir: str, data_workers: int):
    train_config_path = os.path.join(ctx.obj["model_dir"], "train.config")
    if not os.path.exists(train_config_path):
        train_config = TrainConfig()
    else:
        train_config = TrainConfig.from_yaml(train_config_path)

    trainer = Trainer(
        **train_config.__dict__,
        data_dir=data_dir,
        model_dir=ctx.obj["model_dir"],
        model_config=ctx.obj["model_config"],
        resume=resume,
        data_workers=data_workers
    )

    trainer.train()


@cli.command()
@click.option("--thinking-time", type=float, default=3.0)
@click.option("--temperature", type=click.FloatRange(0, min_open=True), default=None)
@click.option("--computer-first", is_flag=True)
@click.pass_context
def play(ctx, thinking_time: float, temperature: Optional[float], computer_first: bool):
    filename = None
    for file in os.listdir(ctx.obj["model_dir"]):
        if os.path.splitext(file)[1] == ".pt":
            filename = os.path.join(ctx.obj["model_dir"], file)

    player = Player(ctx.obj["model_config"], filename, thinking_time, temperature, computer_first)
    player.play()


if __name__ == "__main__":
    cli()
