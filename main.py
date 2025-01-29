import click
import warnings
import os
import torch

from typing import Optional
from src.mcts import MCTS
from src.model import ConvModel, ConvModelConfig
from src.train import Trainer, TrainConfig


@click.group()
@click.argument("model_config", type=click.Path())
@click.pass_context
def cli(ctx: click.Context, model_config: str):
    warnings.simplefilter("ignore")
    ctx.ensure_object(dict)

    ctx.obj["model_config"] = ConvModelConfig.from_yaml(model_config)
    ctx.obj["model_dir"] = os.path.dirname(model_config)

    if not os.path.isdir(ctx.obj["model_dir"]):
        os.makedirs(ctx.obj["model_dir"])


@cli.command()
@click.option("--train-config", type=click.Path(), required=False)
@click.option("--resume", is_flag=True)
@click.option("--data-dir", type=click.Path(), default="data")
@click.option("--data-workers", type=int, default=4)
@click.pass_context
def train(ctx: click.Context, train_config: Optional[str], resume: bool, data_dir: str, data_workers: int):
    if train_config is None:
        train_config = TrainConfig()
    else:
        train_config = TrainConfig.from_yaml(train_config)

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
@click.option("--num-sims", type=int, default=1000)
@click.option("--computer-first", is_flag=True)
@click.pass_context
def play(ctx, num_sims, computer_first):
    mcts = MCTS(sims_per_move=num_sims)

    model = ConvModel.from_config(ctx.obj["model_config"])

    filename = None
    for file in os.listdir(ctx.obj["model_dir"]):
        if os.path.splitext(file)[1] == ".pt":
            filename = os.path.join(ctx.obj["model_dir"], file)

    assert filename is not None, "No model found."
    model.load_state_dict(torch.load(filename, weights_only=True))

    player = not computer_first
    while mcts.root.state.winner() is None:
        mcts.root.state.display()
        if player:
            action = int(input("Enter your move: "))
        else:
            mcts.run_simulations(model)
            action = mcts.root.num_visits.argmax()
            print(f"Computer move: {int(action)}")

        mcts.step(action)
        player = not player

    mcts.root.state.display()
    winner = mcts.root.state.winner()
    if winner == 0:
        print("It's a tie!")
    elif winner == 1:
        print("Player wins!")
    else:
        print("Computer wins!")


if __name__ == "__main__":
    cli()
