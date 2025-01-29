import click
import warnings

from src.mcts import MCTS
from src.model import ConvModel


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    warnings.simplefilter("ignore")
    ctx.ensure_object(dict)


@cli.command()
@click.option("--num-sims", type=int, default=1000)
@click.option("--computer-first", is_flag=True)
def play(num_sims, computer_first):
    mcts = MCTS(sims_per_move=num_sims)

    model = ConvModel(dims=(64,), depths=(3,))

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
