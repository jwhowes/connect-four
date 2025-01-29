import click

from src.mcts import MCTS

@click.group()
@click.pass_context
def cli(ctx: click.Context):
    ctx.ensure_object(dict)


@cli.command()
@click.option("--num-sims", type=int, default=1000)
def play(num_sims):
    mcts = MCTS(sims_per_move=num_sims)

    player = True
    while mcts.root.state.winner() is None:
        mcts.root.state.display()
        if player:
            action = int(input("Enter your move: "))
        else:
            mcts.run_simulations()
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
