"""To be developed, module to use cli to build a recsys model.
"""

import click
import torecsys.trainer.Trainer as Trainer

@click.command()
@click.option("--model", help="Name of model.")
@click.option("--epochs", default=10, help="Number of training epochs.")
def trainer(model: str, epochs: int):
    # Initialize trainer with trainer.factory
    trainer = Trainer.factory(
        inputs = inputs,
        model = model,
        epcohs = epochs
    )

    trainer.fit()
