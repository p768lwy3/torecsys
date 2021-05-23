"""
Interface to build and run models through command line
"""

import json

import click

from torecsys import __version__
from torecsys.trainer import TorecsysModule


def print_version(ctx, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f'The current version is: {__version__}.')
    ctx.exit()


@click.group()
@click.option('--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f'Thank you for using ToR[e]csys.\nDebug mode is {"on" if debug else "off"}.')


@cli.command()
def version():
    click.echo(f'The current version is: {__version__}.')


@cli.command()
@click.option('--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
@click.option('--load_from', type=str, help='File path of trainer configuration.')
@click.option('--inputs_config', type=str, help='Json to configure embedder. Example: .')
@click.option('--model_config', type=str,
              help='Json to configure model. Example: \'{"method": "FM", "embed_size": 8, "num_fields": 2}\'.')
@click.option("--regularizer_config", type=str,
              help="Json to configure regularizer. Example: '{\"weight_decay\": 0.1}'.")
@click.option("--criterion_config", type=str,
              help="Json to configure criterion. Example: '{\"method\":\"MSELoss\", \"reduction\": \"mean\"}'.")
@click.option("--optimizer_config", type=str,
              help="Json to configure optimizer. Example: '{\"method\":\"SGD\", \"lr\": 0.01, \"momentum\": 0.9}'.")
@click.option("--training_set_config", type=str, help="File path of csv training dataset.")
@click.option("--validation_set_config", type=str, help="File path of csv validation dataset.")
@click.option("--targets_name", type=str, help="Targets field name of trainer.")
@click.option("--data_type", type=click.Choice(['double', 'float', 'half'], case_sensitive=False),
              help="Data type of trainer.")
@click.option("--max_num_epochs", type=int, help="Maximum number of training epochs.")
@click.option("--max_num_iterations", type=int, help="Maximum number of training iterations per epoch.")
@click.option("--objective",
              type=click.Choice(['clickthroughrate', 'embedding', 'learningtorank'], case_sensitive=False),
              help="Objective of trainer.")
@click.option("--enable_cuda/--disable_cuda", default=False, help="Enable/Disable cuda in trainer.")
@click.option("--cuda_config", type=str, help="Json to configure cuda in trainer. Example: '{\"devices\": [1, 2]}")
@click.option("--enable_jit/--disable_jit", default=False, help="Enable/Disable jit in trainer.")
def build(load_from: str,
          inputs_config: str,
          model_config: str,
          regularizer_config: str,
          criterion_config: str,
          optimizer_config: str,
          training_set_config: str,
          validation_set_config: str,
          targets_name: str,
          data_type: str,
          max_num_epochs: int,
          max_num_iterations: int,
          objective: str,
          enable_cuda: bool,
          cuda_config: str,
          enable_jit: bool):
    if load_from is not None:
        load_from = json.loads(load_from)
        trainer = TorecsysModule.build(load_from=load_from)
    else:
        if inputs_config is not None:
            inputs_config = json.loads(inputs_config)

        if model_config is not None:
            model_config = json.loads(model_config)

        if regularizer_config is not None:
            regularizer_config = json.loads(regularizer_config)

        if criterion_config is not None:
            criterion_config = json.loads(criterion_config)

        if optimizer_config is not None:
            optimizer_config = json.loads(optimizer_config)

        if training_set_config is not None:
            training_set_config = json.loads(training_set_config)

        if validation_set_config is not None:
            validation_set_config = json.loads(validation_set_config)

        if enable_cuda and cuda_config is not None:
            use_cuda = json.loads(cuda_config)
        else:
            use_cuda = enable_cuda

        trainer = TorecsysModule.build(
            inputs_config=inputs_config,
            model_config=model_config,
            regularizer_config=regularizer_config,
            criterion_config=criterion_config,
            optimizer_config=optimizer_config,
            training_set_config=training_set_config,
            validation_set_config=validation_set_config,
            targets_name=targets_name,
            data_type=data_type,
            max_num_epochs=max_num_epochs,
            max_num_iterations=max_num_iterations,
            objective=objective,
            use_cuda=use_cuda,
            use_jit=enable_jit
        )

    trainer.summary()


def main():
    cli()
