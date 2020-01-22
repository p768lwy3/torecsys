"""To be developed, module to use cli to build a recsys model.
"""

import click
import json
<<<<<<< HEAD
from torecsys.trainer import Trainer
=======
from torecsys import __version__
from torecsys.trainer import DevTrainer
>>>>>>> 34b858f29f45780dc87587de034ec7dc1a770099

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"The current version is: {__version__}.")
    ctx.exit()

@click.group()
@click.option("--version", is_flag=True, callback=print_version, expose_value=False, is_eager=True)
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    click.echo("Thank you for using ToR[e]csys.")
    click.echo("Debug mode is %s." % ('on' if debug else 'off'))

@cli.command()
def version():
    click.echo(f"The current version is: {__version__}.")

@cli.command()
@click.option("--version", is_flag=True, callback=print_version, expose_value=False, is_eager=True)
@click.option("--load_from", type=str, help="File path of trainer configuration.")
@click.option("--inputs_config", type=str, help="Json to configurate inputs. Example: .")
@click.option("--model_config", type=str, help="Json to configurate model. Example: '{\"method\":\"FM\", \"embed_size\": 8, \"num_fields\": 2}'.")
@click.option("--regularizer_config", type=str, help="Json to configurate regularizer. Example: '{\"weight_decay\": 0.1}'.")
@click.option("--criterion_config", type=str, help="Json to configurate criterion. Example: '{\"method\":\"MSELoss\", \"reduction\": \"mean\"}'.")
@click.option("--optimizer_config", type=str, help="Json to configurate optimizer. Example: '{\"method\":\"SGD\", \"lr\": 0.01, \"momentum\": 0.9}'.")
@click.option("--training_set_config", type=str, help="File path of csv training dataset.")
@click.option("--validation_set_config", type=str, help="File path of csv validation dataset.")
@click.option("--targets_name", type=str, help="Targets field name of trainer.")
@click.option("--dtype", type=click.Choice(['double', 'float', 'half'], case_sensitive=False), help="Data type of trainer.")
@click.option("--max_num_epochs", type=int, help="Maxmum number of training epochs.")
@click.option("--max_num_iterations", type=int, help="Maxmum number of training iterations per epoch.")
@click.option("--objective", type=click.Choice(['clickthroughrate', 'embedding', 'learningtorank'], case_sensitive=False), help="Objective of trainer.")
@click.option("--enable_cuda/--disable_cuda", default=False, help="Enable/Disable cuda in trainer.")
@click.option("--cuda_config", type=str, help="Json to configurate cuda in trainer. Example: '{\"devices\": [1, 2]}")
@click.option("--enable_jit/--disable_jit", default=False, help="Enable/Disable jit in trainer.")
# @click.option("--metrics_config", type=str, multiple=True, help="Json to configurate metrics. Example: .")
# @click.option("--save_directory", type=str, help="Directory to save the trained model.")
# @click.option("--save_filename", type=str, default="model", show_default=True, help="File name to save the trained model.")
def build(load_from             : str,
          inputs_config         : str,
          model_config          : str,
          regularizer_config    : str,
          criterion_config      : str,
          optimizer_config      : str,
          training_set_config   : str,
          validation_set_config : str,
          targets_name          : str,
          dtype                 : str,
          max_num_epochs        : int,
          max_num_iterations    : int,
          objective             : str,
          enable_cuda           : bool,
          cuda_config           : str,
          enable_jit            : bool):
    if load_from is not None:
        # Get command line arguments
        load_from = json.loads(load_from)

        # Build trainer with trainer.build(...)
        trainer = Trainer.build(
            load_from = load_from
        )
    else:
        # Get command line arguments
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

        # Initialize trainer with trainer.build(...)
        trainer = Trainer.build(
            inputs_config         = inputs_config,
            model_config          = model_config,
            regularizer_config    = regularizer_config,
            criterion_config      = criterion_config,
            optimizer_config      = optimizer_config,
            training_set_config   = training_set_config,
            validation_set_config = validation_set_config,
            targets_name          = targets_name,
            dtype                 = dtype,
            max_num_epochs        = max_num_epochs,
            max_num_iterations    = max_num_iterations,
            objective             = objective,
            use_cuda              = use_cuda,
            use_jit               = enable_jit
        )

    # Print trainer summary after built
    trainer.summary()

    # Fit data to trainer
    # trainer.fit()

    # Save trained model
    # trainer.save(save_path=save_directory, file_name=save_filename)

def main():
    cli()
    