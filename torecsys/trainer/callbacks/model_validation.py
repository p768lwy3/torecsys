from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from torecsys.trainer import TorecsysModule


class ModelValidationCallback(Callback):
    """
    Callback function on init start to check if the model is ready to train
    """

    def on_init_start(self, trainer: Trainer):
        return

    def on_fit_start(self, trainer: Trainer, module: LightningModule):
        """

        Args:
            trainer:
            module:

        Returns:

        """
        if not isinstance(module, TorecsysModule):
            raise TypeError('')

        objective = module.objective
        if objective == TorecsysModule.MODULE_TYPE_CTR:
            if module.inputs is None:
                raise ValueError('missing inputs in the module')

            if module.model is None:
                raise ValueError('missing model in the module')

            if module.criterion is None:
                raise ValueError('missing criterion in the module')

            if module.optimizer is None:
                raise ValueError('missing optimizer in the module')

        elif objective == TorecsysModule.MODULE_TYPE_EMB:
            pass
        elif objective == TorecsysModule.MODULE_TYPE_LTR:
            pass
        else:
            raise ValueError('')

        print(module.summary())
        return
