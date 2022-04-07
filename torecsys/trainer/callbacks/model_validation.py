from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from torecsys.trainer import TorecsysPipeline


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
        if not isinstance(module, TorecsysPipeline):
            raise TypeError('')

        objective = module.objective
        if objective == TorecsysPipeline.MODULE_TYPE_CTR:
            if module.inputs is None:
                raise ValueError('missing inputs in the module')

            if module.model is None:
                raise ValueError('missing model in the module')

            if module.criterion is None:
                raise ValueError('missing criterion in the module')

            if module.optimizer is None:
                raise ValueError('missing optimizer in the module')

        elif objective == TorecsysPipeline.MODULE_TYPE_EMB:
            pass
        elif objective == TorecsysPipeline.MODULE_TYPE_LTR:
            # TODO: in development
            if not module.has_miner:
                raise ValueError('missing miner in the module')

            if not module.has_miner_target_field:
                raise ValueError('missing miner_target_field in the module')

        else:
            raise ValueError('')

        print(module.summary())
        return
