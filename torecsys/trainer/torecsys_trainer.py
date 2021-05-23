from pytorch_lightning import Trainer

from torecsys.trainer.callbacks import ModelValidationCallback


class TorecsysTrainer(Trainer):
    """
    ...
    """

    def __init__(self, *args, **kwargs):
        callbacks = kwargs.get('callbacks', [])
        callbacks.extend([ModelValidationCallback()])
        kwargs['callbacks'] = callbacks

        super().__init__(*args, **kwargs)
