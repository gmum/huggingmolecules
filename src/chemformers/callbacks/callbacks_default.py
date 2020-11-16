import pytorch_lightning as pl


def get_default_callbacks():
    return [MyPrintingCallback()]


class MyPrintingCallback(pl.callbacks.Callback):

    def on_init_start(self, trainer):
        print('\n###Starting to init trainer!###\n')

    def on_init_end(self, trainer):
        print('\n####Trainer is init now####\n')

    def on_train_end(self, trainer, pl_module):
        print('\n####do something when training ends####\n')
