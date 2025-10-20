import torch
from torch.optim import Adam, AdamW

from nnxnet.training.lr_scheduler.polylr import PolyLRScheduler
from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer


class nnXNetTrainerAdam(nnXNetTrainer):
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnXNetTrainerVanillaAdam(nnXNetTrainer):
    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(),
                         lr=self.initial_lr,
                         weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnXNetTrainerVanillaAdam1en3(nnXNetTrainerVanillaAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnXNetTrainerVanillaAdam3en4(nnXNetTrainerVanillaAdam):
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4


class nnXNetTrainerAdam1en3(nnXNetTrainerAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnXNetTrainerAdam3en4(nnXNetTrainerAdam):
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4
