import copy

import torch
from torch import nn

from python_code import conf
from python_code.detectors.meta_viterbinet.meta_viterbinet_detector import MetaViterbiNetDetector
from python_code.detectors.viterbinet.viterbinet_detector import ViterbiNetDetector
from python_code.detectors.viterbinet.viterbinet_trainer import ViterbiNetTrainer

MAML_FLAG = True
META_LR = 0.01
EPOCHS = 300


class MetaViterbiNetTrainer(ViterbiNetTrainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self):
        super().__init__()
        self.is_online_training = True
        self.is_online_meta = True
        self.initialize_meta_detector()

    def initialize_meta_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.meta_detector = MetaViterbiNetDetector(n_states=self.n_states)

    def copy_model(self, model: nn.Module) -> nn.Module:
        return copy.deepcopy(model)

    def _meta_training(self, saved_detector: ViterbiNetDetector, tx: torch.Tensor, rx: torch.Tensor):
        opt = torch.optim.Adam(saved_detector.parameters(), lr=self.lr)
        support_idx = torch.arange(tx.shape[0] - conf.pilot_size)
        query_idx = torch.arange(conf.pilot_size, tx.shape[0])

        for _ in range(EPOCHS):
            opt.zero_grad()

            # divide the words to following pairs - (support,query)
            support_b, support_y = tx[support_idx], rx[support_idx]
            query_b, query_y = tx[query_idx], rx[query_idx]

            # local update (with support set)
            para_list_detector = list(map(lambda p: p[0], zip(self.detector.parameters())))
            soft_estimation_supp = self.meta_detector(support_y.float(), 'train', para_list_detector)
            loss_supp = self.calc_loss(soft_estimation_supp, support_b)

            # set create_graph to True for MAML, False for FO-MAML
            local_grad = torch.autograd.grad(loss_supp, para_list_detector, create_graph=MAML_FLAG)
            updated_para_list_detector = list(
                map(lambda p: p[1] - META_LR * p[0], zip(local_grad, para_list_detector)))

            # meta-update (with query set) should be same channel with support set
            soft_estimation_query = self.meta_detector(query_y.float(), 'train', updated_para_list_detector)
            loss_query = self.calc_loss(soft_estimation_query, query_b)
            meta_grad = torch.autograd.grad(loss_query, para_list_detector, create_graph=False)

            ind_param = 0
            for param in self.detector.parameters():
                param.grad = None  # zero_grad
                param.grad = meta_grad[ind_param]
                ind_param += 1

            opt.step()
