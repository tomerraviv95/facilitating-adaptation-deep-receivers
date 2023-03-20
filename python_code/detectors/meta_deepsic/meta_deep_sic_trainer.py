import copy
from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.meta_deepsic.meta_deep_sic_detector import MetaDeepSICDetector

MAML_FLAG = True
META_LR = 0.01
ITERATIONS = 3
EPOCHS = 300


class MetaDeepSICTrainer(DeepSICTrainer):
    """
    Trainer for the DeepSIC model.
    """

    def __init__(self):
        super().__init__()
        self.is_online_meta = True

    def copy_model(self, model: List[nn.Module]) -> List[nn.Module]:
        return [copy.deepcopy(single_model) for single_model in model]

    def _meta_train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main meta-training loop. Runs in minibatches, each minibatch is split to pairs of following words.
        The pairs are comprised of (support,query) words.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        opt = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        crt = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        meta_model = MetaDeepSICDetector()
        support_idx = torch.arange(tx.shape[0] - conf.pilot_size)
        query_idx = torch.arange(conf.pilot_size, tx.shape[0])
        y_total = self.preprocess(rx)

        for _ in range(EPOCHS):
            opt.zero_grad()

            # divide the words to following pairs - (support,query)
            support_b, support_y = tx[support_idx], y_total[support_idx]
            query_b, query_y = tx[query_idx], y_total[query_idx]

            # local update (with support set)
            para_list_detector = list(map(lambda p: p[0], zip(single_model.parameters())))
            soft_estimation_supp = meta_model(support_y.float(), para_list_detector)
            loss_supp = crt(soft_estimation_supp, support_b.long())

            # set create_graph to True for MAML, False for FO-MAML
            local_grad = torch.autograd.grad(loss_supp, para_list_detector, create_graph=MAML_FLAG)
            updated_para_list_detector = list(
                map(lambda p: p[1] - META_LR * p[0], zip(local_grad, para_list_detector)))

            # meta-update (with query set) should be same channel with support set
            soft_estimation_query = meta_model(query_y.float(), updated_para_list_detector)
            loss_query = crt(soft_estimation_query, query_b.long())
            meta_grad = torch.autograd.grad(loss_query, para_list_detector, create_graph=False)

            ind_param = 0
            for param in single_model.parameters():
                param.grad = None  # zero_grad
                param.grad = meta_grad[ind_param]
                ind_param += 1

            opt.step()

    def _meta_train_models(self, model: List[List[DeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                           rx_all: List[torch.Tensor]):
        for user in range(self.n_user):
            self._meta_train_model(model[user][i], tx_all[user], rx_all[user])

    def _meta_training(self, saved_detector: List[List[DeepSICDetector]], tx: torch.Tensor, rx: torch.Tensor):
        initial_probs = self._initialize_probs(tx)
        tx_all, rx_all = self.prepare_data_for_training(tx, rx, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self._meta_train_models(saved_detector, 0, tx_all, rx_all)
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(saved_detector, i, probs_vec, rx)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self._meta_train_models(saved_detector, i, tx_all, rx_all)
