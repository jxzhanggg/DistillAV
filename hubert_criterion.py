# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class AVHubertCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )

    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )

    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


def scale_losses(losses, ratios):
    """
    缩放损失函数值,使其比例等于预设的比例

    Args:
        losses (list): 损失函数值的列表
        ratios (list): 预设比例的列表

    Returns:
        list: 缩放后的损失函数值的列表
    """
    # 计算损失函数值的总和
    total_loss = sum(losses)

    # 计算预设比例的总和
    total_ratio = sum(ratios)

    # 缩放损失函数值
    
    weights =  [ratio / total_ratio * total_loss / loss for loss, ratio in zip(losses, ratios)]

    norm_weights = [w / sum(weights) for w in weights]

    # scaled_losses = [loss *w for loss, w in zip(losses, norm_weights)]

    return norm_weights


@register_criterion("av_hubert", dataclass=AVHubertCriterionConfig)
class AVHubertCriterion(FairseqCriterion):
    def __init__(self, task, 
                pred_masked_weight, 
                pred_nomask_weight, 
                loss_weights=None, 
                log_keys=None):
        super().__init__(task)

        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

        
    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"],
                             **sample["net_input"])
        # loss = 0.
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"


        m_regs, m_tars = net_output['m_reg'], net_output['m_tar']

        fea_dims = [m_reg.size(-1) for m_reg in m_regs]

        loss_list = []    
        # tmp_loss_reg = []
        for ii, (m_reg, m_tar) in enumerate(zip(m_regs, m_tars)):
            loss_m_tmp = F.mse_loss(m_reg.float(), m_tar.float(), reduction="none").sum(dim=-1)
            loss_m_tmp = loss_m_tmp.sum() / math.sqrt(fea_dims[ii])
            # tmp_loss_reg.append(loss_m_tmp)
            loss_list.append(loss_m_tmp)
            logging_output[f"loss_m_fea_{ii}"] = loss_m_tmp.detach().item()


        m_clss, m_dis_tars, m_diss = net_output["m_cls"], net_output["m_dis_tar"], net_output["m_dis"]
        tmp_loss_kld, tmp_loss_dis = [], []
        for ii, (m_cls, m_dis_tar, m_dis) in enumerate(zip(m_clss, m_dis_tars, m_diss)):
            loss_m_dis_tar_tmp = F.cross_entropy(m_cls, m_dis_tar, reduction="sum")
            loss_m_kld_tmp = - (F.log_softmax(m_cls, dim=-1) * m_dis).sum() # [ p * log(q)]
            logging_output[f"loss_m_dis_{ii}"] = loss_m_dis_tar_tmp.detach().item()
            logging_output[f"loss_m_kld_{ii}"] = loss_m_kld_tmp.detach().item()
            tmp_loss_kld.append(loss_m_kld_tmp)
            loss_list.append(loss_m_kld_tmp)
            tmp_loss_dis.append(loss_m_dis_tar_tmp)


        sample_size += m_tars[1][:, 0].numel()
        
        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    # loss += p
                    loss_list.append(p)
                    logging_output[f"loss_{n}"] = p.item()

        logging_output = {
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))


        return loss_list, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
