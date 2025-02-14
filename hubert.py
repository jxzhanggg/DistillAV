# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
import logging
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
# np.random.seed(1234)
import torch

############### 
# torch.manual_seed(0)
# np.random.seed(0)
# import random
# random.seed(0)
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.benchmark = False
############### 

# from .asr_model.asr_model_nolong import Conformer_fb80,ConformerModelConfig
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from .search_functions import search_offline
from copy import deepcopy
import math

# from .cross_attention_poscode import CrossAttention2View
from .weighting import MGDA, Aligned_MTL

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from resnet import ResEncoder
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    from utils import compute_mask_indices
    from decoder import TransformerDecoder

else:
    from .hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from .resnet import ResEncoder
    from .utils import compute_mask_indices
    from .decoder import TransformerDecoder

from omegaconf import II

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)

@dataclass
class AVHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})

    # cross_attention_n_layer: int = field(
    #     default=2, metadata={'help': "cross attention n layers"}
    # )

    instance_norm_target_layer: bool = True
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )
    use_ffn: bool = False
    use_km_emb: bool = False

class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None
        # self.post_norm = LayerNorm(cfg.encoder_embed_dim)

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))

        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x

class ApplyKmeansCUDA(torch.nn.Module):
    def __init__(self, km_path):
        super().__init__()
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose() # 2000,d -> d, 2000
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True) # 1, 2000

        C = torch.from_numpy(self.C_np)
        Cnorm = torch.from_numpy(self.Cnorm_np)

        self.C = torch.nn.Parameter(C, requires_grad=False)
        self.Cnorm = torch.nn.Parameter(Cnorm, requires_grad=False)
        

    def forward(self, x):
        # if isinstance(x, torch.Tensor):
            # x = x.to(self.C.device)
        # 't, d'

        dist = (
            x.pow(2).sum(1, keepdim=True) #t, 1
            - 2 * torch.matmul(x, self.C) # t, d x d, 2000
            + self.Cnorm  #1, 2000
        )
        return dist.argmin(dim=1), dist # t, 2000

class Vocabulary(object):
    def __init__(self, f_dic, encoding="utf-8"):
        self.idx2word = []
        self.word2idx = {}
        with open(f_dic, "r", encoding=encoding) as f:
            for idx, line in enumerate(f):
                x = line.strip()
                self.idx2word.append(x)
                self.word2idx[x] = idx


    def list_to_sent(self, li):
        words = [self.idx2word[i] for i in li]
        return " ".join(words)



@register_model("av_hubert", dataclass=AVHubertConfig)
class AVHubertModel(BaseFairseqModel, Aligned_MTL):
    def __init__(
        self,
        cfg: AVHubertConfig,
        task_cfg: AVHubertPretrainingConfig,
        dictionaries: List[Dictionary],
        **kwargs
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg)
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)
        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        # elif self.modality_fuse == 'cross_attention':
        #     self.embed = cfg.encoder_embed_dim * 2 
        
        # if self.modality_fuse == 'cross_attention':
        #     self.cross_attention = CrossAttention2View(cfg, cfg.cross_attention_n_layer,
        #                                     left_context=16, right_context=16)
        

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        if self.masking_type == 'feature':
            self.mask_emb_video = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj

        # regression
        self.regression_proj_wavlm = nn.Linear(cfg.encoder_embed_dim, 1024 * 2)
        self.regression_proj_2 = nn.Linear(cfg.encoder_embed_dim, 512)
        
        self.average_top_k_layers = cfg.average_top_k_layers
        self.use_ffn = cfg.use_ffn
        self.instance_norm_target_layer = cfg.instance_norm_target_layer

        word_dict = "/work2/asrprg/pcli2/experiments/theta-workspace/lexicon_norm3_final_w0.1_preplus.wlist.v2"
        self.voc = Vocabulary(word_dict)


        km_path = "/work2/asrprg/jxzhang46/lippretrain/extract_wavLM_fea/wavlm_full_avg8_c2000.km"
        self.km_model = ApplyKmeansCUDA(km_path)
        self.I =  352.62631
        self.temp = 0.1

        km_path_2 = "/work2/asrprg/jxzhang46/lippretrain/lippretrainv1/0_dump_asr_features/asr_layer15_c2000.km"
        self.km_model_2 = ApplyKmeansCUDA(km_path_2)
        self.I_2 =  19.01680

        self.final_dim = final_dim
        self.use_km_emb = cfg.use_km_emb

        assert(self.use_km_emb == False)
        if not self.use_km_emb:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, 
                                        final_dim * 2)
            self.final_proj_2 = nn.Linear(
                cfg.encoder_embed_dim,
                final_dim)
            
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(2000, final_dim))
            self.label_embs_concat_2 = nn.Parameter(
                torch.FloatTensor(2000, final_dim))
            nn.init.uniform_(self.label_embs_concat)
            nn.init.uniform_(self.label_embs_concat_2)
        else:
            self.final_proj =  nn.Linear(cfg.encoder_embed_dim, 
                                        512)
        
        self.rep_tasks = {}
        self.rep = {}
        self.rep_grad = True
        self.task_name = ["reg_wavlm", "reg_conformer", 'kld_wavlm', "kld_conformer"]
        self.task_num = len(self.task_name)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: AVHubertConfig, task: AVHubertPretrainingTask):
        """Build a new model instance."""

        kwargs = {}
        model = AVHubertModel(cfg, task.cfg, task.dictionaries, **kwargs)
        return model

    def apply_input_mask(self, x, padding_mask, target_list):
        B, C, T = x.shape[:3] #B, C
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:
            # mask_indices [B, T] mask=True, unmask=False
            # starts, ends, batch_indexes: 1D arrays with start_pos, end_pos, batch_id
            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous() # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.selection_type == 'same_other_seq':
                # use clip from other batch to fill the masked range
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.selection_type == 'same_seq':
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end-start
                    # ensure the other clip won't overlap with the range(start, end)
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start-length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    # clip to prevent index exceeding the max length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T-1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64)+batch_index)
                # gen indexes 
                batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                
                x[mask_indices] = x[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            logger.info(f"No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask, stream, target_list):

        B, T, C = x.shape
        if stream == 'audio':
            mask_emb, mask_prob, mask_length = self.mask_emb, self.mask_prob_audio, self.mask_length_audio
        elif stream == 'video':
            mask_emb, mask_prob, mask_length = self.mask_emb_video, self.mask_prob_image, self.mask_length_image
        else:
            assert False
        
        #assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
        # mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image 0.8 5
        
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            # x[mask_indices] = self.mask_emb
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask
    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        # make input output the same frame rate.
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

  

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1) # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0) # [B*T, V]
            logits = (nom/denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        
        elif self.sim_type == "mse":
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            feats2 = (feats_ ** 2).sum(dim=-1, keepdims=True) # t, 1
            emb_mat2 = (emb_mat ** 2).sum(dim=-1, keepdims=True) # v, 1
            mse = feats2 + emb_mat2.transpose(0, 1) - 2 * torch.matmul(feats_, emb_mat.transpose(0, 1)) # t, 1 + 1, v - 2 * t, f x f, v -> (t, v)
            logits = mse.reshape(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source['audio'], source['video']

        if mask and self.masking_type == 'input':
            # target_list not used here.
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list)
            # union indices.
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None
        
        features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
        features_video = self.forward_features(src_video, modality='video')

        if self.masking_type == 'feature' and mask:
            features_audio, mask_indices_audio = self.apply_feature_mask(features_audio.transpose(1,2).clone(), padding_mask, "audio", target_list)
            features_video, mask_indices_video = self.apply_feature_mask(features_video.transpose(1,2).clone(), padding_mask, "video", target_list)
            features_audio = features_audio.transpose(1,2)
            features_video = features_video.transpose(1,2)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()

        only_video = False
        only_audio = False

        if self.training:
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    features_audio = 0 * features_audio
                    only_video = True
                else:
                    features_video = 0 * features_video
                    only_audio = True
                    # mask_indices = mask_indices_audio

        
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        
        # elif self.modality_fuse == 'cross_attention':
        #     features_audio, features_video = \
        #         self.cross_attention(features_audio, features_video, padding_mask)
        #     features = torch.cat([features_audio, features_video], dim=1)

        # assert False
        if target_list is not None:
            features, mask_indices, target_list = self.forward_targets(features, mask_indices, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2) # [b, c, t] -> [b, t, c]
        features = self.layer_norm(features)

        if padding_mask is not None:
            # subsample the padding mask
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        x, _ = self.encoder(
            features,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )


        
        for t in self.task_name:
            self.rep_tasks[t] = x.detach().clone()
            self.rep_tasks[t].requires_grad = True
        
        self.rep = x

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        reg_x = self.regression_proj_wavlm(
            self.rep_tasks["reg_wavlm"]) #[B, T, 512] -> [B*T, 512]
        reg_x = reg_x.reshape(-1, int(reg_x.shape[-1] / 2))

        reg_x_2 = self.regression_proj_2(self.rep_tasks["reg_conformer"]) #[B, T, 512] -> [B*T, 512]
        reg_x_2 = reg_x_2.reshape(-1, reg_x_2.shape[-1])


        asr_fea_padding_mask = source["asr_fea_padding_mask"]
        batch_size, time = padding_mask.shape
        upsample_padding_mask = padding_mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, time * 2)
        mask = torch.logical_or(upsample_padding_mask, asr_fea_padding_mask) #unio
        mask = mask.reshape(batch_size, time, 2)
        mask = torch.logical_or(mask[:, :, 0], mask[:, :, 1])
        mask = ~mask
        mask = mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, time * 2)
        wavlm_mask = mask.reshape(-1)

        conformer_mask = torch.logical_and(~padding_mask, ~source["conformer_padding_mask"])
        conformer_mask = conformer_mask.reshape(-1)


        m_reg = reg_x[wavlm_mask]
        m_reg_2 = reg_x_2[conformer_mask]
        
        
        
        cls_x = self.final_proj(self.rep_tasks["kld_wavlm"])
        cls_x = cls_x.reshape(batch_size, -1, int(cls_x.shape[-1] / 2))
        cls_x = self.compute_logits(cls_x, self.label_embs_concat)
        cls_x = cls_x.reshape(-1, 2000).float()

        cls_x_2 = self.final_proj_2(self.rep_tasks["kld_conformer"])
        cls_x_2 = self.compute_logits(cls_x_2, self.label_embs_concat_2)
        cls_x_2 = cls_x_2.reshape(-1, 2000).float()


        m_cls = cls_x[wavlm_mask]
        m_cls_2 = cls_x_2[conformer_mask]

        with torch.no_grad():
  
            fea_target = source["asr_fea"].transpose(1, 2) # b, d, t -> b, t, d
            b, t, d = fea_target.shape
            # print(fea_target.shape)
            dis_targets, dist = self.km_model(fea_target.reshape(-1, d))
            dis_targets = dis_targets.reshape(b, t)
            dist = dist.reshape(b, t, -1).float()
            dist = nn.functional.softmax(-dist / (self.I * self.temp), dim=-1)


            fea_target_2 = source["conformer_asr_fea"].transpose(1, 2)
            b, t, d = fea_target_2.shape
            # print(fea_target_2.shape)
            dis_targets_2, dist_2 = self.km_model_2(fea_target_2.reshape(-1, d))
            dis_targets_2 = dis_targets_2.reshape(b, t)
            dist_2 = dist_2.reshape(b, t, -1).float()
            dist_2 = nn.functional.softmax(-dist_2 / (self.I_2 * self.temp), dim=-1)


            def _get_pred_tar(fea_target, dis_targets, dist, mask):

                
                fea_target = fea_target.reshape(-1, fea_target.shape[-1])
                dis_targets = dis_targets.reshape(-1)
                dist = dist.reshape(-1, 2000)

                m_tar = fea_target[mask]
                m_dis_tar = dis_targets[mask]
                m_dis = dist[mask]

                return m_tar, m_dis_tar, m_dis


            m_tar, m_dis_tar, m_dis = _get_pred_tar(
                fea_target, dis_targets, dist, wavlm_mask)

            
            m_tar_2, m_dis_tar_2, m_dis_2 = _get_pred_tar(
                fea_target_2, dis_targets_2, dist_2, conformer_mask)


        result = {
            "m_reg": [m_reg, m_reg_2],
            "m_tar": [m_tar, m_tar_2],
            "m_cls": [m_cls, m_cls_2],
            "m_dis_tar": [m_dis_tar, m_dis_tar_2],
            "m_dis": [m_dis, m_dis_2],
            "padding_mask": padding_mask,
            "features_pen": features_pen}

        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video) # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim, features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]

        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        
        # elif self.modality_fuse == 'cross_attention':
        #     features_audio, features_video = \
        #         self.cross_attention(features_audio, features_video, padding_mask)
        #     features = torch.cat([features_audio, features_video], dim=1)
        
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        return x, padding_mask


    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
        self.asr_teacher = None
        self.regression_proj_wavlm = None

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits



    
