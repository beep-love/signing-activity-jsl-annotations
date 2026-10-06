"""RO-MAN pose network, adapted from the supplied MaAI model.py.
Pose parameter names and numerical forward are unchanged. Audio-only support omitted.
See LICENSE and SOURCE_MANIFEST.json for provenance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from .objective import ObjectiveVAP
from .modules import GPT, GPTStereo
from .utils import (
    everything_deterministic,
    vad_fill_silences,
    vad_omit_spikes,
)

BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]

everything_deterministic()



def build_pose_index(mod: str) -> torch.LongTensor:
    def pts_to_feat(pts):
        idx=[]
        for p in pts:
            idx += [3*p, 3*p+1, 3*p+2]
        return torch.tensor(idx, dtype=torch.long)

    if mod == "body":
        return pts_to_feat(range(0, 25))
    if mod == "hands":
        return pts_to_feat(range(25, 67))
    if mod == "face":
        return pts_to_feat(range(67, 137))
    if mod == "mouth":
        return pts_to_feat(range(115, 135))  # 115..134
    if mod == "eyes":
        pts = list(range(103, 115)) + [135, 136]
        return pts_to_feat(pts)
    raise ValueError(mod)

class PoseLinear(nn.Module):
    def __init__(self, in_dim, out_dim=256, dropout=0.1):
        super().__init__()
        # self.ln = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        # self.dp = nn.Dropout(dropout)

    def forward(self, x):  # (B,T,D)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # x = self.ln(x)
        x = self.fc(x)
        # x = F.gelu(x)
        # x = self.dp(x)
        return x


@dataclass
class VapConfig:
    sample_rate: int = 16_000
    frame_hz: int = 50
    bin_times: List[float] = field(default_factory=lambda: BIN_TIMES)

    # Encoder (training flag)
    encoder_type: str = "cpc"
    cpc_model_pt: str = "default"
    linear_in_dim: int = 0
    pose_modalities: str = "hands"   # "hands,face,body" etc
    pose_fusion: str = "concat"      # "concat" or "sum" or "linear"
    pose_dim_total: int = 411
    
    freeze_encoder: int = 1  # stupid but works (--vap_freeze_encoder 1)
    load_pretrained: int = 1  # stupid but works (--vap_load_pretrained 1)
    only_feature_extraction: int = 0

    # GPT
    dim: int = 256
    channel_layers: int = 1
    cross_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    context_limit: int = -1

    context_limit_cpc_sec: float = -1

    # Added Multi-task
    lid_classify: int = 0   # 1...last layer, 2...middle layer
    lid_classify_num_class: int = 3
    lid_classify_adversarial: int = 0
    lang_cond: int = 0

    @staticmethod
    def add_argparse_args(parser, fields_added=[]):
        for k, v in VapConfig.__dataclass_fields__.items():
            if k == "bin_times":
                parser.add_argument(
                    f"--vap_{k}", nargs="+", type=float, default=v.default_factory()
                )
            else:
                parser.add_argument(f"--vap_{k}", type=v.type, default=v.default)
            fields_added.append(k)
        return parser, fields_added

    @staticmethod
    def args_to_conf(args):
        return VapConfig(
            **{
                k.replace("vap_", ""): v
                for k, v in vars(args).items()
                if k.startswith("vap_")
            }
        )


class VapGPT(nn.Module):
    def __init__(self, conf: Optional[VapConfig] = None):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        # Objective (defines horizon, number of VAP classes, etc.)
        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        self.temp_elapse_time = []

        # Only the pose path is part of this baseline distribution.
        if self.conf.encoder_type != "pose":
            raise ValueError("This baseline supports encoder_type='pose' only.")
        self.encoder = None

        mods = [m.strip() for m in conf.pose_modalities.split(",") if m.strip()]

        self.pose_index = {m: build_pose_index(m) for m in mods}

        # encoder: (D_m -> 256) をモダリティごとに
        self.enc = nn.ModuleDict({m: PoseLinear(in_dim=len(self.pose_index[m]), out_dim=conf.dim, dropout=conf.dropout)
                                for m in mods})

        # Separate attention branches for each modality, as in the training model.
        self.ar_channel = nn.ModuleDict({m: GPT(dim=conf.dim, dff_k=3, num_layers=conf.channel_layers,
                                                num_heads=conf.num_heads, dropout=conf.dropout,
                                                context_limit=conf.context_limit)
                                        for m in mods})

        self.ar = nn.ModuleDict({m: GPTStereo(dim=conf.dim, dff_k=3, num_layers=conf.cross_layers,
                                            num_heads=conf.num_heads, dropout=conf.dropout,
                                            context_limit=conf.context_limit)
                                for m in mods})

        # fusion
        if conf.pose_fusion == "concat":
            fused_dim = conf.dim * len(mods)
            self.fuse = None
        elif conf.pose_fusion == "sum":
            fused_dim = conf.dim
            self.fuse = None
        elif conf.pose_fusion == "linear":
            fused_dim = conf.dim
            self.fuse = nn.Linear(conf.dim * len(mods), conf.dim)
        else:
            raise ValueError(conf.pose_fusion)

        self.va_classifier = nn.Linear(fused_dim, 1)
        self.vap_head = nn.Linear(fused_dim, self.objective.n_classes)
        
    @property
    def horizon_time(self):
        return self.objective.horizon_time

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    # def freeze(self):
    #     for p in self.encoder.parameters():
    #         p.requires_grad_(False)
    #     print(f"Froze {self.__class__.__name__}!")

    @torch.no_grad()
    def probs(
        self,
        waveform: Tensor,
        vad: Optional[Tensor] = None,
        now_lims: List[int] = [0, 1],
        future_lims: List[int] = [2, 3],
    ) -> Dict[str, Tensor]:
        
        vad_ref = vad
        out = self(waveform)
        probs = out["logits"].softmax(dim=-1)
        vad_prob = out["vad"].sigmoid()
        
        if self.conf.lid_classify >= 1:
            lid = out["lid"].softmax(dim=-1)

        # Calculate entropy over each projection-window prediction (i.e. over
        # frames/time) If we have C=256 possible states the maximum bit entropy
        # is 8 (2^8 = 256) this means that the model have a one in 256 chance
        # to randomly be right. The model can't do better than to uniformly
        # guess each state, it has learned (less than) nothing. We want the
        # model to have low entropy over the course of a dialog, "thinks it
        # understands how the dialog is going", it's a measure of how close the
        # information in the unseen data is to the knowledge encoded in the
        # training data.
        h = -probs * probs.log2()  # Entropy
        H = h.sum(dim=-1)  # average entropy per frame

        # first two bins
        p_now = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=now_lims[0], to_bin=now_lims[-1]
        )
        p_future = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=future_lims[0], to_bin=future_lims[1]
        )

        ret = {
            "probs": probs,
            "vad": vad_prob,
            "p_now": p_now,
            "p_future": p_future,
            "H": H,
        }

        if self.conf.lid_classify >= 1:
            ret.update({"lid": lid})

        if vad_ref is not None:
            labels = self.objective.get_labels(vad_ref)
            ret["loss"] = self.objective.loss_vap(
                out["logits"], labels, reduction="none"
            )
        return ret

    @torch.no_grad()
    def vad(
        self,
        waveform: Tensor,
        max_fill_silence_time: float = 0.02,
        max_omit_spike_time: float = 0.02,
        vad_cutoff: float = 0.5,
    ) -> Tensor:
        """
        Extract (binary) Voice Activity Detection from model
        """
        vad = (self(waveform)["vad"].sigmoid() >= vad_cutoff).float()
        for b in range(vad.shape[0]):
            # TODO: which order is better?
            vad[b] = vad_fill_silences(
                vad[b], max_fill_time=max_fill_silence_time, frame_hz=self.frame_hz
            )
            vad[b] = vad_omit_spikes(
                vad[b], max_omit_time=max_omit_spike_time, frame_hz=self.frame_hz
            )
        return vad
    
    def forward(self, waveform, attention=False, lang_info=None):
        # waveform: (B,2,T,411)
        if self.conf.encoder_type == "pose":
            xs, xs1, xs2 = [], [], []
            for m, idx in self.pose_index.items():
                idx = idx.to(waveform.device)
                w_m = waveform.index_select(-1, idx)   # (B,2,T,Dm)

                x1 = self.enc[m](w_m[:, 0])  # (B,T,256)
                x2 = self.enc[m](w_m[:, 1])

                o1 = self.ar_channel[m](x1, attention=attention)
                o2 = self.ar_channel[m](x2, attention=attention)
                out = self.ar[m](o1["x"], o2["x"], attention=attention)

                xs.append(out["x"])
                xs1.append(out["x1"])
                xs2.append(out["x2"])

            if self.conf.pose_fusion == "sum":
                x  = sum(xs)
                x1 = sum(xs1)
                x2 = sum(xs2)
            else:
                x  = torch.cat(xs,  dim=-1)
                x1 = torch.cat(xs1, dim=-1)
                x2 = torch.cat(xs2, dim=-1)
                if self.fuse is not None:
                    x  = self.fuse(x)
                    x1 = self.fuse(x1)
                    x2 = self.fuse(x2)

            v1 = self.va_classifier(x1)
            v2 = self.va_classifier(x2)
            vad = torch.cat((v1, v2), dim=-1)   # (B,T,2)
            logits = self.vap_head(x)           # (B,T,n_classes)
            return {"logits": logits, "vad": vad}

        raise NotImplementedError(
            f"forward() for encoder_type={self.conf.encoder_type} is not implemented in this modified model. "
            "Use --vap_encoder_type pose (SLAP/OpenPose features) or restore the original audio forward."
        )

