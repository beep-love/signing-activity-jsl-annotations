"""Unmodified utility functions used by the pose and event paths."""
import torch
from torch import Tensor
from typing import List, Optional, Tuple
VAD_LIST = List[List[List[float]]]

def time_to_frames(t: float, hop_time: float) -> int:
    return int(t / hop_time)


def find_island_idx_len(
    x: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat(
        (torch.where(y)[0], torch.tensor(n - 1, device=x.device).unsqueeze(0))
    ).long()
    it = torch.cat((torch.tensor(-1, device=x.device).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(
        torch.cat((torch.tensor([0], device=x.device, dtype=torch.long), dur)), dim=0
    )[
        :-1
    ]  # positions
    return idx, dur, x[i]


def everything_deterministic():
    """
    -----------------------------
    Wav2Vec
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: replication_pad1d_backward_cuda does not have a deterministic
    implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can
    turn off determinism just for this operation if that's acceptable for your
    application. You can also file an issue at
    https://github.com/pytorch/pytorch/issues to help us prioritize adding
    deterministic support for this operation.


    -----------------------------
    CPC
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: Deterministic behavior was enabled with either
    `torch.use_deterministic_algorithms(True)` or
    `at::Context::setDeterministicAlgorithms(true)`, but this operation is not
    deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable
    deterministic behavior in this case, you must set an environment variable
    before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
    CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


    Set these ENV variables and it works with the above recipe

    bash:
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export CUBLAS_WORKSPACE_CONFIG=:16:8

    """
    from os import environ

    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def vad_list_to_onehot(
    vad_list: VAD_LIST,
    duration: float,
    hop_time: float = 0,
    frame_hz: float = 0,
    channel_first: bool = False,
) -> Tensor:
    assert (
        hop_time > 0 or frame_hz > 0
    ), "vad_list_to_onehot requires `frame_hz` or `hop_time`"

    if frame_hz > 0:
        hop_time = 1 / frame_hz

    n_frames = time_to_frames(duration, hop_time)
    vad_tensor = torch.zeros((n_frames, 2))
    for ch, ch_vad in enumerate(vad_list):
        for v in ch_vad:
            s = time_to_frames(v[0], hop_time)
            e = time_to_frames(v[1], hop_time)
            vad_tensor[s:e, ch] = 1.0

    if channel_first:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


def vad_fill_silences(
    vad: Tensor, max_fill_time: float = 0.02, frame_hz: float = 50
) -> Tensor:
    assert vad.ndim == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    assert vad.shape[-1] == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    max_fill_frame = round(max_fill_time * frame_hz)
    for ch in range(2):
        starts, dur, on_off = find_island_idx_len(vad[:, ch])
        sil_starts = starts[on_off == 0]
        sil_durs = dur[on_off == 0]
        w = torch.where(sil_durs <= max_fill_frame)[0]
        fill_starts = sil_starts[w]
        fill_durs = sil_durs[w]
        for s, d in zip(fill_starts, fill_durs):
            vad[s : s + d, ch] = 1.0
    return vad


def vad_omit_spikes(
    vad: Tensor, max_omit_time: float = 0.02, frame_hz: float = 50
) -> Tensor:
    assert vad.ndim == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    assert vad.shape[-1] == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    max_omit_frame = round(max_omit_time * frame_hz)
    for ch in range(2):
        starts, dur, on_off = find_island_idx_len(vad[:, ch])
        sil_starts = starts[on_off == 1]
        sil_durs = dur[on_off == 1]
        w = torch.where(sil_durs <= max_omit_frame)[0]
        fill_starts = sil_starts[w]
        fill_durs = sil_durs[w]
        for s, d in zip(fill_starts, fill_durs):
            vad[s : s + d, ch] = 0.0
    return vad

