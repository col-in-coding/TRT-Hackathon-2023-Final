import numpy as np
import tensorrt as trt
from typing import Tuple
from tensorrt_llm.functional import Tensor, _create_tensor, einsum, interpolate, constant, slice
from tensorrt_llm._common import default_trtnet


def pad(input: Tensor, pads):
    """
    pads: pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    """
    ndim = input.ndim()

    diff = ndim - len(pads) // 2
    assert diff <= 0

    start = []
    shape = []
    for i in range(ndim):
        idx = i - diff
        pre = pads[idx]
        post = pads[len(pads) // 2 + idx]
        assert pre >= 0
        assert post >= 0
        start.append(-pre)
        shape.append(pre + post + input.size()[i])
    stride_values = [1 for _ in range(ndim)]

    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=start,
                                       shape=shape,
                                       stride=stride_values)
    layer.mode = trt.SampleMode.FILL
    return _create_tensor(layer.get_output(0), layer)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = pad(x, (0, 0, 0, 0, 0, pad_w, pad_h, 0))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view((B, Hp // window_size, window_size, Wp // window_size, window_size, C),
               zero_is_placeholder=False)
    windows = x.permute((0, 1, 3, 2, 4, 5))
    windows = windows.view((-1, window_size, window_size, C),
                           zero_is_placeholder=False)
    return windows, (Hp, Wp)


def gather(input: Tensor, indices: np.ndarray):
    indices = indices.astype(np.int32)
    indices = constant(indices)
    # TODO: try add_gather_v2
    layer = default_trtnet().add_gather(input.trt_tensor,
                                        indices.trt_tensor,
                                        0)
    return _create_tensor(layer.get_output(0), layer)


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = interpolate(
            rel_pos.view((1, rel_pos.shape[0], -1)).permute((0, 2, 1)),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.view((-1, max_rel_dist)).permute((1, 0))
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = np.arange(q_size).reshape(-1, 1) * max(k_size / q_size, 1.0)
    k_coords = np.arange(k_size).reshape(1, -1) * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    # return rel_pos_resized[relative_coords.long()]
    return gather(rel_pos_resized, relative_coords)


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.view((B, q_h, q_w, dim))
    rel_h = einsum("bhwc,hkc->bhwk", (r_q, Rh))
    rel_w = einsum("bhwc,wkc->bhwk", (r_q, Rw))

    attn = attn.view((B, q_h, q_w, k_h, k_w))

    # attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    rel_h = rel_h.view(([rel_h.size(0), rel_h.size(1), rel_h.size(2), rel_h.size(3), 1]))
    rel_w = rel_w.view(([rel_w.size(0), rel_w.size(1), rel_w.size(2), 1, rel_w.size(3)]))
    attn = attn + rel_h + rel_w

    attn = attn.view((B, q_h * q_w, k_h * k_w))

    return attn


def window_unpartition(
    windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view((B, Hp // window_size, Wp // window_size, window_size, window_size, -1))
    x = x.permute((0, 1, 3, 2, 4, 5)).view((B, Hp, Wp, -1))

    if Hp > H or Wp > W:
        x = slice(x, (0, 0, 0, 0), (x.shape[0], H, W, x.shape[3]))
    return x
