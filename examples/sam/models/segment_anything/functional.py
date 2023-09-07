import tensorrt as trt
from tensorrt_llm.functional import Tensor, _create_tensor
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
