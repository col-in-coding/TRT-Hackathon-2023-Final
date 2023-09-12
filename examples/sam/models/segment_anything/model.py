import numpy as np
import tensorrt as trt
from typing import Optional, Tuple, Union
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.layers import LayerNorm, Conv2d, Linear
from tensorrt_llm.functional import Tensor, matmul, softmax, gelu, pow
from tensorrt_llm._utils import str_dtype_to_trt, str_dtype_to_np

from .functional import window_partition, add_decomposed_rel_pos, window_unpartition


class TestModel(Module):
    def __init__(self):
        super().__init__()
        dtype = str_dtype_to_trt('float32')
        self.dtype = dtype
        self.layernorm = LayerNorm(1280, dtype=dtype, elementwise_affine=True)

    def forward(self, inp):
        out = self.layernorm(inp)
        out.mark_output("output", self.dtype)
        return out

    def prepare_inputs(self):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        inp = Tensor(name="input", dtype=self.dtype, shape=[1, 64, 64, 1280])
        return (inp, )


class ImageEncoderViT(Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        dtype: Union[str, trt.DataType] = "float32"
    ) -> None:
        super().__init__()
        self.dtype = str_dtype_to_trt(dtype)
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            dtype=self.dtype
        )

        self.pos_embed = Parameter(
            shape=(1, img_size // patch_size, img_size // patch_size, embed_dim),
            dtype=self.dtype)

        blocks = []
        # FOR TEST
        # depth = 1
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                dtype=dtype
            )
            blocks.append(block)
        self.blocks = ModuleList(blocks)
        self.neck = Neck(embed_dim, out_chans, dtype=dtype)

    def forward(self, x: Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed.value

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute((0, 3, 1, 2)))

        x.mark_output("output", self.dtype)
        return x

    def prepare_inputs(self):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        inp = Tensor(name="input_image", dtype=self.dtype, shape=[1, 3, 1024, 1024])
        return (inp, )


class Block(Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        dtype: Union[str, trt.DataType] = "float32"
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, dtype=dtype)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            dtype=dtype
        )
        self.norm2 = LayerNorm(dim, dtype=dtype)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), dtype=dtype)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


# TODO: create Attention plugin
class Attention(Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
            dtype: str = "float32"
            ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.dtype = dtype

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias, dtype=str_dtype_to_trt(dtype))
        self.proj = Linear(dim, dim, dtype=str_dtype_to_trt(dtype))

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            np_dtype = str_dtype_to_np(dtype)
            # initialize relative positional embeddings
            self.rel_pos_h = Parameter(np.zeros((2 * input_size[0] - 1, head_dim), dtype=np_dtype), dtype=dtype)
            self.rel_pos_w = Parameter(np.zeros((2 * input_size[1] - 1, head_dim), dtype=np_dtype), dtype=dtype)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).view((B, H * W, 3, self.num_heads, -1)).permute((2, 0, 3, 1, 4))
        # q, k, v with shape (B * nHead, H * W, C)
        # TODO: try option2
        # option 1
        qkv = qkv.view((3, B * self.num_heads, H * W, -1))
        q, k, v = qkv.split(1, dim=0)
        q = q.view((B * self.num_heads, H * W, -1))
        k = k.view((B * self.num_heads, H * W, -1))
        v = v.view((B * self.num_heads, H * W, -1))
        # # option 2
        # qkv = qkv.view((3 * B * self.num_heads, H * W, -1))
        # q, k, v = qkv.split(B * self.num_heads, dim=0)

        attn = matmul(q * self.scale, k.transpose(-2, -1))

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h.value, self.rel_pos_w.value, (H, W), (H, W))

        attn = softmax(attn, dim=-1)
        x = (matmul(attn, v)).view((B, self.num_heads, H, W, -1)).permute((0, 2, 3, 1, 4)).view((B, H, W, -1))
        x = self.proj(x)

        return x


class PatchEmbed(Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        dtype: Union[str, trt.DataType] = "float32"
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride,
            padding=padding, dtype=dtype
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute((0, 2, 3, 1))
        return x


# TODO: create gelu plugin
class MLPBlock(Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        dtype: Union[str, trt.DataType] = "float32"
    ) -> None:
        super().__init__()
        self.lin1 = Linear(embedding_dim, mlp_dim, dtype=dtype)
        self.lin2 = Linear(mlp_dim, embedding_dim, dtype=dtype)
        self.act = gelu

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


# TODO: create LayerNorm2d plugin
class LayerNorm2d(Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
        dtype: Union[str, trt.DataType] = "float32"
    ) -> None:
        super().__init__()
        self.weight = Parameter(shape=(num_channels, 1, 1), dtype=dtype)
        self.bias = Parameter(shape=(num_channels, 1, 1), dtype=dtype)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s1 = pow((x - u), 2.0)
        s = s1.mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        x = self.weight.value * x + self.bias.value
        return x


class Neck(Module):
    def __init__(
        self,
        embed_dim,
        out_chans,
        dtype: Union[str, trt.DataType] = "float32"
    ) -> None:
        super().__init__()
        self.conv1 = Conv2d(embed_dim,
                            out_chans,
                            kernel_size=(1, 1),
                            bias=False,
                            dtype=dtype)
        self.norm1 = LayerNorm2d(out_chans, dtype=dtype)
        self.conv2 = Conv2d(out_chans,
                            out_chans,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            bias=False,
                            dtype=dtype)
        self.norm2 = LayerNorm2d(out_chans, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x
