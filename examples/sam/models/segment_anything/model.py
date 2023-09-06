from typing import Optional, Tuple, Type
from tensorrt_llm import Module
from tensorrt_llm.layers import LayerNorm, Conv2d
from tensorrt_llm.functional import Tensor
from tensorrt_llm._utils import str_dtype_to_trt


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
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        dtype: str = "float32"
    ) -> None:
        super().__init__()
        dtype = str_dtype_to_trt('float32')
        self.dtype = dtype
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            dtype=dtype
        )

    def forward(self, x: Tensor):
        x = self.patch_embed(x)
        # if self.pos_embed is not None:
        #     x = x + self.pos_embed

        # for blk in self.blocks:
        #     x = blk(x)

        # x = self.neck(x.permute(0, 3, 1, 2))

        x.mark_output("output", self.dtype)
        return x

    def prepare_inputs(self):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        inp = Tensor(name="input_image", dtype=self.dtype, shape=[1, 3, 1024, 1024])
        return (inp, )


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
        dtype: str = "float32"
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
