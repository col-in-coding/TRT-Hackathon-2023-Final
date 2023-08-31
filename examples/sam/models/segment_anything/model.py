from tensorrt_llm import Module
from tensorrt_llm.layers import LayerNorm
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
