import torch
import time
import json
import tensorrt as trt
from pathlib import Path

import tensorrt_llm
from tensorrt_llm import Module
from tensorrt_llm.layers import LayerNorm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.network import net_guard
from tensorrt_llm.builder import Builder
from tensorrt_llm.functional import Tensor
from tensorrt_llm.logger import logger


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(1280, elementwise_affine=True)

    def forward(self, x):
        return self.layernorm(x)


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


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')

if __name__ == "__main__":
    # ############# Pytorch inference
    # torch.manual_seed(0)
    # inp = torch.randn(1, 64, 64, 1280)
    # torch_model = TorchModel()
    # torch_out = torch_model(inp)
    # print(torch_out.shape, torch_out.sum())

    engine_dir = "sam_outputs"
    # # Build TRT network
    # trt_llm_model = TestModel()
    # # Module -> Network
    # builder = Builder()
    # builder_config = builder.create_builder_config(
    #     name="sam",
    #     precision="float32",
    #     timing_cache=None,
    #     tensor_parallel=1,
    #     parallel_build=False,
    # )
    # network = builder.create_network()
    # engine_name = "test.engine"
    # network.trt_network.name = engine_name

    # with net_guard(network):
    #     # Prepare
    #     network.set_named_parameters(trt_llm_model.named_parameters())
    #     # Forward
    #     inputs = trt_llm_model.prepare_inputs()
    #     trt_llm_model(*inputs)

    # # Network -> Engine
    # # engine = None
    # engine = builder.build_engine(network, builder_config)
    # config_path = engine_dir + '/config.json'
    # builder.save_config(builder_config, config_path)
    # serialize_engine(engine, engine_name)

    # run
    engine_name = "test.engine"
    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    dtype = config['builder_config']['precision']

    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    runtime_mapping = tensorrt_llm.Mapping(world_size=1, runtime_rank=0)
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)