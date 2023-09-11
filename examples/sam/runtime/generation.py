import torch
from dataclasses import dataclass, field
from tensorrt_llm.runtime.generation import _Runtime
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.logger import logger
from tensorrt_llm._utils import trt_dtype_to_torch


@dataclass
class ModelConfig:
    model_name: str = ""


class SAMGenerationSession(object):

    _model_config: ModelConfig
    mapping: Mapping
    runtime: _Runtime
    device: torch.device
    batch_size: int
    buffer_allocated: bool
    debug_mode: bool

    def __init__(self,
                 model_config: ModelConfig,
                 engine_buffer,
                 mapping: Mapping,
                 debug_mode=False):
        assert isinstance(model_config, ModelConfig)
        self._model_config = model_config
        runtime = _Runtime(engine_buffer, mapping)

        self.mapping = mapping
        self.runtime = runtime
        self.device = torch.device(
            f'cuda:{runtime.runtime_rank % mapping.gpus_per_node}')
        torch.cuda.set_device(self.device)
        self.debug_mode = debug_mode

        self.buffer = None
        self.buffer_allocated = False

        # validate engine input output names
        expected_tensor_names = ["input", "output"]
        found_tensor_names = [
            runtime.engine.get_tensor_name(i)
            for i in range(runtime.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected, to use this GenerationSession, " \
                    "you need to use GPTLMHeadModel.prepare_inputs to create TRT Network inputs."
            )

    def setup(self, batch_size):
        self.batch_size = batch_size

        def tensor_dtype(name):
            # return torch dtype given tensor name for convenience
            dtype = trt_dtype_to_torch(
                self.runtime.engine.get_tensor_dtype(name))
            return dtype

        self.buffer = {
            'output':
            torch.empty((batch_size, 64, 64, 1280),
                        dtype=tensor_dtype('output'),
                        device=self.device)
        }

        self.buffer_allocated = True

    def _get_context_shape_buffer(self, inp):
        ctx_shape = {
            "input_image": inp.shape
        }
        ctx_buffer = {
            "input_image": inp.contiguous(),
            "output": self.buffer['output']
        }
        return ctx_shape, ctx_buffer

    def encode(self, inp):
        batch_size = inp.size(0)
        assert batch_size == self.batch_size, \
            "Given batch size is different from the one used in setup()," \
            "rerun the setup function with the new batch size to avoid buffer overflow."

        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        context = self.runtime.context_0
        ctx_shape, ctx_buffer = self._get_context_shape_buffer(inp)
        self.runtime._set_shape(context, ctx_shape)
        self.runtime._set_buffer(context, ctx_buffer)

        # currently use torch's current stream, so must let TRT enqueue use same stream here
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.runtime._run(context, stream)
        if not ok:
            raise RuntimeError('Executing TRT engine failed!')
        if self.debug_mode:
            torch.cuda.synchronize()

        output = self.buffer['output']

        return output
