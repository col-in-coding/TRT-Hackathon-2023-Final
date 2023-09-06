import json
from typing import Any
import torch
import tensorrt_llm
import tensorrt as trt
from pathlib import Path
from runtime.generation import ModelConfig, SAMGenerationSession

# from tensorrt_base_v4 import TensorrtBaseV4


# class TRTModel(TensorrtBaseV4):
#     def __init__(self, plan_file_path, gpu_id=0):
#         profiles_max_shapes = [{
#             0: (1, 3, 1024, 1024),
#             1: (1, 64, 64, 1280)
#         }]
#         super().__init__(plan_file_path, profiles_max_shapes, gpu_id)

#     def __call__(self, inp):
#         profile_num = 0
#         output_shape = (1, 64, 64, 1280)
#         bufferH = [
#             inp.float().contiguous(),
#             torch.empty(output_shape, dtype=torch.float32, device=inp.device)
#         ]
#         trt_outputs = self.do_inference(bufferH, profile_num)
#         return trt_outputs[0]


if __name__ == "__main__":

    data = torch.load("data.pt")
    inp = data["inp"]
    out = data["x"]
    print("===> valid: ", out.sum())

    engine_name = "sam_vit_h.engine"
    engine_dir = "sam_outputs"
    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    engine_path = engine_dir / engine_name

    # # run trt
    # trt_model = TRTModel(plan_file_path=engine_path)
    # image_embeddings  = trt_model(inp)
    # print("===> trt: ", image_embeddings.sum())

    # run trt-llm
    with open(config_path, 'r') as f:
        config = json.load(f)
    dtype = config['builder_config']['precision']

    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    runtime_mapping = tensorrt_llm.Mapping(world_size=1, rank=0)
    model_config = ModelConfig(model_name="sam_vit_h")

    img_encoder = SAMGenerationSession(model_config, engine_buffer, runtime_mapping, debug_mode=True)
    img_encoder.setup(inp.size(0))
    image_embeddings = img_encoder.encode(inp)
    torch.cuda.synchronize()

    print(image_embeddings.shape, image_embeddings.sum())
