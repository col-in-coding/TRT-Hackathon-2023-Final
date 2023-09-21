import cv2
import json
import tqdm
import time
import torch
import numpy as np
import argparse
import dataclasses
import tensorrt_llm
import onnxruntime
from pathlib import Path
from functools import partial
from collections import OrderedDict
from runtime.generation import ModelConfig, SAMGenerationSession
from segment_anything.modeling.image_encoder import ImageEncoderViT


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    engine_dir: str = "sam_outputs"
    engine_name: str = "sam_vit_h.engine"
    mask_decoder: str = "mask_decoder.onnx"

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--engine-dir',
                            type=str,
                            help='directory name of engine',
                            default="sam_outputs")
        parser.add_argument('--engine-name',
                            type=str,
                            help='name of the output engne',
                            default="sam_vit_h.engine")
        parser.add_argument('--mask-decoder',
                            type=str,
                            help='onnx path of the image decoder',
                            default="mask_decoder.onnx")

        return ProgArgs(**vars(parser.parse_args(args)))


def fetch_image_encoder_params(sam_ckpt):
    state_dict = torch.load(sam_ckpt)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    new_dict = OrderedDict()
    for key in state_dict.keys():
        if key.startswith("image_encoder."):
            new_key = key.replace("image_encoder.", "")
            new_dict[new_key] = state_dict[key]
    return new_dict


@torch.inference_mode()
def run_torch(inp):
    sam_ckpt = "sam_models/sam_vit_h_4b8939.pth"
    image_encoder = ImageEncoderViT(
            depth=32,
            embed_dim=1280,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
            window_size=14,
            out_chans=256,
        )
    image_encoder = image_encoder.cuda()
    state_dict = fetch_image_encoder_params(sam_ckpt)
    image_encoder.load_state_dict(state_dict)

    # warmup
    for _ in range(10):
        image_embeddings = image_encoder(inp)
        torch.cuda.synchronize()

    start = time.time()
    round = 10
    for _ in tqdm.tqdm(range(round)):
        image_embeddings = image_encoder(inp)
        torch.cuda.synchronize()
    end = time.time()
    t = end - start
    print(f"===> torch time spend: {t*100} ms")

    return image_embeddings


def run_trt_llm(inp, args):

    engine_name = args.engine_name
    engine_dir = args.engine_dir
    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'

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

    # warmup
    for _ in range(10):
        image_embeddings = img_encoder.encode(inp)
        torch.cuda.synchronize()

    start = time.time()
    round = 10
    for _ in tqdm.tqdm(range(round)):
        image_embeddings = img_encoder.encode(inp)
        torch.cuda.synchronize()
    end = time.time()
    t = end - start
    print(f"===> trt llm time spend: {t*100} ms")

    return image_embeddings


def main(args):
    inp = torch.randn(1, 3, 1024, 1024).cuda()
    torch_res = run_torch(inp)
    trt_res = run_trt_llm(inp, args)

    print(torch_res.shape, torch_res.sum())
    print(trt_res.shape, trt_res.sum())

    print("===> absolute err... ",
          f"max: {(torch_res - trt_res).abs().max()}, "
          f"mid: {(torch_res - trt_res).abs().median()}, ",
          f"min: {(torch_res - trt_res).abs().min()}, ",)

    print("===> relative err... ",
          f"max: {((torch_res - trt_res)/torch_res).abs().max()}",
          f"mid: {((torch_res - trt_res)/torch_res).abs().median()}",
          f"min: {((torch_res - trt_res)/torch_res).abs().min()}")


if __name__ == "__main__":
    args = ProgArgs.parse()
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    main(args)
