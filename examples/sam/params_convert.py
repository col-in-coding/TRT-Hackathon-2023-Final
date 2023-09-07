import torch
import argparse
import dataclasses
from pathlib import Path
from functools import partial
from collections import OrderedDict
from segment_anything.modeling.image_encoder import ImageEncoderViT
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "sam"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            default="c-model")
                            # required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            default="sam_models/sam_vit_h_4b8939.pth")
                            # required=True)
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")

        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])

        return ProgArgs(**vars(parser.parse_args(args)))


def sam_to_ft_name(orig_name):
    return orig_name


def fetch_image_encoder_params(sam_ckpt):
    state_dict = torch.load(sam_ckpt, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    new_dict = OrderedDict()
    for key in state_dict.keys():
        if key.startswith("image_encoder."):
            new_key = key.replace("image_encoder.", "")
            new_dict[new_key] = state_dict[key]
    return new_dict


@torch.no_grad()
def run_conversion(args):
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = ImageEncoderViT(
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
        out_chans=256
    )
    state_dict = fetch_image_encoder_params(args.in_file)
    model.load_state_dict(state_dict)

    storage_type = str_dtype_to_torch(args.storage_type)

    for name, param in model.named_parameters():
        # if "weight" not in name and "bias" not in name:
        #     print("useless name: ", name)
        #     continue
        print("===> ", name)
        ft_name = sam_to_ft_name(name)

        torch_to_numpy(param.to(storage_type).cpu()).tofile(
            save_dir / f"{ft_name}.bin"
        )


if __name__ == "__main__":
    args = ProgArgs.parse()
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    run_conversion(args)
