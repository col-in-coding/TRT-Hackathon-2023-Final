import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import torch
from tqdm import tqdm
from pathlib import Path
from torch_model import TorchModel
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
                            default="state_dict.ckpt")
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

@torch.no_grad()
def run_conversion(args):
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = TorchModel()
    model.load_state_dict(torch.load("state_dict.ckpt", map_location="cpu"))

    storage_type = str_dtype_to_torch(args.storage_type)

    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            print("useless name: ", name)
            continue
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