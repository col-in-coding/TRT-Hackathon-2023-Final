import cv2
import json
import tqdm
import torch
import numpy as np
import argparse
import dataclasses
import tensorrt_llm
import onnxruntime
from pathlib import Path
from runtime.generation import ModelConfig, SAMGenerationSession


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    input_image: str
    point_coords: tuple
    point_labels: tuple
    engine_dir: str = "sam_outputs"
    engine_name: str = "sam_vit_h.engine"
    mask_decoder: str = "mask_decoder.onnx"

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--input-image',
                            type=str,
                            help='file name of input image',
                            default="dog.jpg")
                            # required=True)
        parser.add_argument('--point-coords',
                            nargs='+',
                            type=int,
                            help='coordinates for selected point, example: --point-coords W0 H0 W1 H1')
                            # required=True)
        parser.add_argument('--point-labels',
                            nargs='+',
                            type=int,
                            help='labels for selected point, 1 is for selected, 0 is for unselected',)
                            # required=True)
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


def preprocess(image_hwc_bgr: np.ndarray, point_coords):
    old_w = image_hwc_bgr.shape[1]
    old_h = image_hwc_bgr.shape[0]
    scale = 1024. / max(image_hwc_bgr.shape[0], image_hwc_bgr.shape[1])
    neww = int(image_hwc_bgr.shape[1] * scale + 0.5)
    newh = int(image_hwc_bgr.shape[0] * scale + 0.5)
    image = cv2.resize(image_hwc_bgr, dsize=(neww, newh), interpolation=cv2.INTER_LINEAR)

    image = image[:, :, ::-1]
    image = image.transpose(2, 0, 1)

    pixel_mean = np.asarray([123.675, 116.28, 103.53]).reshape(-1, 1, 1)
    pixel_std = np.asarray([58.395, 57.12, 57.375]).reshape(-1, 1, 1)
    image = (image - pixel_mean) / pixel_std

    _, h, w = image.shape
    image_new = torch.zeros(1, 3, 1024, 1024)
    image_new[:, :, :h, :w] = torch.from_numpy(image)

    coords = np.asarray(point_coords).reshape(1, -1, 2).astype(np.float32)

    coords[..., 0] = coords[..., 0] * (neww / old_w)
    coords[..., 1] = coords[..., 1] * (newh / old_h)
    return image_new, coords


def get_image_embedding(input_img, args):
    # data = torch.load("data.pt")
    # data2 = torch.load("data2.pt")

    # inp = data["inp"]
    # out = data["x"]
    # print("===> inp: ", inp.shape, inp.dtype, inp.device, inp.min(), inp.max())
    # print("===> valid: ", out.sum())
    # return out

    inp = input_img.cuda()
    # print(inp.shape, inp.dtype, inp.device, inp.min(), inp.max())
    # exit(0)
    engine_name = args.engine_name
    engine_dir = args.engine_dir
    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'

    # # run trt
    # engine_path = engine_dir / engine_name
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

    # for _ in tqdm.tqdm(range(100)):
    #     image_embeddings = img_encoder.encode(inp)
    #     torch.cuda.synchronize()

    print(image_embeddings.shape, image_embeddings.sum())
    return image_embeddings


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.cpu().numpy()


def decode_mask(image_embeddings, orig_im_size, point_coords, args):
    point_labels = torch.tensor(args.point_labels).reshape(1, -1).float()

    ort_inputs = {
        "image_embeddings": image_embeddings.float(),
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": torch.randn(1, 1, 256, 256, dtype=torch.float),
        "has_mask_input": torch.tensor([0], dtype=torch.float),
        "orig_im_size": torch.tensor(orig_im_size, dtype=torch.float),
    }
    # set cpu provider default
    providers = ["CPUExecutionProvider"]
    ort_inputs = {k: to_numpy(v) for k, v in ort_inputs.items()}
    ort_session = onnxruntime.InferenceSession(args.mask_decoder, providers=providers)
    result = ort_session.run(None, ort_inputs)
    return result[0][0]


def main(args):
    input_image = cv2.imread(args.input_image)
    # H, W
    orig_im_size = (input_image.shape[0], input_image.shape[1])
    inp, point_coords = preprocess(input_image, args.point_coords)

    image_embeddings = get_image_embedding(inp, args)
    masks = decode_mask(image_embeddings, orig_im_size, point_coords, args)
    masks = masks > 0.0

    mask = masks[0]
    masks = input_image * mask[..., None]
    output_path = "output.png"
    cv2.imwrite(output_path, masks)
    print(f"output image saved at: {output_path}")


if __name__ == "__main__":
    args = ProgArgs.parse()
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    main(args)
