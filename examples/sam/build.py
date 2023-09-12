import time
import tensorrt_llm
import numpy as np
from pathlib import Path
from tensorrt_llm.network import net_guard
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm._utils import str_dtype_to_np
from models.segment_anything.model import ImageEncoderViT

logger.set_level("info")


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def load_from_ft(tensorrt_llm_sam, dir_path, dtype='float32'):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()
    np_dtype = str_dtype_to_np(dtype)

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            print("===> ", p, t.shape)
            if shape is not None:
                t = t.reshape(shape)
            return t
        logger.error(f"Param file not found: {p}")
        return None

    tensorrt_llm_sam.patch_embed.proj.weight.value = fromfile(
        dir_path, "patch_embed.proj.weight.bin", (1280, 3, 16, 16))
    tensorrt_llm_sam.patch_embed.proj.bias.value = fromfile(
        dir_path, "patch_embed.proj.bias.bin")
    tensorrt_llm_sam.pos_embed.value = fromfile(
        dir_path, "pos_embed.bin", (1, 64, 64, 1280)
    )

    global_attn_indexes = [7, 15, 23, 31]

    depth = 32
    for i in range(depth):
        # Blocks
        tensorrt_llm_sam.blocks[i].norm1.weight.value = fromfile(
            dir_path, f"blocks.{i}.norm1.weight.bin"
        )
        tensorrt_llm_sam.blocks[i].norm1.bias.value = fromfile(
            dir_path, f"blocks.{i}.norm1.bias.bin"
        )
        if i in global_attn_indexes:
            tensorrt_llm_sam.blocks[i].attn.rel_pos_h.value = fromfile(
                dir_path, f"blocks.{i}.attn.rel_pos_h.bin", (127, 80)
            )
            tensorrt_llm_sam.blocks[i].attn.rel_pos_w.value = fromfile(
                dir_path, f"blocks.{i}.attn.rel_pos_w.bin", (127, 80)
            )
        else:
            tensorrt_llm_sam.blocks[i].attn.rel_pos_h.value = fromfile(
                dir_path, f"blocks.{i}.attn.rel_pos_h.bin", (27, 80)
            )
            tensorrt_llm_sam.blocks[i].attn.rel_pos_w.value = fromfile(
                dir_path, f"blocks.{i}.attn.rel_pos_w.bin", (27, 80)
            )

        tensorrt_llm_sam.blocks[i].attn.qkv.weight.value = fromfile(
            dir_path, f"blocks.{i}.attn.qkv.weight.bin", (3840, 1280)
        )
        tensorrt_llm_sam.blocks[i].attn.qkv.bias.value = fromfile(
            dir_path, f"blocks.{i}.attn.qkv.bias.bin"
        )
        tensorrt_llm_sam.blocks[i].attn.proj.weight.value = fromfile(
            dir_path, f"blocks.{i}.attn.proj.weight.bin", (1280, 1280)
        )
        tensorrt_llm_sam.blocks[i].attn.proj.bias.value = fromfile(
            dir_path, f"blocks.{i}.attn.proj.bias.bin"
        )
        tensorrt_llm_sam.blocks[i].norm2.weight.value = fromfile(
            dir_path, f"blocks.{i}.norm2.weight.bin"
        )
        tensorrt_llm_sam.blocks[i].norm2.bias.value = fromfile(
            dir_path, f"blocks.{i}.norm2.bias.bin"
        )
        tensorrt_llm_sam.blocks[i].mlp.lin1.weight.value = fromfile(
            dir_path, f"blocks.{i}.mlp.lin1.weight.bin", (5120, 1280)
        )
        tensorrt_llm_sam.blocks[i].mlp.lin1.bias.value = fromfile(
            dir_path, f"blocks.{i}.mlp.lin1.bias.bin"
        )
        tensorrt_llm_sam.blocks[i].mlp.lin2.weight.value = fromfile(
            dir_path, f"blocks.{i}.mlp.lin2.weight.bin", (1280, 5120)
        )
        tensorrt_llm_sam.blocks[i].mlp.lin2.bias.value = fromfile(
            dir_path, f"blocks.{i}.mlp.lin2.bias.bin"
        )

    tensorrt_llm_sam.neck.conv1.weight.value = fromfile(
        dir_path, "neck.0.weight.bin", (256, 1280, 1, 1)
    )
    tensorrt_llm_sam.neck.norm1.weight.value = fromfile(
        dir_path, "neck.1.weight.bin", (256, 1, 1)
    )
    tensorrt_llm_sam.neck.norm1.bias.value = fromfile(
        dir_path, "neck.1.bias.bin", (256, 1, 1)
    )
    tensorrt_llm_sam.neck.conv2.weight.value = fromfile(
        dir_path, "neck.2.weight.bin", (256, 256, 3, 3)
    )
    tensorrt_llm_sam.neck.norm2.weight.value = fromfile(
        dir_path, "neck.3.weight.bin", (256, 1, 1)
    )
    tensorrt_llm_sam.neck.norm2.bias.value = fromfile(
        dir_path, "neck.3.bias.bin", (256, 1, 1)
    )

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


if __name__ == "__main__":

    engine_dir = "sam_outputs"
    engine_name = "sam_vit_h.engine"
    model_dir = "c-model"
    dtype = "float32"
    engine_dir = Path(engine_dir)
    engine_path = engine_dir / engine_name
    # Build TRT network
    trt_llm_model = ImageEncoderViT(
        img_size=1024,
        patch_size=16,
        embed_dim=1280,
        num_heads=16,
        mlp_ratio=4,
        out_chans=256,
        qkv_bias=True,
        use_rel_pos=True,
        depth=32,
        window_size=14,
        global_attn_indexes=[7, 15, 23, 31]
    )
    load_from_ft(trt_llm_model, model_dir, dtype)

    # Module -> Network
    builder = Builder()
    builder_config = builder.create_builder_config(
        name="sam",
        precision="float32",
        timing_cache=None,
        tensor_parallel=1,
        parallel_build=False,
    )
    network = builder.create_network()
    network.trt_network.name = engine_name

    with net_guard(network):
        # Prepare
        network.set_named_parameters(trt_llm_model.named_parameters())
        # Forward
        inputs = trt_llm_model.prepare_inputs()
        trt_llm_model(*inputs)

    # Network -> Engine
    # engine = None
    engine = builder.build_engine(network, builder_config)
    config_path = engine_dir / 'config.json'
    builder.save_config(builder_config, config_path)
    serialize_engine(engine, engine_path)
