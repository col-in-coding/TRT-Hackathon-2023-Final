# NVIDIA TensorRT Hackathon 2023

### 简介

本项目为 NVIDIA TensorRT Hackathon 2023 参赛项目
_Team: Imagination_
- 选题，用TensorRT-LLM实现新模型；为TensorRT-LLM添加新feature，或者在模型上启用了现有feature. (2+4)
- 优化模型 [SAM](https://github.com/facebookresearch/segment-anything) Image Encoder (ViT-H)

优化效果（例如给出精度和加速比），简单给出关键的数字即可

### 环境准备

Start Docker Environment

```bash
# 进入项目根目录
export PROJ_PATH=`pwd`
docker run -it --rm --name=trt2023 \
-v $PROJ_PATH:/workspace -w /workspace --gpus all --user root \
registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1 bash
```

### 运行步骤
安装 Segment Anything，并下载 SAM 模型权重

```bash
# 进入 SAM example 目录
cd examples/sam
# 下载原始项目并安装
rm -rf sam && git clone https://github.com/facebookresearch/segment-anything sam
cd sam && pip install -e . && cd -
# 下载 Vit-h 模型权重
mkdir -p sam_models
pushd sam_models && rm sam_vit_h_4b8939.pth && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && popd

```

模型参数转换

```bash
python params_convert.py
```

构建 Image Encoder 推理引擎

```bash
python build.py
```

构建 Mask Decoder

```bash
python sam/scripts/export_onnx_model.py --checkpoint=sam_models/sam_vit_h_4b8939.pth --model-type='vit_h' --return-single-mask --output=mask_decoder.onnx
```

运行推理

```bash
python run.py --point-coords 800 460 --point-labels 1
```

### 主要开发工作

#### 开发工作的难点

搭建新模型的时候，难点主要在于对大语言模型不是很熟，学习 TensorRT-LLM 中的示例花了不少时间。

其次是并不是所有的算子在 TensorRT-LLM 的库中支持，所以需要通过TensorRT的原生api去实现，对与已经封装好的function或者layer，也要进行结果验证。

最后是要进一步优化推理性能，需要手动融合部分算子。

### 开发与优化过程

1. 熟悉原始的pytorch模型的结构。

2. 通过gpt的例子熟悉 TensorRT-LLM 的模型构建与推理过程。

3. 在正式开发 TensorRT-LLM 新模型之前，先自己做了一个单层结构的小模型，把整个流程（包括权重转换，模型构建，模型推理）再手动实现了一遍。

4. 正式搭建网络，这里我并不是一次搭完所有网络结构的，而是分层次一边搭建一边对比pytorch的推理结果。搭建过程中，同样是参考layers和functional中是否有可以复用的模块。发现还是有一些基本算子需要使用Tensorrt的原生库去实现，比如说padding，以前都是习惯用onnx parser去构建网络，所以只好跑去翻看了onnx parser的源码，知道padding在tensorrt中使用slice实现。

5. 验证整个推理过程的结果。

### 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：

- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 如果选用TensorRT-LLM，请跑summarize任务并使用 [Rouge](https://huggingface.co/spaces/evaluate-metric/rouge) 来对比模型优化前后的精度差距。如果精度良好，原始模型与优化模型的Rouge score的差异一般在1以内。例子见 TensorRT-LLM docker 中 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/summarize.py
  - 如果选用TensorRT，这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
    - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。
    - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力。
- 性能：例如可以用图表展示不同batch size或sequence length下性能加速效果（考虑到可能模型可能比较大，可以只给batch size为1的数据）
  - 一般用原始模型作为baseline
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

### Bug报告（可选）
https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/92

### 送分题答案（可选）

-  root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Single node, single GPU” 部分如下命令的输出（10分）模型为gpt2-medium,
    - `python3 run.py --max_output_len=8`
    - result
    ```
    Input: Born in north-east France, Soyer trained as a
    Output:  chef and eventually became a chef at a
    ```

- /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Summarization using the GPT model” 部分如下命令的rouge 分数（10分）模型为gpt2-medium
    - `python3 summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu --test_hf --batch_size1 --test_trt_llm --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14`
    - result
    ```
    [08/25/2023-07:31:55] Reusing dataset cnn_dailymail (data/ccdv___cnn_dailymail/3.0.0/3.0.0/0107f7388b5c6fae455a5661bcd134fc22da53ea75852027040d8d1e997f101f)
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 647.00it/s]
    [08/25/2023-07:31:55] [TRT] [I] Loaded engine size: 311 MiB
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +6, GPU +10, now: CPU 536, GPU 1100 (MiB)
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 538, GPU 1110 (MiB)
    [08/25/2023-07:31:55] [TRT] [W] TensorRT was linked against cuDNN 8.9.2 but loaded cuDNN 8.9.0
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +310, now: CPU 0, GPU 310 (MiB)
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +1, GPU +8, now: CPU 538, GPU 1744 (MiB)
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 538, GPU 1752 (MiB)
    [08/25/2023-07:31:55] [TRT] [W] TensorRT was linked against cuDNN 8.9.2 but loaded cuDNN 8.9.0
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 310 (MiB)
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 539, GPU 1760 (MiB)
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +1, GPU +10, now: CPU 540, GPU 1770 (MiB)
    [08/25/2023-07:31:55] [TRT] [W] TensorRT was linked against cuDNN 8.9.2 but loaded cuDNN 8.9.0
    [08/25/2023-07:31:55] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 310 (MiB)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
            - Avoid using `tokenizers` before the fork if possible
            - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    [08/25/2023-07:32:00] [TRT-LLM] [I] ---------------------------------------------------------
    [08/25/2023-07:32:00] [TRT-LLM] [I] TensorRT-LLM Generated :
    [08/25/2023-07:32:00] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
    [08/25/2023-07:32:00] [TRT-LLM] [I]
    Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
    [08/25/2023-07:32:00] [TRT-LLM] [I]
    Output : [[' Best died at age 88.']]
    [08/25/2023-07:32:00] [TRT-LLM] [I] ---------------------------------------------------------
    [08/25/2023-07:32:00] [TRT-LLM] [I] ---------------------------------------------------------
    [08/25/2023-07:32:00] [TRT-LLM] [I] HF Generated :
    [08/25/2023-07:32:00] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
    [08/25/2023-07:32:00] [TRT-LLM] [I]
    Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
    [08/25/2023-07:32:00] [TRT-LLM] [I]
    Output : [[' Best died at age 88.']]
    [08/25/2023-07:32:00] [TRT-LLM] [I] ---------------------------------------------------------
    Token indices sequence length is longer than the specified maximum sequence length for this model (1151 > 1024). Running this sequence through the model will result in indexing errors
    [08/25/2023-07:32:19] [TRT-LLM] [I] TensorRT-LLM (total latency: 2.181581735610962 sec)
    [08/25/2023-07:32:19] [TRT-LLM] [I] TensorRT-LLM beam 0 result
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rouge1 : 14.700185379688484
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rouge2 : 3.75886473151702
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rougeL : 12.002855916633356
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rougeLsum : 13.092895095507263
    [08/25/2023-07:32:19] [TRT-LLM] [I] Hugging Face (total latency: 12.6636061668396 sec)
    [08/25/2023-07:32:19] [TRT-LLM] [I] HF beam 0 result
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rouge1 : 14.75593024343394
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rouge2 : 3.3647470801871733
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rougeL : 11.124766996533
    [08/25/2023-07:32:19] [TRT-LLM] [I]   rougeLsum : 13.031128048110618
    ```
