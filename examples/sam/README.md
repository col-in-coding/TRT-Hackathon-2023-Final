# SAM

请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：

- 介绍本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目（请给出上述链接），并介绍具体选题是什么（参见“选题得分”小节，应为如下之一：1，2，3，4，2+4，3+4）
    - 如果是优化新模型，原始模型的名称及链接，并对该模型做个简要介绍
- 优化效果（例如给出精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明
  - 请做到只要逐行运行你给的命令，就能把代码跑起来

This document explains how to build the [SAM](https://github.com/facebookresearch/segment-anything) model using TensorRT-LLM and run on single GPU.

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image.

Jobs: 用TensorRT-LLM实现新模型;为TensorRT-LLM添加新feature，或者在模型上启用了现有feature. (2+4) from [Guide](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/Hackathon2023/HackathonGuide.md)

### Usage
Start Docker Environment

```bash
export PROJ_PATH=`pwd`
docker run -it --rm --name=trt2023 \
-v $PROJ_PATH:/workspace -w /workspace --gpus all --user root \
registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1 bash
```

Install segment-anything and download weights from SAM repository

```bash
rm -rf sam && git clone https://github.com/facebookresearch/segment-anything sam
cd sam && pip install -e . && cd -
mkdir -p sam_models
pushd sam_models && rm sam_vit_h_4b8939.pth && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && popd

```

Convert params

```bash
python params_convert.py
```

### 主要开发工作

#### 开发工作的难点

请在这一节里总结你的工作难点与亮点。
- 如果使用 TensorRT 进行优化，请介绍一下在模型在导出时、或用polygraphy/trtexec解析时，或在使用TensorRT中，遇到了什么问题并解决了。换句话说，针对这个模型，我们为什么需要额外的工程手段。
- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

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

提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。

对于每个bug，请标记上hackathon2023标签，并写好正文：

- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 9.0.0.1可复现。
- 然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 9.0.0.1
    - Versions of CUDA, CUBLAS, CuDNN used
    - Container used
    - NVIDIA driver version
  - Reproduction Steps
    - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
  - Expected Behavior
    - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
  - Actual Behavior
    - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
  - Additional Notes
    - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

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

### 经验与体会（可选）

欢迎在这里总结经验，抒发感慨。