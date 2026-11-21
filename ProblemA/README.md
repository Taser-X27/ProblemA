# CUHK-X：ViT-S与CNN重新训练工程

ProblemA 从公开预训练权重开始复现 ViT-S 和 CNN 动作分类。源码包含固定的 40 类融合权重。准备好原始数据和预训练权重后，复现流程会生成 `cache/` 和 `results/`。

## 从零复现当前结果

在 `ProblemA/` 内依次运行以下命令。原实验使用 Python 3.12、PyTorch `2.11.0+cu128`、torchvision `0.26.0+cu128` 和 RTX 5070。ViT 蒸馏与训练需要支持 BF16 的 NVIDIA GPU。Windows 可使用原 `pt_qwen` 环境。Linux 需要安装对应平台的 CUDA 版 PyTorch、torchvision 和 `requirements.txt` 中的依赖，项目会使用 Linux 环境已安装的包。

```bash
# 1. H→S 蒸馏，30 轮
python -u distill_vit.py --epochs 30 --batch-size 1 --effective-batch-size 32

# 2. 训练当前融合实验使用的联合 ViT，60 轮，按验证集选最佳权重
python -u train_vit.py --modes joint --epochs 60 --batch-size 4 --joint-batch-size 1 --effective-batch-size 4 --encoder-batch-size 2 --output-dir results/vit_batch1_accum4

# 3. 生成 CNN 裁剪缓存，训练四通道 IR CNN，40 轮
python -u prepare_cnn.py
python -u train_cnn.py --batch-size 8 --grad-accum 2

# 4. 导出验证集与测试集概率。两套数据使用相同的分类权重。
python -u evaluate.py --model vit --split validation --checkpoint results/vit_batch1_accum4/checkpoints/joint_best.pt
python -u evaluate.py --model cnn --split validation --checkpoint results/cnn_ir/best_model.pt
python -u prepare_cnn.py --split test
python -u evaluate.py --model vit --split test --checkpoint results/vit_batch1_accum4/checkpoints/joint_best.pt
python -u evaluate.py --model cnn --split test --checkpoint results/cnn_ir/best_model.pt

# 5. 使用 configs/fusion.json 中的固定权重融合
python fusion.py --check
python -u fusion.py
```


## 从 GitHub 克隆后准备环境

克隆后，将原始数据放到项目根目录的 `dataset/`。预训练权重可用 `python prepare_weights.py` 从上游下载并生成到 `weights/`。权重文件名、下载地址及 SHA-256 见 `configs/assets.json`。原 Windows 环境保留的 `ProblemA_local_assets/` 也可作为本地数据和权重的来源。

在支持 CUDA 12.8 的 Linux + NVIDIA 环境中，可用 Python 3.12 安装原实验对应的 PyTorch 版本，再安装其余依赖。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python prepare_weights.py
python check_project.py --forward
```

准备权重只需 CPU。ViT-H 官方 checkpoint 约 5.05 GB，提取时会同时保留约 2.53 GB 的输出文件，建议预留至少 8 GB 磁盘空间。脚本逐个验证 SHA-256，已通过校验的文件会直接复用。若已手动下载官方 ViT-H 文件，可运行 `python prepare_weights.py --teacher-source /path/to/pytorch_model.bin`。只校验现有权重可运行 `python prepare_weights.py --check`。

上传时保留 `configs/`、`training/`、`upstream/` 的目录结构。`.gitignore` 会排除后来添加的本地依赖、数据、权重和生成产物。

## 目录

```text
dataset/               train 1995、validation 466、历史融合数据 test 470
weights/               官方H/S、CNN初始化与YOLO26n/pose（共5个文件）
configs/               数据路径、初始化文件校验、训练配方
training/              两模型、数据处理及训练循环
upstream/              必要的VideoMAEv2/CNN源码与来源、许可证
vendor/                当前Windows环境需要的第三方依赖
cache/                 运行时生成的人物/手部缓存
results/               新训练权重、日志、曲线和验证结果
train_vit.py           训练ViT-S
distill_vit.py         冻结H教师，蒸馏S＋Linear
prepare_cnn.py        检测人物/手部并生成双模态缓存
train_cnn.py          训练CNN
evaluate.py            在validation或test上评估新训练模型
fusion.py              按configs/fusion.json的40类固定权重融合概率
check_project.py       检查路径、权重和数据；不训练
prepare_weights.py     下载预训练权重、提取 ViT-H 骨干并校验
```

`weights/`必须保留：官方ViT-H骨干、ViT-S官方giant蒸馏K710权重、R(2+1)D-34的IG65M→K400权重、YOLO26n、YOLO26n-pose。H文件只提取公开checkpoint的骨干原始张量，去除UCF101分类头和优化器，约2.53GB。H只在蒸馏时使用，不进入最终部署模型。

## 数据流

ViT阶段一：同一视频张量分别送入冻结ViT-H与可训练ViT-S。S输出`[B,1568,384]`，经随机初始化的可学习Linear(384,1280)对齐H的`[B,1568,1280]`，优化原始token MSE；没有分类头，不使用动作标签。S初始化仍用官方giant蒸馏权重，然后在本次train上进一步接受H蒸馏。S的fc_norm不在token路径上，阶段一冻结。

ViT阶段二：载入本次蒸馏最佳S，丢弃Linear，Depth_Color RGB与IR复制RGB各16帧、224×224 → 共享S → 各384维池化特征 → 拼接768维 → 随机MLP → 40类。S与MLP全部参数训练，包括fc_norm；此时不加载H。

CNN：在IR上使用YOLO26n和YOLO26n-pose定位人物与手部 → 从现有裁剪缓存取出人物和手部的IR帧 → 每路IR扩为四通道`[B,4,32,128,128]` → 共享去除时间下采样的R(2+1)D-34 → 基础分类头＋交互Transformer → 40类。全参数训练，BN运行统计冻结。

CNN训练和评估均使用`[IR,IR,IR,IR]`。扩展后按Noah IR分支的四组Kinetics统计量归一化。第一层的前三个卷积核沿用RGB预训练权重，第四个由三个RGB卷积核取均值得到；启动训练时核验并记录。

## 先检查，不启动训练

```powershell
conda activate pt_qwen
cd E:\coding\LLM\ProblemA
python train_vit.py --check
python distill_vit.py --check
python train_cnn.py --check
python check_project.py
```

`python check_project.py --forward`额外执行分类模型CUDA前向；追加`--distill-forward`检查H/S对齐。不反向传播，不创建优化器。蒸馏未完成时`train_vit.py --check`会显示`classification_ready: false`，提示先运行蒸馏。

## 训练接口

```powershell
# 先H→S蒸馏，默认输出results/distill
python -u distill_vit.py

# 再加载本次最佳S，训练分类模型，默认输出results/vit
python -u train_vit.py

# 首次运行先生成缓存；cache/cnn/complete.json已存在时可直接训练
python -u prepare_cnn.py
# 默认输出results/cnn_ir
python -u train_cnn.py
```

| 参数 | ViT-S | CNN |
|---|---|---|
| 分类阶段初始化 | 本次H→S蒸馏最佳学生 | IG65M→K400 R(2+1)D-34 |
| 轮次 | 60 | 40 |
| 骨干学习率 | 5e-5，layer decay0.75 | 1e-5 |
| 新分类头学习率 | 3e-4 | 1e-3，交互层3e-4 |
| 有效batch | 32（实际2对视频＋累积） | 16（实际8＋累积2） |
| Warmup | 6轮 | 2轮 |
| 验证 | 初始、每5轮及末轮，保留最佳 | 初始、每5轮及末轮，保留最佳 |

反向传播只使用`train/`。选最佳轮使用`validation/`，单模型评估默认也使用`validation/`，评估前向关闭梯度。融合阶段导出所需集合的概率，直接应用给定的40类固定λ，不再拟合。H→S蒸馏沿用之前的30轮配方：S LR1e-5、Linear LR1e-4、warmup3轮、有效batch32个单模态视频、每5轮验证，以验证token MSE最低选择最佳S。阶段二默认从`results/distill/checkpoints/best.pt`加载，而不是直接把官方S用于分类，也不使用历史第30轮学生。

数据根目录由`configs/dataset.json`控制。`configs/training.json`记录并提供distill、vit、cnn三个阶段的默认配方。命令行参数可用`--help`查看；CNN目前固定使用配置中的轮次与学习率，没有`--epochs`/`--lr`命令行选项。

```powershell
python train_vit.py --help
python train_cnn.py --help
python -u train_vit.py --epochs 80 --output-dir results/vit_80ep
```

新实验使用新的`--output-dir results/实验名`。已有目录不会直接覆盖；中断恢复须保留相同参数与输出目录，追加`--resume`：

```powershell
python -u train_vit.py --resume
python -u distill_vit.py --resume
python -u train_cnn.py --resume
```

CNN裁剪缓存位于`cache/cnn/`，约10.32GB，磁盘格式为`[Depth_R,Depth_G,Depth_B,IR]`。四通道IR模型直接取第4个通道。`cache/cnn/`和`cache/cnn_test/`的索引路径均相对于数据集根目录，复制同一份数据集和缓存到新目录后，可通过`--data-root`指定数据位置并复用裁剪数据。旧版完整缓存可用`python prepare_cnn.py`或`python prepare_cnn.py --split test`核验并更新索引。更新保留已有的`.npy`文件。如果数据内容、检测器或预处理变更，应重新准备缓存。

## 结果与日志

| 模型 | 最佳权重 | 断点 | 日志与曲线 |
|---|---|---|---|
| H→S蒸馏 | `results/distill/checkpoints/best.pt` | `checkpoints/last.pt` | `run.log`、`training.jsonl`、`distillation_curves.png/.svg` |
| ViT-S | `results/vit/checkpoints/joint_best.pt` | `checkpoints/joint_last.pt` | `run.log`、`training_joint.jsonl`、`curves_joint.png/.svg/.csv` |
| CNN | `results/cnn_ir/best_model.pt` | `last_checkpoint.pt` | `train.log`、`history.jsonl`、`training_curves.png/.svg/.csv` |

两者均保存最佳验证点的逐视频预测概率、40类统计及被试统计；CNN另保存末轮权重。训练日志中的train准确率包含训练增强，不能直接与干净验证集准确率等同。

## 单模型评估与错例分析

```powershell
python -u evaluate.py --model vit --checkpoint results/vit/checkpoints/joint_best.pt
python -u evaluate.py --model cnn --checkpoint results/cnn_ir/best_model.pt
```

两条命令均评估`validation/`，输出到`results/eval_vit_validation/`或`results/eval_cnn_validation/`。`metrics.json`保存按类别和被试的指标，`predictions.jsonl`中`correct: false`的记录可用于逐视频错例分析。重复评估可指定新的`--output-dir`。

`evaluate.py --split eval`是`--split validation`的别名。`--split test`读取`dataset/test/`，默认写到`results/eval_vit_test/`或`results/eval_cnn_test/`。每次导出同时记录类别顺序、checkpoint哈希和预测文件哈希，供融合时核对。validation和test必须为每个模型使用同一份分类checkpoint。

## 40类固定权重概率融合

`configs/fusion.json`按类别编号0～39记录ViT/CNN权重。`fusion.py`不进行反向传播、坐标搜索或权重拟合，也不读取历史融合结果中的系数。

两个模型的类别顺序必须与权重表一致。对所有视频的每个输出类别计算：

```text
s[k] = λ[k] * p_vit[k] + (1 - λ[k]) * p_cnn[k]
p = s / s.sum()
```

λ为ViT权重，CNN权重为1−λ；每类两路权重之和为1，但逐类加权后的40维分数仍需整体归一化。融合概率只依赖两路模型概率，不需要真实标签；标签仅用于统计准确率。

```powershell
# 1. 导出validation（eval）概率；已有非空输出目录时指定新的--output-dir
python -u evaluate.py --model vit --split eval --checkpoint results/vit/checkpoints/joint_best.pt
python -u evaluate.py --model cnn --split eval --checkpoint results/cnn_ir/best_model.pt

# 2. 如需test，先准备独立缓存，再用相同的两个模型导出概率
python -u prepare_cnn.py --split test
python -u evaluate.py --model vit --split test --checkpoint results/vit/checkpoints/joint_best.pt
python -u evaluate.py --model cnn --split test --checkpoint results/cnn_ir/best_model.pt

# 3. 核对四份预测并直接融合；只需CPU，不启动训练
python fusion.py --check
python -u fusion.py
```

默认融合validation和test，输出到`results/fusion_fixed_validation_test/`。也可单独处理一个集合，无需另一集合的预测文件：

```powershell
python fusion.py --split validation
python fusion.py --split test
```

单集合默认输出目录分别为`results/fusion_fixed_validation/`与`results/fusion_fixed_test/`。`--split eval`等同于`--split validation`。

脚本按样本ID对齐两路预测，核对类别顺序、标签、被试、帧数、原视频身份、checkpoint及预测文件哈希。每份预测的同目录需保留`evaluate.py`生成的`config.json`。融合本身不加载模型、不生成缓存。

| 文件 | 内容 |
|---|---|
| `selection.json` | 固定40类权重、类别顺序、图表来源、输入与checkpoint哈希；标记未进行拟合 |
| `lambda_by_class.csv` | 每类ViT/CNN固定权重 |
| `metrics.json` | ViT、CNN、固定逐类融合在各集合与合并集合上的指标 |
| `predictions_combined.jsonl` | 所选样本的逐视频融合结果与40类概率 |
| `predictions_validation.jsonl`、`predictions_test.jsonl` | 所选集合的逐视频融合结果 |
| `per_subject.json` | 分被试融合指标 |
| `report.md`、`fusion.log`、`verification.json` | 汇总报告、日志及保存结果核验 |

图表说明原权重在validation466段和test470段上共同拟合。因此这两个集合的融合成绩不属于独立留出测试成绩。本次仅应用固定表，不重新拟合；`test/`不参与神经网络的梯度训练。

自定义输入与输出目录（已有结果不会覆盖）：

```powershell
python fusion.py --vit-eval results/eval_vit_validation/predictions.jsonl --cnn-eval results/eval_cnn_validation/predictions.jsonl --vit-test results/eval_vit_test/predictions.jsonl --cnn-test results/eval_cnn_test/predictions.jsonl --output-dir results/fusion_fixed_run2
```

在其他推理代码中可直接调用，输入须按权重表顺序排列：

```python
from fusion import normalized_fusion
probabilities = normalized_fusion(vit_probabilities, cnn_probabilities)  # 两路均为[N,40]
predictions = probabilities.argmax(axis=1)
```

## 环境与来源

基础依赖见`requirements.txt`，来源见`THIRD_PARTY.md`。运行训练前可用`prepare_weights.py`准备预训练权重。
