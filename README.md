# MowerSeg / Mower Perception

面向智能割草机的感知起步仓库。当前真正落地的能力：

- **Task**: `semantic_segmentation`
- **Models**:
  - `segformer_b0_ade20k`（SegFormer-B0 / ADE20K 零样本）
  - `deeplabv3plus_mobilenet_v2`（DeepLabV3+ + MobileNetV2 / PASCAL VOC 零样本）
  - `mit_resnet18_ade20k`（MIT ResNet18-dilated + PPM_deepsup / 完整 ADE20K 权重）
- **Backend**: `torch`（Hugging Face Transformers + PyTorch）
- Web 预览页支持手动切换模型对比效果

架构目标是硬件无关的感知 runtime：

```text
Task（做什么） × Model（用什么网） × Backend（在哪执行）
```

`HorizonBackend` / `RKNNBackend` 已预留接口骨架，尚未接入真实 SDK。

## 产品类别

可割草坪、灌木植被、裸土地形、铺装路面、人与动物、车辆、障碍与结构、水体、天空。

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
export PYTHONPATH=.
python -m mowerseg assets/samples/lawn_path.jpg -o outputs
```

统一入口（推荐）：

```python
from mowerseg import PerceptionEngine

engine = PerceptionEngine(
    task="semantic_segmentation",
    model="segformer_b0_ade20k",
    backend="torch",
    config_path="configs/mower_seg.yaml",
)
result = engine.predict("assets/samples/lawn_path.jpg")
```

兼容旧接口：

```python
from mowerseg import InferenceEngine
engine = InferenceEngine("configs/mower_seg.yaml")
```

同时开 Web 预览：

```bash
npm ci
export PYTHONPATH=.
./scripts/dev.sh
```

浏览器打开 `http://127.0.0.1:43129`。推理 API 在 `http://127.0.0.1:43131`。

## 架构

```text
Camera / Image
      ↓
    Frame
      ↓
PerceptionEngine
      ↓
SemanticSegmentationTask  →  preprocess / postprocess / taxonomy remap
      ↓
   Backend (torch | horizon* | rknn*)
      ↓
SemanticResult → CLI / FastAPI / Robot App
```

目录要点：

| 路径 | 作用 |
|------|------|
| `mowerseg/core/` | Frame / Result / Task / Backend 抽象 |
| `mowerseg/backends/` | Torch 实现 + Horizon/RKNN skeleton |
| `mowerseg/tasks/` | 语义分割（depth/privacy 占位） |
| `mowerseg/models/configs/` | 逻辑模型 yaml 与 artifact 映射 |
| `mowerseg/pipeline/` | `PerceptionEngine` |
| `mowerseg/cli.py` / `server.py` | Adapter（不进 core） |
| `configs/mower_seg.yaml` | 产品类别 + perception 入口 |
| `tools/deployment/` | Build-time 导出/编译占位（非 runtime） |

## 如何新增 Backend

1. 在 `mowerseg/backends/` 实现 `InferenceBackend`（`load` / `infer` / `close`）
2. 在 `backends/registry.py` 注册名称
3. 在对应 model yaml 的 `artifacts:` 下增加编译产物路径
4. 将 `configs/mower_seg.yaml` 里 `perception.backend` 改为新名称

Backend 只处理 tensor / device，不要写 grass / taxonomy 逻辑。

## 如何新增 Task

1. 在 `mowerseg/tasks/<name>/` 增加 preprocess / postprocess / task
2. 定义专用 Result（参考 `SemanticResult` / `DepthResult`）
3. 在 `tasks/registry.py` 注册
4. 增加 model yaml，`task:` 字段指向新 task

## 如何新增 Model

1. 新增 `mowerseg/models/configs/<model_name>.yaml`
2. 填写 `loader` / `hub_id` 或 `artifacts`、`output.taxonomy`、`requires_remapping`
3. 在产品配置里把 `perception.model` 改成新名字

若模型直接输出 mower 类别，设 `requires_remapping: false` 并删除 ADE20K mapping。

## 打开网页评估（WSL）

在 Ubuntu-22.04 终端中只需：

```bash
cd /home/watermango/github/mower-perception
./scripts/dev.sh
```

打开 http://127.0.0.1:43129。网页分为两个标签：“数据集横向对比”先展示完整 256 张评测的总体指标，再按庭院逐图查看原图、标注和两模型的历史预测；“单图试验”用于三张样例、自己上传的图片或从评测集带入的图片。点击“在单图试验中打开”会切换标签并重新推理，之后切换模型保持同一张图。ADE20K 明确标为训练数据，GrassSegHB 标为评测数据；无人工标注的图片不显示准确率。

模型自动优先使用本地缓存，缺少文件时才尝试下载，不需要手动设置 HF_HUB_OFFLINE。首次环境安装和 CNN 权重下载见上文及评测文档。数据尚未下载时网页会说明；不自动下载近 1 GB 数据。Ctrl+C 停止服务。

### 开发者验证（日常使用无需执行）

`./scripts/test.sh` 汇总 Python 测试、lint、类型检查、构建；`python -m tools.verify_api` 在服务运行时验证样例及上传接口。这些是维护代码的工具，不是使用网页前必须执行的步骤。

需要重新生成全部标注评测时，参见 [评测流程](docs/evaluation.md)。已有完整结果在 `outputs/grassseghb/`，网页直接读取它们，不会在启动时重复跑 256 张推理。

## 换自己的数据

1. 按 yaml 里的 `id` 准备 `images/` 与 `masks/`
2. 微调轻量分割网后新增 model yaml（`output.taxonomy: mower`）
3. 改 `perception.model`，CLI / FastAPI / 前端契约保持不变

## 公平对比与割草场景评测

参见 [评测流程、权重和数据许可](docs/evaluation.md) 与 [实测报告](evaluation/REPORT.md)。首次选择 MIT CNN 前运行 `python -m tools.download_mit_weights`。ADE20K grass 仅是可割代理；VOC 无草地类别，不参与可割准确率比较。

## 已训练 YCOR 模型的网页入口

在“单图试验”中选择 **YCOR 可通行草地代理模型**，可对样例或上传图片推理。
使用 ImageNet 骨干经 YCOR 二分类训练的权重；绿色为原始标签1（可通行草地代理），灰色为0（其他有效类别）。
它不是安全可割验证，不输出人/水体等细分类，也不参与 GrassSegHB 可割准确率计算。
可下载原始0/1 PNG；整图输入/输出512×384，图例和占比仅有两类。
网页“模型前向”耗时不等于离线报告的完整流水线延迟。

默认加载 `weights/ycor/best.pt`（Git忽略），可通过 `YCOR_CHECKPOINT` 覆盖。
来源、SHA-256及恢复方式见 [训练报告](evaluation/ycor/REPORT.md)。缺失或无效权重会显示明确错误，其他模型仍可用。
本机已配置 `.env.runtime.sh`，从仓库根目录执行：

```bash
source .env.runtime.sh
bash scripts/dev.sh
```

其他机器按上文安装环境并放入权重后运行已有 `scripts/dev.sh`。
