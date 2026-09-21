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

## 测试与网页启动（WSL）

在 Ubuntu-22.04 中执行，Windows 协调目录不是代码仓库：

```bash
cd /home/watermango/github/mower-perception
source .venv/bin/activate
# 若当前终端没有 node/npm，先加载已安装的 Node（例如 nvm use）
./scripts/test.sh
```

`test.sh` 依次运行 Python 测试、前端 lint、TypeScript 检查和生产构建；失败即停止。使用已有 `.venv` 和 `node_modules`，不自动安装或降级依赖。首次准备环境参见上面的安装命令（包含 `pip install pytest` 和 `npm ci`）。它不会启动网页，也不会自动执行完整数据评测。

启动交互网页，保持这个终端运行：

```bash
# CNN 权重已有缓存时会复用；首次需联网下载
python -m tools.download_mit_weights
./scripts/dev.sh
```

浏览器打开 http://127.0.0.1:43129，切换模型、选择样例或上传图片；Ctrl+C 停止。已有全部 HF 权重缓存时可使用 `HF_HUB_OFFLINE=1 ./scripts/dev.sh`，减少弱网元数据查询；首次下载不要开启离线模式。

保持网页服务运行，在第二个 WSL 终端执行真实样例和上传接口测试：

```bash
cd /home/watermango/github/mower-perception
.venv/bin/python -m tools.verify_api
```

该测试对每个模型执行三张样例和一次上传，结果写入 `outputs/api-validation.json`。API 测试不等于标注准确率评测。

重跑 GrassSegHB 对比（无需网页服务，建议先停止服务释放显存）：

```bash
source .venv/bin/activate
python -m tools.prepare_grassseghb --download --workers 8
python -m tools.audit_grassseghb
# 先用 2 张真实标注图快速验证流程
python -m mowerseg.evaluate --limit 2 --output outputs/smoke
# 完整 256 张，耗时较长
python -m mowerseg.evaluate --warmup 5 --threads 4
```

下载器复用通过校验的已有数据，不下载整个归档。完整结果在 `outputs/grassseghb/results.json`，同图对比与失败案例入口为 `outputs/grassseghb/index.html`。采样、计时口径和许可限制见 [评测文档](docs/evaluation.md)。

## 换自己的数据

1. 按 yaml 里的 `id` 准备 `images/` 与 `masks/`
2. 微调轻量分割网后新增 model yaml（`output.taxonomy: mower`）
3. 改 `perception.model`，CLI / FastAPI / 前端契约保持不变

## 公平对比与割草场景评测

参见 [评测流程、权重和数据许可](docs/evaluation.md) 与 [实测报告](evaluation/REPORT.md)。首次选择 MIT CNN 前运行 `python -m tools.download_mit_weights`。ADE20K grass 仅是可割代理；VOC 无草地类别，不参与可割准确率比较。
