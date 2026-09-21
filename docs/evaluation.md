# ADE20K CNN 与 SegFormer-B0 的割草场景初评

后续 LR-ASPP 真实训练请求目前因图像/标注训练授权未确认而阻塞，见 [训练状态与核查证据](../evaluation/LRASPP-TRAINING-STATUS.md)。旧报告的“不能声称商用”不代表已确认研究训练授权。

本流程无需训练，比较的是现成 ADE20K 模型的 **grass 可割代理预测**。它不是安全可通行判定，也不验证人、动物或小障碍物的完整检测能力。不要用像素占比、置信度或演示图片替代标注评测。

## 模型与来源

| 模型 | 权重/类别来源 | 实际预处理 |
| --- | --- | --- |
| SegFormer-B0 | [NVIDIA ADE20K checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)，revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da` | checkpoint 自带 ImageProcessor；RGB / 255，ImageNet mean/std；整图双线性缩放到 512×512，无裁切 |
| ResNet18-dilated + PPM_deepsup | [MIT CSAIL 官方项目](https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/8f27c9b97d2ca7c6e05333d5766d144bf7d8c31b)，完整 ADE20K encoder **和** decoder，150 类 | RGB / 255，同样 mean/std；短边 600、长边至多 1000，再把两边分别向上取整到 8 的倍数；整图双线性缩放，无裁切；单尺度 |
| 旧 DeepLabV3+ MobileNetV2 | Google PASCAL VOC 权重 | 保留原有全幅长边 513 规则与中心裁剪回归测试；VOC 没有 grass，排除在本次可割能力比较之外 |

mean = `[0.485, 0.456, 0.406]`，std = `[0.229, 0.224, 0.225]`。两种 ADE20K 模型接收相同原始图片和相同目标类别，但按各自官方规则处理尺寸。这是现成模型/推理配置的比较，不是控制训练数据量、训练配方、参数量或算力后的纯架构消融。不得由此泛化到全部轻量 CNN。

### 为什么使用备用 CNN

2026-09-21 核实：官方 README 报告 MobileNetV2-dilated + PPM_deepsup 单尺度 ADE20K mIoU 35.76；这是**来源资料，不是本项目实测**。上述固定 commit 中没有该组合的配置（只有 MobileNetV2 + C1_deepsup）。按官方 demo 的目录命名规则请求 `ade20k-mobilenetv2dilated-ppm_deepsup/{encoder,decoder}_epoch_20.pth`，两者均为 HTTP 404；目录列表为 403，HTTPS 的证书域名不匹配。不能据此声称世界上不存在该权重，但本次不能验证并获取这套完整 artifact。

同项目 `config/ade20k-resnet18dilated-ppm_deepsup.yaml` 和对应两个官方 HTTP 权重均可用（200）：encoder 45,415,518 bytes，decoder 53,019,215 bytes。因此使用用户指定的 ResNet18 备用组合。没有随机分割头、VOC 冒充或 ImageNet-only 替代。ResNet18 这里采用 MIT 的三层 3×3 stem，不是直接替换成 torchvision 的标准 7×7 stem。

`mowerseg/models/mit_resnet.py` 仅包含所需推理结构，保留官方层名和完整辅助头参数。普通 BatchNorm 的 eval 行为与原 SyncBN 相同；只移除 SyncBN 三个训练缓冲区，其余参数严格加载，缺失或错误 decoder 会失败。输出 raw logits 在原图尺寸上双线性插值（`align_corners=False`）后 argmax；单尺度省略最后 softmax 不改变 argmax。按 8 个类别分块还原以控制 4000×3000 图像的内存；测试验证其与一次性插值完全一致。

MIT 代码以 BSD-3-Clause 发布，保留了 `mowerseg/models/MIT_CSAIL_LICENSE.txt`。代码许可证不自动解决训练数据、图像内容或权重的所有使用权问题；官方权重目录未附独立权重许可。B0 模型卡标注 `License: other`，并指向 [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)，其 3.3 条限制为非商业研究/评估使用；不得标成 MIT 或承诺可商用。数据权利需另行核实。

### 下载与运行

在 WSL 实际仓库根目录执行，沿用现有 `.venv`，不要安装 MIT 旧 requirements：

```bash
source .venv/bin/activate
python -m tools.download_mit_weights
python -m mowerseg --help
```

通过现有网页按钮选择 `ResNet18-dilated + PPM (ADE20K)`，或者：

```python
from mowerseg import PerceptionEngine
with PerceptionEngine(model="mit_resnet18_ade20k", config_path="configs/mower_seg.yaml") as engine:
    result = engine.predict("assets/samples/lawn_path.jpg")
    print(result.metadata["input_shape"], result.latency_ms)
```

权重默认位于被忽略的 `.cache/mit-resnet18/`；可用 `MOWER_MIT_WEIGHTS` 指定目录。完整权重的 SHA-256 固定在下载器/加载器中。官方服务仅 HTTP 可用，摘要可用于复现与后续完整性检查，不是上游签名。模型前向时间在 UI 单独标示，实际输入显示为高×宽。首次加载不算稳定延迟。HF 权重缓存完整后可设置 `HF_HUB_OFFLINE=1` 启动服务或评测，避免弱网络下元数据查询让首次切换超过网页代理等待时间；尚未缓存时不要设置该变量。

## 标签语义

仓库附带上游 `data/object150_info.csv` 的原始类别表。ADE20K 标注 id 0 为未标注，1..150 是有效类别；训练减 1 后为 -1（ignore）和 0..149。HF 训练标注转换则用 255 表示 ignore；测试显式验证 `[0,1,10,150] → [255,0,9,149]`。当前 Transformers 对旧 checkpoint 的 `reduce_labels: true` 不会自动启用 `do_reduce_labels`；若未来传入 ADE20K segmentation_maps，必须显式设置 `do_reduce_labels=True`。本次仅传图片，不触发标签转换。模型输出已经是 0..149，**不得再次减 1**。grass 的官方表 id=10，对应输出 index=9；输出 0 是 wall，不是 ignore。

产品映射只把输出 9 映射到产品 grass=1。tree/plant 等 vegetation 和 terrain 不并入可割。未列出的模型类别保守映射 obstacle；255 并不是模型的有效类别，不能作为可割。评测直接取 `raw_mask == 9`，并断言它与产品 grass 映射一致。

GrassSegHB 的标注是二分类：0=不可割、1=可割，**0 不可忽略**。本评测拒绝 255 或其他未声明值、彩色标注和不匹配尺寸，不猜测标签编码，不自动拉伸标注。

## 数据与许可

优先来源：[GrassSegHB 官方页](https://www.informatik.uni-bremen.de/agebv2/pub/grassseghb/)。页面介绍约 7,500 张、秋季、公园和庭院、模拟割草机视角，包含不同光照/阴影/落叶。引用：Oliver Gruhlke, *Deep Learning zur Grassegmentierung im Arbeitsraum eines Rasenmähroboters*, Universität Bremen, 2018。

2026-09-21 实际 ZIP：29,431,557,236 bytes，解压文件总计 29,573,927,368 bytes。实际 7,491 对，`images/<garden>G<frame>.jpg` 与 `annotations/<garden>G<frame>_mask.png`；16 个 garden id（0..15）。网页提到 2 公园+16 私人庭院，但归档编号只有 16 组，不把网页地点数当作实际分组数。目录中未发现 README、LICENSE 或文本许可；官网仅要求引用。**许可未明确，不能声称允许商用。** 原始图像、标注和含原图对比图均留在忽略目录，不随 PR 再分发。

补充来源（此次不下载）：

* [YCOR](https://theairlab.org/yamaha-offroad-dataset/)：1,076 张，官网 CC BY 4.0；其 traversable grass 仍不能自动等价于 mowable。
* [RELLIS-3D](https://www.unmannedlab.org/research/RELLIS-3D)：6,235 张标注图像，官网 CC BY-NC-SA 3.0，有非商业限制。不下载 LiDAR。

### 可复现子集

```bash
# 先检查下载量/空间并保存计划清单，不下载整包
python -m tools.prepare_grassseghb
# 实际只取选中的图像与标签；CRC 校验、SHA-256 清单、可重复执行复用完整文件
python -m tools.prepare_grassseghb --download --workers 8
```

默认种子 `20260921`，16 组各取 16 张，共 256 张。每组按数值 frame id 排序，均分 16 个时间段，每段固定种子随机取一张，降低连续帧重复，不宣称相互独立。输入 ZIP 顺序不影响抽样；清单保留分组、路径、种子及每张文件摘要。下载前检查空间，分段读取 ZIP，拒绝服务器忽略 Range 后整包下载；数据在 `/data/` 下被 Git 忽略。

这是探索性基线子集，不是训练/验证/测试混合切分。未来必须先冻结整庭院（或确认独立采集序列）的划分，使任一庭院只属于一个 split。禁止随机逐帧拆分。由于本次已查看全部初评组的表现，后续若据此调参，须新增未参与决策的独立庭院作最终测试；不能继续把这 256 张宣称为未见测试集。亮度不等于精确光照标签，自动亮度统计只能辅助覆盖检查，需人工查看阴影/逆光/落叶/边界样例。

## 批量评测

```bash
python -m tools.audit_grassseghb
python -m mowerseg.evaluate --warmup 5 --threads 4
# 先做少量真实数据冒烟；必须指定独立输出，禁止覆盖默认完整报告
python -m mowerseg.evaluate --limit 2 --output outputs/smoke
# 可反转顺序重测，检查散热/负载影响
python -m mowerseg.evaluate --models mit_resnet18_ade20k segformer_b0_ade20k --output outputs/reverse
```

默认两种 ADE20K 模型、batch=1、FP32（同时记录 TF32 后端开关），使用清单中**相同原始图片**。每模型先在首张图预热 5 次；加载和下载不计入计时，保留每张实际输入尺寸和原图尺寸。CUDA 前向用 Events 且同步，端到端测量前后也同步。报告硬件、Torch/CUDA、线程数、均值、中位数、P95、原始逐图时间。模型时间排除输入/输出传输和输出还原；端到端包括文件解码、预处理、传输、前向、原尺寸还原和产品可视化，不包括校验摘要、计算指标、保存评测图或 HTTP/浏览器。不得用两者混淆宣称实时性能。

指标按原始标注尺寸累计所有图像像素（micro），另给每庭院结果及逐图明细：

* 可割 IoU = TP / (TP + FP + FN)。
* 不可割误判可割率 = FP / (FP + TN)。
* 漏割率 = FN / (TP + FN)；后续运行记录为 `mowable_false_negative_rate`，不改写历史冻结结果。
* 预测可割中的错误比例 = FP / (TP + FP)。
* 边界 F1：提取二值 mask 的四邻域标签转变，边界两侧像素都保留；不把图像外框当边界。在原标注分辨率以 3 像素 Chebyshev（方形）容差匹配两侧边界，分别算 precision/recall 再 F1。
* 边界错误率：GT 边界的同样 3 像素扩张带内，预测错误像素 / 带内像素。不同分辨率比较时必须同时报告尺寸和半径。

分母为零输出 `null`，不偷偷填 1。`outputs/grassseghb/index.html` 并排展示同一失败样例在两个模型下的结果。每张图保存原图、标注、预测、错误图并排预览（红=FP、蓝=FN），以及原始分辨率 0/1 预测 PNG。`results.json` 含按不可割误判率排序的十个失败样例 id，便于重复核查。统计检验或置信区间需以庭院为抽样单位；这里不把百万相关像素当成独立样本宣称显著性。

## 验证入口

```bash
python -m pytest tests -q
# 先启动后端和生产前端，再执行真实接口验证
python -m tools.verify_api
npm run lint
npx tsc --noEmit
npm run build

# 可选：只克隆源码用于对照，不安装其依赖
git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git .cache/sources/mit
git -C .cache/sources/mit checkout 8f27c9b97d2ca7c6e05333d5766d144bf7d8c31b
python -m tools.verify_mit_runtime --source .cache/sources/mit
```

具体实测、人工标注检查和是否值得小规模微调的结论见 `evaluation/REPORT.md`。公开榜单不替代此报告。此次不训练、不接自动驾驶控制、不创建自动审查任务。

网页只接受每个模型逐图清单与冻结 manifest 完全一致的完整报告（包含分组、标注和文件摘要）；部分试跑不会作为总体成绩展示。庭院数、输入尺寸及边界容差来自实际报告。单图实时推理的边界容差另行显示，当前为 3 像素。
