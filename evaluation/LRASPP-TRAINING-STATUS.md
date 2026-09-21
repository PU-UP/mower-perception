# LR-ASPP 草坪训练：未完成（许可阻塞）

2026-09-21。**本轮没有完成真实训练，未交付可割二分类 checkpoint，也没有 LR-ASPP 与 B0 的新效果/速度比较。** GPU 兼容性检查不是训练、过拟合检查或推理基准。这个 PR 仅保留许可核查、环境准备和漏割率指标补充，不能作为训练目标验收通过的依据。

## 1. 实际执行了什么

从最新 `origin/main` 的 `fc293d450e025e4afdc658ed509064a1f37b4007` 建立独立分支 `codex/lraspp-training-audit`，worktree 为 `/home/watermango/github/mower-perception-lraspp`。原工作区检查时干净，未重置、未修改原工作区、未更新其 main。已阅读实际 AGENTS.md、`docs/evaluation.md`、`evaluation/REPORT.md` 和现有评测/下载实现。

本机现有 Python 环境直接支持 torchvision 官方 MobileNetV3-Large + LR-ASPP，未安装旧依赖，未使用云算力。实际下载并严格加载 **ImageNet-1K V1 分类骨干预训练**，`weights=None`、`weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1`、`num_classes=2`。整个 LR-ASPP 分割头随机初始化，没有草坪分割能力。没有加载 COCO/VOC 完整分割预训练，也没有将 VOC 类别伪装成 grass。

数据无关的 GPU 检查使用随机诊断张量 `[2,3,384,512]`，验证输出 `[2,2,384,512]`，以及骨干、低层分类器、高层分类器的有限非零梯度。仅对输出平方均值做反向传播，没有标注、监督损失、优化器、参数更新或 checkpoint。**这不是用户要求的少量真实样本过拟合检查。** 实测 3,218,308 参数、峰值分配 481,563,136 bytes；该显存不是完整训练预算，未含优化器状态和数据加载。一次冷前后向耗时仅记录为环境诊断，不能用于与 B0 的推理速度比较。

完整机器记录：[lraspp-environment.json](lraspp-environment.json)。设备 RTX 5060 Laptop GPU，8151 MiB；驱动 592.27；Python 3.10.12、Torch 2.14.0+cu130、torchvision 0.29.0+cu130、CUDA 13.0；4 CPU 线程，FP32，关闭 TF32。检查时磁盘可用约 780 GiB。种子 20260921 仅用于诊断，不是已经执行的训练配置。

## 2. 模型实现、预处理与许可

直接调用 [torchvision 官方 LR-ASPP](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)，未重写结构。安装包源码 commit 为 `fc73f5a8567734ffa4c988b24d5432f0d3005974`，源文件摘要见环境记录。结构使用膨胀 MobileNetV3，低层 stride 8 / 高层 stride 16，128 通道 LR-ASPP 中间层，融合低/高层两路分类器。官方 forward 最后双线性插值到输入大小，`align_corners=False`，返回 raw logits。

[ImageNet V1 分类权重规范](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)是 RGB、缩放至短边 256 后中心裁剪 224、除以 255、mean `[.485,.456,.406]` / std `[.229,.224,.225]`。这是分类评测配方，不能对本任务整图分割直接套中心裁剪。完整 COCO/VOC 分割权重的官方短边 520 配方也不是本次已执行的草坪训练。后续领域训练应明确制定整幅图像与 mask 一致的几何变换，mask 最近邻；实际训练输入、增强和阈值目前均未确定。诊断直接输入张量，没有执行图像预处理。

[torchvision 代码许可](https://github.com/pytorch/vision/blob/main/LICENSE)为 BSD-3-Clause；[官方预训练模型说明](https://github.com/pytorch/vision#pre-trained-model-license)明确预训练权重可能受训练数据条款约束，不能把代码 BSD 当成所有模型/数据的商用授权。本轮未确认独立的权重商用授权。B0 延用历史报告中的 NVIDIA 非商业研究/评估限制。未启动 PP-LiteSeg-T：没有主候选兼容性或真实训练失败的证据；ResNet18 + PPM 保留历史结果，不再优化。

## 3. 数据许可与独立性

[GrassSegHB 官方页](https://www.informatik.uni-bremen.de/agebv2/pub/grassseghb/)描述了 0=不可割、1=可割并要求引用论文，没有找到图像、标注及本次训练用途的明确授权。实际再次读取 ZIP 中央目录：14,984 项（7,491 对加目录），除图像/标注及目录外没有任何成员，ZIP comment 为空。不存在归档内 README/LICENSE 可据以授权。[原论文](https://www.informatik.uni-bremen.de/agebv2/pub/grassseghb/thesis_ogruhlke2018.pdf)的文本检索 `Lizenz`、`Nutzung`、`copyright` 无命中；这不是穷尽法律分析，也不能证明授权永不存在。

检查实际仓库、数据目录及缓存未发现另存的 GrassSegHB 授权材料。用户说明若 WSL 仓库没有则没有；本轮没有自行联系作者。**阻塞范围包括本次图像/标注训练使用，不仅是商用或再分发。** 公开下载、要求引用、前次评测记录都不构成本次授权依据。没有继续下载图像或进行新的数据训练/效果评测。

数据来源、网页和目录摘要、ETag、已探索组及清单 SHA-256 见 [lraspp-data-audit.json](lraspp-data-audit.json)。原始数据仍在原仓库被忽略的 `data/grassseghb/` 中，没有复制至本分支。已有 256 张清单的图像摘要没有完全重复；**只检查已有摘要，不是重新解码像素查重，也不是近重复检查。** 未核实跨编号同庭院身份；近重复和相邻帧风险仍待审查。全部 0..15 组已用于旧报告决策，不能声称是独立最终测试。由于许可阻塞，没有假造训练/验证/测试冻结清单或宣称庭院独立性已验证。若仅使用这些组，后续只能标为探索性留出评测。

已核查替代来源：

- [YCOR 官方页](https://theairlab.org/yamaha-offroad-dataset/)明确 CC BY 4.0，但标签是 traversable grass 等；可通行不等于可割。其公开划分采集 session 不交叉但地点交叉。未直接转换为可割，也未下载或训练。
- [forefield_grassland 作者数据页](https://zenodo.org/records/10371371)提供 dry/grass/mowed/unplanted 图块分类，摄像机约 4 米高，没有提供本任务所需的可割像素标签说明，未作为替代训练集。

这不是宣称全网没有合适数据，而是本轮核查尚未得到同时满足许可、标签语义和独立分组要求的数据。

## 4. 效果、速度和失败案例

| 本轮所需结果 | LR-ASPP | 同冻结测试集 B0 |
| --- | --- | --- |
| IoU、误割率、漏割率、预测可割错误比例 | 未测 | 未测 |
| 边界 precision / recall / F1 | 未测 | 未测 |
| 每庭院指标、混淆计数、真实失败案例 | 未测 | 未测 |
| 预处理/前向/后处理/完整流水线 P50/P95 | 未测 | 未测 |
| 速度输出分辨率及对应效果变化 | 未选定/未测 | 未选定/未测 |

历史 B0 初评仍见 [REPORT.md](REPORT.md)，其完整分辨率、含可视化的端到端速度不能填入本表。不能把历史准确率拼接到未来低分辨率速度上。未声称 LR-ASPP 超过 B0、实时或适合板端。没有新模型失败案例可分析，不能将旧 B0/ResNet 错误冒充新模型证据。

复用现有 `mowerseg.evaluate.metrics` 增加 `mowable_false_negative_rate = FN/(TP+FN)`，分母为零记 null，既有边界和标签校验不变。旧冻结报告不重写。未来正式比较需先冻结 checkpoint/阈值，再以相同测试清单与输出几何评测；logits 插值后分类，速度必须包含传输，排除可视化/HTTP/保存/指标，并在相同速度输出配置下补测效果。

## 5. 复现环境检查与 artifact

在 WSL 独立 worktree 中运行（每次使用新输出名，工具拒绝覆盖）：

```bash
cd /home/watermango/github/mower-perception-lraspp
/home/watermango/github/mower-perception/.venv/bin/python -m tools.verify_lraspp_environment \
  --output outputs/lraspp-environment-recheck.json
HF_HUB_OFFLINE=1 /home/watermango/github/mower-perception/.venv/bin/python -m pytest tests -q
```

合并后也可在任意检出的仓库根目录，使用兼容的上述版本环境执行同一模块。它只验证环境，不提供训练命令或草坪推理功能。

实际下载的骨干文件位置：`/home/watermango/.cache/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth`；SHA-256：`8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04`。可从环境 JSON 内记录的官方 URL 或上述命令重新获取；这是 ImageNet 骨干，**不是训练后的二分类 artifact**。没有训练 checkpoint、最佳/最终配置、学习曲线或训练日志可提供或在 PR 合并后下载。大权重、原图和含原图产物没有提交 Git。

未注册未训练模型，未改模型选择界面。未来原始二分类 0 必须保留为不可割，1 为可割；产品映射不能把 0 当 ignore 或安全可通行，不能产生人/动物/水体等细分类。本轮不存在可验证的已训练模型加载或新图推理。

## 6. 验证与下一步

Python 38 项测试通过，包括新增漏割率 .5、全漏割 1、无真可割 null 断言；原有标签、几何、摘要和完整覆盖检查保留。GPU 兼容性检查通过。前端 ESLint、TypeScript（先执行 Next typegen）、默认 Turbopack 生产构建全部通过。新 worktree 首次缺少生成的 LayoutProps 类型，typegen 后解决；跨 worktree 的 node_modules 软链接被 Turbopack 拒绝，改为已有依赖本地副本后默认构建通过。没有改前端源码、依赖版本或构建配置。`git diff --check` 通过，原工作区仍干净。没有已训练模型，因此无法进行该模型的真实图片推理验收，未把随机张量检查算作该项通过。

解除阻塞需要权利方明确覆盖图像、标注与本次训练用途的授权依据，或具备相应权利的自有真实庭院数据及可割二分类标注。随后必须核实庭院/采集序列、查重及近重复，冻结划分；再进行标签可视化、真实小样本过拟合、预算内正式训练、验证选择与冻结、同集效果及纯推理速度比较。当前尚无依据讨论优化器、学习率、epoch 或速度阈值的最优选择，也无依据保证训练效果。

结论：**下一步应先解决数据权利和独立性，再实际训练；暂不值得开展导出、量化或板端性能承诺。** 这个草稿 PR 保持未合并，待项目审核 agent 复审；不创建自动审核任务。不能将准备工作算成本轮目标完成。
