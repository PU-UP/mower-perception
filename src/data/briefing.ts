export type TaskId =
  | "backbone"
  | "segmentation"
  | "depth"
  | "geometry"
  | "stereo"
  | "flow";

export const TASKS: { id: TaskId | "all"; label: string; hint: string }[] = [
  { id: "all", label: "全部", hint: "2025 下半年到 2026 年 9 月" },
  { id: "backbone", label: "稠密骨干", hint: "DINOv3 / RADIO / MLLM 编码器" },
  { id: "segmentation", label: "分割", hint: "概念分割与开集" },
  { id: "depth", label: "深度估计", hint: "度量、鲁棒、无内参" },
  { id: "geometry", label: "前馈几何", hint: "多视图空间恢复" },
  { id: "stereo", label: "立体匹配", hint: "零样本与边缘部署" },
  { id: "flow", label: "光流", hint: "连续动力学与 Flow Matching" },
];

export type Paper = {
  id: string;
  title: string;
  venue: string;
  date: string;
  tasks: TaskId[];
  oneLiner: string;
  whyItMatters: string;
  numbers: string[];
  links: { label: string; href: string }[];
};

export const TRENDS = [
  {
    index: "01",
    title: "稠密骨干换代，线性探测就能打专用模型",
    body: "DINOv3 把自监督特征做到冻结骨干 + 线性头即可在 ADE20K / Cityscapes / NYUv2 上超过 RADIO、Perception Encoder 等聚合骨干。2026 年再训一套 U-Net 骨干，已经不是默认选项。",
  },
  {
    index: "02",
    title: "分割从「点一下」变成「说出概念」",
    body: "SAM 3 把任务正式定义为 Promptable Concept Segmentation：用短名词短语或示例图找出并跟踪所有实例。SAM 3.1 再把多目标视频跟踪提速约 7 倍。闭集语义分割还在，但研究主场已经是开集。",
  },
  {
    index: "03",
    title: "深度估计被并入「视觉空间恢复」",
    body: "VGGT、π³、Depth Anything 3 把相机位姿、深度、点图、甚至 3DGS 放进同一个前馈 Transformer。单目深度仍在（MoGe-3、MD2E），但 2026 年的新问题是：任意视角、有无位姿、度量尺度，一次出几何。",
  },
  {
    index: "04",
    title: "分割与深度开始互为先验",
    body: "RoSAMDepth 用 SAM 掩码稳住恶劣天气下的自监督深度；PDFNet 用 Depth Anything v2 的「深度完整性先验」做高精度抠图；GeomPrompt 在深度缺失时不为重建深度而学几何提示。任务边界在变薄。",
  },
  {
    index: "05",
    title: "立体匹配和光流没有过时，只是接上了基础模型",
    body: "PromptStereo、FoundationStereo 把单目深度基础模型当结构先验；Pip-Stereo 把迭代立体匹配压到 Jetson 实时；光流则从离散对应转向 Flow Matching 和连续输运场。经典对应问题仍在出新方法。",
  },
];

export const TIMELINE = [
  {
    date: "2025.06",
    title: "VGGT",
    detail: "CVPR 2025。前馈一次给出相机、深度、点图和轨迹，成为后续几何模型的对照基线。",
  },
  {
    date: "2025.07",
    title: "MoGe-2 / π³",
    detail: "单目度量几何更锐、更准；π³ 去掉固定参考视图，用置换等变结构做无序多视图几何。",
  },
  {
    date: "2025.08",
    title: "DINOv3",
    detail: "自监督稠密特征大幅抬升。冻结线性探测：ADE20K 55.9 mIoU，NYUv2 RMSE 0.309。",
  },
  {
    date: "2025.10",
    title: "COSINE",
    detail: "ICCV 2025。把开词汇分割和 in-context 分割收进同一套多模态提示解码器。",
  },
  {
    date: "2025.11",
    title: "SAM 3 / SAM 3D / DA3",
    detail: "概念分割、单图 3D 物体重建、任意视角视觉空间恢复几乎同时落地，定义了 2026 年的默认工具箱。",
  },
  {
    date: "2026.01",
    title: "C-RADIOv4",
    detail: "教师集升级为 SigLIP2 + DINOv3 + SAM 3，聚合骨干继续跟老师迭代。",
  },
  {
    date: "2026.03",
    title: "SAM 3.1",
    detail: "Object Multiplex：128 个目标约 7 倍加速，视频分割在 MOSEv2 上再涨 2.0。",
  },
  {
    date: "2026.06",
    title: "CVPR 2026 丹佛",
    detail: "16,092 篇投稿，4,089 篇接收。生成与多模态最热，但分割、深度、立体、光流仍有一批可直接用的系统。",
  },
  {
    date: "2026.07",
    title: "MoGe-3",
    detail: "把几何精修从 2D 解码抬到稀疏体素壳，专门救细杆、薄结构和小物体。",
  },
  {
    date: "2026.09",
    title: "ECCV 2026",
    detail: "VersaViT 明确指出：MLLM 视觉编码器做分割/深度仍然偏弱，需要面向稠密任务的后训练。",
  },
];

export const PAPERS: Paper[] = [
  {
    id: "dinov3",
    title: "DINOv3",
    venue: "arXiv 2025.08 · Meta",
    date: "2025.08",
    tasks: ["backbone", "segmentation", "depth"],
    oneLiner: "自监督稠密特征第一次在分割和深度的线性探测上全面压过聚合骨干。",
    whyItMatters:
      "对做传统 dense prediction 的人，这意味着 2026 年的默认 backbone 不再是 ImageNet 分类预训练，甚至不必是 CLIP。Gram 正则让后期训练不再毁掉早期学到的稠密结构。ViT-7B 冻结线性探测就能到 ADE20K 55.9、Cityscapes 81.1、VOC 86.6。",
    numbers: [
      "ADE20K 55.9 / Cityscapes 81.1 / VOC 86.6 mIoU（冻结线性探测）",
      "NYUv2 RMSE 0.309，KITTI RMSE 2.346",
      "同设定下超过 DINOv2、AM-RADIOv2.5、PEspatial",
    ],
    links: [
      { label: "论文", href: "https://arxiv.org/abs/2508.10104" },
      { label: "代码", href: "https://github.com/facebookresearch/dinov3" },
    ],
  },
  {
    id: "c-radio-v4",
    title: "C-RADIOv4",
    venue: "Tech Report 2026.01 · NVIDIA",
    date: "2026.01",
    tasks: ["backbone"],
    oneLiner: "把教师从 CLIP + DINOv2 + SAM 换成 SigLIP2 + DINOv3 + SAM 3。",
    whyItMatters:
      "聚合骨干这条线没有停。谁当老师，学生就继承谁的稠密能力。C-RADIOv4 明确写了：SAM 升级到 SAM 3 之后，分割教师本身已经接近「能用概念做视觉」。",
    numbers: [
      "教师集：SigLIP2、DINOv3、SAM 3",
      "继承 DINOv3 的语义分割能力与 SigLIP2 的图文对齐",
    ],
    links: [{ label: "技术报告", href: "https://arxiv.org/html/2601.17237" }],
  },
  {
    id: "versavit",
    title: "VersaViT",
    venue: "ECCV 2026",
    date: "2026.09",
    tasks: ["backbone", "segmentation", "depth"],
    oneLiner: "MLLM 里的视觉编码器并不天然擅长分割和深度，需要面向稠密任务的后训练。",
    whyItMatters:
      "这是对 2026 年一个流行误解的纠正：把 Qwen-VL / 同类模型的 vision tower 当通用 CV 骨干，在像素级任务上往往不如 DINOv3。VersaViT 用轻量任务头和多粒度监督回灌编码器，让同一套 ViT 既能对接语言，又能做 dense prediction。",
    numbers: ["指出 MLLM 编码器在语义分割、深度估计上的稠密表征不足"],
    links: [
      {
        label: "ECCV 章节",
        href: "https://link.springer.com/chapter/10.1007/978-3-032-37577-3_5",
      },
    ],
  },
  {
    id: "sam3",
    title: "SAM 3: Segment Anything with Concepts",
    venue: "Meta · 2025.11 发布",
    date: "2025.11",
    tasks: ["segmentation"],
    oneLiner: "用文本概念或示例图，检测、分割并跟踪图中/视频中所有匹配实例。",
    whyItMatters:
      "SAM 1/2 是交互式视觉提示；SAM 3 把开词汇检测和视频跟踪收进同一骨干，并提出 SA-Co 基准。对传统分割产线，这相当于「标注器 + 开集检测器 + 跟踪器」被一个模型替换。LVIS 零样本 mask AP 从 38.5 提到 48.8。",
    numbers: [
      "SA-Co/Gold cgF1 54.1，约为 OWLv2（24.6）的两倍",
      "LVIS 零样本 mask AP 48.8 vs 此前 38.5",
      "数据引擎产出约 400 万独特概念标签",
    ],
    links: [
      { label: "论文", href: "https://arxiv.org/abs/2511.16719" },
      {
        label: "Meta 介绍",
        href: "https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/",
      },
    ],
  },
  {
    id: "sam3-1",
    title: "SAM 3.1",
    venue: "Meta · 2026.03.27",
    date: "2026.03",
    tasks: ["segmentation"],
    oneLiner: "Object Multiplex：多目标视频跟踪不再随目标数线性爆炸。",
    whyItMatters:
      "SAM 3 的视频管线对每个目标独立跑，128 个目标时开销过大。3.1 把目标打进固定容量的 bucket 联合推理，H100 上约 7 倍加速，同时 MOSEv2 等 VOS 指标继续涨。这是从「能用」到「能上线」的一步。",
    numbers: [
      "128 目标约 7× 加速（相对 2025.11 版 SAM 3）",
      "MOSEv2 val J&F：60.3 → 62.3",
      "MOSEv1 val：78.4 → 79.6",
    ],
    links: [
      {
        label: "发布说明",
        href: "https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md",
      },
      {
        label: "博客",
        href: "https://ai.meta.com/blog/segment-anything-model-3/",
      },
    ],
  },
  {
    id: "cosine",
    title: "COSINE",
    venue: "ICCV 2025",
    date: "2025.10",
    tasks: ["segmentation"],
    oneLiner: "一套解码器同时做开词汇分割和 in-context 分割。",
    whyItMatters:
      "2024–2025 的开集分割分裂成两条互不兼容的管线。COSINE 用冻结 DINOv2/CLIP 特征池 + SegDecoder，证明视觉提示和文本提示可以协同，而不是互相替代。",
    numbers: ["开词汇与 few-shot / in-context 设定上相对此前专用模型全面提升"],
    links: [
      {
        label: "论文",
        href: "https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Unified_Open-World_Segmentation_with_Multi-Modal_Prompts_ICCV_2025_paper.html",
      },
      { label: "代码", href: "https://github.com/aim-uofa/COSINE" },
    ],
  },
  {
    id: "dinov3-seg",
    title: "dinov3.seg",
    venue: "arXiv 2026.03",
    date: "2026.03",
    tasks: ["segmentation", "backbone"],
    oneLiner: "在 DINOv3 的图文接口上专门做开词汇语义分割，而不是零样本硬套。",
    whyItMatters:
      "有了强骨干不等于分割完成。dinov3.seg 补了任务头、全局/局部文本视图、以及 SAM 引导的后期细化，说明 2026 年「开词汇分割」仍值得做架构，而不只是换 backbone。",
    numbers: ["明确建立在 dinov3.txt 上的 OVSS 框架，含双阶段细化与高低分辨率推理"],
    links: [{ label: "论文", href: "https://arxiv.org/html/2603.19531" }],
  },
  {
    id: "pdfnet",
    title: "PDFNet",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["segmentation", "depth"],
    oneLiner: "用单目伪深度的「完整性先验」做高精度二分分割，不必上扩散模型。",
    whyItMatters:
      "DIS 长期在「快但不准」和「扩散很准但太贵」之间摇摆。PDFNet 观察到：完整物体在深度图里是低方差、边界锐利的连通区域。借助 Depth Anything v2，非扩散方法在 DIS-VD 上达到 Fmax 0.915，参数量不到扩散方法的一半。",
    numbers: ["DIS-VD / DIS-TE Fmax β 0.915", "参数量不到扩散类方法的一半"],
    links: [
      {
        label: "论文 PDF",
        href: "https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_High-Precision_Dichotomous_Image_Segmentation_via_Depth_Integrity-Prior_and_Fine-Grained_Patch_Strategy_CVPR_2026_paper.pdf",
      },
      { label: "项目页", href: "https://tennine2077.github.io/PDFNet.github.io/" },
    ],
  },
  {
    id: "geomprompt",
    title: "GeomPrompt",
    venue: "CVPR 2026 Workshop",
    date: "2026.06",
    tasks: ["segmentation", "depth"],
    oneLiner: "深度缺失或损坏时，不要先估度量深度，直接学给冻结 RGB-D 分割器的几何提示。",
    whyItMatters:
      "机器人现场的深度经常是空的、花的。GeomPrompt 说明：下游分割监督就够学出第四通道，不必为了补深度再挂一个单目深度模型。这是「任务驱动的跨模态补偿」，很工程。",
    numbers: ["无深度监督即可驱动冻结 RGB-D 分割器", "在模拟传感器失效下优于直接 RGB-only"],
    links: [
      {
        label: "论文 PDF",
        href: "https://openaccess.thecvf.com/content/CVPR2026W/URVIS/papers/Jaganathan_GeomPrompt_Geometric_Prompt_Learning_for_RGB-D_Semantic_Segmentation_Under_Missing_CVPRW_2026_paper.pdf",
      },
    ],
  },
  {
    id: "vggt",
    title: "VGGT",
    venue: "CVPR 2025",
    date: "2025.06",
    tasks: ["geometry", "depth"],
    oneLiner: "一个前馈网络同时估计相机、深度、点图和 3D 轨迹。",
    whyItMatters:
      "它把「深度估计」从单任务模型，推成大规模 3D 标注上的通用几何 Transformer。2026 年几乎所有视觉几何论文都以它为对照。子秒级、可到上百张图，且常优于需要后处理优化的方法。",
    numbers: ["一张到上百张图一次推理，通常不到 1 秒", "位姿、多视图深度、点云、点跟踪多项 SOTA（当时）"],
    links: [
      {
        label: "论文",
        href: "https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html",
      },
      { label: "代码", href: "https://github.com/facebookresearch/vggt" },
    ],
  },
  {
    id: "pi3",
    title: "π³（Pi3）",
    venue: "ICLR 2026",
    date: "2025.07 / 2026",
    tasks: ["geometry", "depth"],
    oneLiner: "置换等变的视觉几何：不再指定参考视图。",
    whyItMatters:
      "VGGT 一类方法常把第一帧钉成世界坐标，参考帧选差就会漂。π³ 直接预测仿射不变位姿和尺度不变局部点图，输入顺序不再是隐患。这对无序相册、随意视频帧更稳。",
    numbers: ["无参考帧、对输入排列等变", "位姿、单目/视频深度、稠密点图多项领先"],
    links: [
      { label: "论文", href: "https://arxiv.org/abs/2507.13347" },
      { label: "代码", href: "https://github.com/yyfz/pi3" },
    ],
  },
  {
    id: "da3",
    title: "Depth Anything 3",
    venue: "ICLR 2026 Oral",
    date: "2025.11 / 2026",
    tasks: ["geometry", "depth"],
    oneLiner: "任意视角恢复视觉空间：普通 Transformer + 单一 depth-ray 目标就够。",
    whyItMatters:
      "DA3 的研究判断很「传统 CV」：不要复杂多任务头，不要特制几何结构。一个 vanilla DINO 编码器和 depth-ray 表示，就能同时做位姿、任意视图几何和渲染。相对 VGGT，位姿精度平均高 35.7%，几何精度高 23.6%；单目也超过 DA2。DA3-Long 在大场景 SLAM 上甚至优于跑 48 小时的 COLMAP。",
    numbers: [
      "相对 VGGT：位姿 +35.7%，几何 +23.6%",
      "单目深度超过 Depth Anything 2",
      "DA3-Large（0.30B）可在多项指标上超过更大的 VGGT（0.9B）",
    ],
    links: [
      { label: "项目页", href: "https://depth-anything-3.github.io/" },
      { label: "技术报告", href: "https://depth-anything-3.github.io/assets/da3_tech_report_2025.pdf" },
    ],
  },
  {
    id: "moge2",
    title: "MoGe-2",
    venue: "arXiv 2025.07 · Microsoft",
    date: "2025.07",
    tasks: ["depth", "geometry"],
    oneLiner: "单目同时要度量尺度和锐利细节，而不是二选一。",
    whyItMatters:
      "Depth Anything v2 偏合成数据、看起来锐但几何不一定准；Depth Pro 锐但几何仍有缺口。MoGe-2 坚持用真实深度（先过滤边界噪声再修复）换几何精度，同时保留度量尺度。",
    numbers: ["输出度量点图 / 深度 / 可选法向和相机", "ViT-L 约 326M 参数"],
    links: [
      { label: "论文", href: "https://arxiv.org/abs/2507.02546" },
      { label: "代码", href: "https://github.com/microsoft/MoGe" },
    ],
  },
  {
    id: "moge3",
    title: "MoGe-3",
    venue: "arXiv 2026.07 · Microsoft / 清华",
    date: "2026.07",
    tasks: ["depth", "geometry"],
    oneLiner: "在 3D 稀疏体素壳里精修点图，专门解决细结构和深度不连续处的过平滑。",
    whyItMatters:
      "2026 年单目几何的剩余误差，很多不是「整体尺度错了」，而是 2D 解码器按图像邻域混合了 3D 上不相邻的表面。SSR 用稀疏 3D 卷积按真实空间邻域聚合，细杆、栏杆、小物体明显更干净。这是深度估计从「好看的深度图」回到「可用的 3D」的一步。",
    numbers: [
      "9 个零样本基准上全局和局部精度领先",
      "在严格细结构指标 δ<0.01 上提升最大",
      "ViT-L 370M / ViT-G 1.25B，度量 + 法向",
    ],
    links: [
      { label: "论文", href: "https://arxiv.org/abs/2607.17967" },
      { label: "项目页", href: "https://qft-333.github.io/moge3page/" },
    ],
  },
  {
    id: "md2e",
    title: "MD2E",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["depth"],
    oneLiner: "训练和推理都不要相机内参，用深度到边缘的频谱线索标定度量尺度。",
    whyItMatters:
      "Metric3D 需要标定；UniDepth / Depth Pro 尝试摆脱内参。MD2E 的观察很经典：焦距和深度一起变时，RGB 看起来差不多，但边缘图的频谱会系统性偏移。光谱分位数估计器把「边缘频率」当成尺度代理。这是 2026 年度量深度一条很「CV」的路。",
    numbers: ["无内参的零样本 / 微调度量深度均为当时领先", "用预测边缘正则化深度边界"],
    links: [
      {
        label: "论文 PDF",
        href: "https://openaccess.thecvf.com/content/CVPR2026/papers/Ning_MD2E_Modeling_Depth-to-Edge_Cues_for_Monocular_Metric_Depth_Estimation_CVPR_2026_paper.pdf",
      },
      { label: "项目页", href: "https://2j472no.github.io/MD2E/" },
    ],
  },
  {
    id: "rosamdepth",
    title: "RoSAMDepth",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["depth", "segmentation"],
    oneLiner: "恶劣天气下的自监督深度，用 SAM 的物体级掩码当结构先验。",
    whyItMatters:
      "语义分割当深度先验有两个老问题：同类实例深度差很大，以及闭集类别见不到新物体。SAM 的实例掩码更合适。RoSAMDepth 把物体信息写进表征对比、区域平滑和可靠性加权，雨雾夜间深度不再糊成一片。",
    numbers: ["多种真实恶劣天气数据上超过此前自监督深度方法", "物体内部深度更锐、更一致"],
    links: [
      {
        label: "论文",
        href: "https://openaccess.thecvf.com/content/CVPR2026/html/Gao_RoSAMDepth_Robust_Self-supervised_Depth_Estimation_Leveraging_Segment_Anything_Model_CVPR_2026_paper.html",
      },
    ],
  },
  {
    id: "sam3d",
    title: "SAM 3D",
    venue: "CVPR 2026 · Meta",
    date: "2025.11 / 2026.06",
    tasks: ["geometry", "segmentation"],
    oneLiner: "从自然图像重建物体的几何、纹理和布局，遮挡和杂乱场景是主场。",
    whyItMatters:
      "它不完全是「深度估计」，但属于传统 3D 视觉被基础模型接管的同一波。人在环 + 模型在环的数据引擎打破 3D 标注瓶颈。真实物体人偏好约 5:1，场景约 6:1。和 DA3 / MoGe 互补：一个出场景几何，一个出可资产化的物体。",
    numbers: ["真实物体人偏好至少 5:1，场景 6:1", "新基准 SA-3DAO：1K 艺术家网格"],
    links: [
      {
        label: "论文 PDF",
        href: "https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_SAM_3D_3Dfy_Anything_in_Images_CVPR_2026_paper.pdf",
      },
      { label: "项目", href: "https://ai.meta.com/sam3d" },
    ],
  },
  {
    id: "foundation-stereo",
    title: "FoundationStereo",
    venue: "NVIDIA",
    date: "2025",
    tasks: ["stereo", "depth"],
    oneLiner: "给立体匹配补上「零样本泛化」这块基础模型该有的能力。",
    whyItMatters:
      "立体匹配长期靠每个数据集微调刷榜。FoundationStereo 用约 100 万对高逼真合成数据、侧调 Depth Anything v2 的单目先验、再加长程 cost volume 滤波，把立体匹配做成可跨域直接用的深度传感器替代。",
    numbers: ["约 1M 合成立体对 + 自动清洗", "冻结 DA-v2 单目先验 + CNN 高频特征"],
    links: [{ label: "项目页", href: "https://nvlabs.github.io/FoundationStereo/" }],
  },
  {
    id: "promptstereo",
    title: "PromptStereo",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["stereo", "depth"],
    oneLiner: "迭代立体匹配的 refinement 阶段，用单目深度基础模型的解码器当 Prompt Recurrent Unit。",
    whyItMatters:
      "很多人只用基础模型提特征或初始化视差，GRU 吃不下单目结构先验。PromptStereo 把单目结构和立体运动写成 prompt 灌进深度基础模型解码器，零样本泛化明显更好，速度还跟得上。",
    numbers: ["多项数据集零样本立体匹配领先", "推理速度与现有迭代方法相当或更快"],
    links: [
      {
        label: "论文",
        href: "https://openaccess.thecvf.com/content/CVPR2026/html/Wang_PromptStereo_Zero-Shot_Stereo_Matching_via_Structure_and_Motion_Prompts_CVPR_2026_paper.html",
      },
    ],
  },
  {
    id: "pip-stereo",
    title: "Pip-Stereo",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["stereo"],
    oneLiner: "视差更新在空间上稀疏、时间上冗余，于是可以把迭代立体匹配剪成近乎一遍推理。",
    whyItMatters:
      "RAFT 式迭代在实验室很准，在 Jetson 上很痛。Pip-Stereo 的 FlashGRU 相对原生 ConvGRU 在 2K 分辨率约 7.28× 加速、峰值显存降 76.6%。Orin NX 上 320×640 约 75 ms。传统 CV 的工程线并没有停。",
    numbers: [
      "Jetson Orin NX FP16：320×640 约 75 ms",
      "RTX 4090：约 19 ms",
      "FlashGRU：7.28× 加速，峰值显存 -76.6%",
    ],
    links: [
      {
        label: "论文",
        href: "https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_Pip-Stereo_Progressive_Iterations_Pruner_for_Iterative_Optimization_based_Stereo_Matching_CVPR_2026_paper.html",
      },
    ],
  },
  {
    id: "ofm",
    title: "Optical Flow Matching",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["flow"],
    oneLiner: "光流不再只估计「像素去了哪」，而是学一个随时间演化的速度场。",
    whyItMatters:
      "从 FlowNet 到 RAFT，深度光流仍是离散对应。OFM 把它改写成连续输运：用 Triangle Velocities Synergy 构造物理上稳定的速度，再用 ODE 求解。Sintel / KITTI / Spring 上精度和时间一致性一起涨，跨数据集泛化更明显。",
    numbers: ["Sintel、KITTI、Spring 上达到当时领先", "更强跨数据集泛化与时间平滑"],
    links: [
      {
        label: "论文",
        href: "https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Optical_Flow_Matching_Reframing_Optical_Flow_as_Continuous_Transport_Dynamics_CVPR_2026_paper.html",
      },
    ],
  },
  {
    id: "flowfm",
    title: "FlowFM",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["flow"],
    oneLiner: "给暗光光流做一个真正的 Flow Matching 模型，而不是再套一遍扩散。",
    whyItMatters:
      "暗光里判别模型不稳、扩散模型场不连续且慢。FlowFM 用真值光流约束的全局变换路径，一步去噪，并用傅里叶解码器分别处理运动强度（振幅）和空间关系（相位）。FCDN / VBOF 上刷新暗光光流记录。",
    numbers: ["FCDN、VBOF 上超过此前暗光光流方法", "一步去噪，兼顾精度和效率"],
    links: [
      {
        label: "论文",
        href: "https://openaccess.thecvf.com/content/CVPR2026/html/Zuo_FlowFM_Advancing_Dark_Optical_Flow_Estimation_with_Flow_Matching_CVPR_2026_paper.html",
      },
    ],
  },
  {
    id: "ares",
    title: "ARES",
    venue: "CVPR 2026",
    date: "2026.06",
    tasks: ["stereo", "flow"],
    oneLiner: "非对称 RGB–事件立体，一次性给出视差、光流和概率场景流。",
    whyItMatters:
      "传统 CV 的多传感器融合没有消失。事件相机补帧率，RGB 补纹理。ARES 不再把光流和视差分家，这和几何模型「一次出所有 3D 属性」是同一哲学，只是落在动态场景流上。",
    numbers: ["统一视差、光流、场景流", "面向事件与 RGB 不对称配置"],
    links: [
      {
        label: "论文 PDF",
        href: "https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_ARES_Unifying_Asymmetric_RGB-Event_Stereo_for_Probabilistic_Scene_Flow_Estimation_CVPR_2026_paper.pdf",
      },
    ],
  },
];

export const PLAYBOOK = [
  {
    job: "做分割系统",
    doThis: "先用 SAM 3 / 3.1 做开集与视频，闭集语义分割用冻结 DINOv3 + 轻量解码器。不要从随机初始化的 Mask2Former 重开。",
  },
  {
    job: "做单目深度",
    doThis: "要度量且要细结构用 MoGe-3；完全没有内参用 MD2E；恶劣天气自监督看 RoSAMDepth。深度图好看不等于点云能用。",
  },
  {
    job: "做多视图 / SLAM / 三维",
    doThis: "默认前馈几何：DA3 或 π³，而不是先跑 COLMAP 再补深度。大场景可看 DA3-Long 这类替换。",
  },
  {
    job: "做双目 / 边缘端",
    doThis: "泛化用 FoundationStereo 或 PromptStereo；上车用 Pip-Stereo 这类稀疏迭代，而不是把实验室 GRU 原样搬到 Jetson。",
  },
  {
    job: "想接 MLLM",
    doThis: "语言推理可以接 MLLM，像素级头不要直接信任它的 vision tower。VersaViT 已经把这个问题写清楚了。",
  },
];

export const CAVEATS = [
  "这不是对 CVPR 2026 全部 4,089 篇论文的穷尽综述。生成、多模态和具身智能投稿量更大，这里只收与分割、深度、立体、光流直接相关、且已经改变默认工具箱的工作。",
  "若干 2025 年中后期工作（DINOv3、VGGT、SAM 3、DA3 技术报告）被收进来，是因为它们构成了 2026 年的实际起点，而不是「日历年」切一刀。",
  "数字均摘自公开论文或官方仓库，不同设定（分辨率、是否微调）不可直接横比。需要落地时仍应在自己的域上复测。",
];
