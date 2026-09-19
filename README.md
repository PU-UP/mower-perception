# 2026 传统 CV 简报

一份面向分割、深度估计、立体匹配和光流从业者的 2026 年研究进展简报。结论不是「传统 CV 消失了」，而是默认方法已经变成：稠密基础模型 + 开集提示 + 前馈几何。

覆盖的节点包括 DINOv3、SAM 3 / 3.1、Depth Anything 3、MoGe-3、π³、CVPR 2026 上的 PromptStereo / MD2E / RoSAMDepth，以及 ECCV 2026 的 VersaViT。

## 本地运行

```bash
npm install
npm run dev -- --hostname 127.0.0.1 --port 43129
```

浏览器打开 `http://127.0.0.1:43129`。

## 技术栈

Next.js、TypeScript、Tailwind CSS、shadcn/ui。论文条目写在 `src/data/briefing.ts`。
