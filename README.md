# MowerSeg

面向智能割草机感知的起步仓库：闭集可通行语义分割。当前用 **SegFormer-B0（ADE20K）** 做零样本推理，再映射到草坪产品类别，直接输出叠加图、色块 mask 和每类占比。

这是量产学生网的框架，不是 SAM / DA3 上板方案。入职后替换 `configs/mower_seg.yaml` 和自己的数据即可复用同一套推理接口。

## 产品类别

可割草坪、灌木植被、裸土地形、铺装路面、人与动物、车辆、障碍与结构、水体、天空。

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
python -m mowerseg assets/samples/lawn_path.jpg -o outputs
```

同时开 Web 预览：

```bash
export PYTHONPATH=.
./scripts/dev.sh
```

浏览器打开 `http://127.0.0.1:43129`。推理 API 在 `http://127.0.0.1:43131`。

## 目录

- `configs/mower_seg.yaml`：类别、颜色、ADE20K 映射、样例
- `mowerseg/`：模型加载、映射、可视化、CLI、FastAPI
- `assets/samples/`：花园 / 草坪样例图
- `src/`：Next.js 预览页

## 换自己的数据

1. 按 yaml 里的 `id` 准备 `images/` 与 `masks/`
2. 用同一套类别微调轻量分割网（PP-LiteSeg / YOLO26n-sem 等）
3. 把 `model.name` 换成新权重，推理脚本不用改
