"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

type Metrics = Record<string, number | null>
type Report = {
  available: boolean; reason?: string; sample_count: number; scope: string
  frozen: { output_wh: number[]; selected_epoch: number; boundary_radius: number; checkpoint_sha256: string }
  hardware: { gpu: string; precision: string; threads: number }
  timing_protocol: { batch: number; warmup_per_model: number; images: number; repeats_per_image: number }
  models: { id: string; name: string; metrics: Metrics; input_nchw: number[]; counts: Record<string, number>
    per_group: Record<string, { n: number; metrics: Metrics }>
    latency: Record<string, { n: number; p50_ms: number; p95_ms: number }> }[]
}
const metrics = [
  ["grass_iou", "草地 IoU ↑"], ["grass_false_positive_rate", "草地假阳性率 ↓"],
  ["grass_false_negative_rate", "草地漏检率 ↓"], ["predicted_grass_error_fraction", "预测草地中的错误比例 ↓"],
  ["boundary_precision", "边界 precision ↑"], ["boundary_recall", "边界 recall ↑"], ["boundary_f1", "边界 F1 ↑"],
]
const score = (n: number | null | undefined) => n == null ? "无定义" : `${(n * 100).toFixed(2)}%`

export function YcorComparison() {
  const [report, setReport] = useState<Report | null>(null)
  const [error, setError] = useState("")
  useEffect(() => {
    const controller = new AbortController()
    fetch("/api/ycor-summary", { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error("YCOR 对比结果读取失败")
      setReport(await response.json())
    }).catch(err => { if (!controller.signal.aborted) setError(String(err)) })
    return () => controller.abort()
  }, [])
  if (error || !report?.available) return <p role="status">{error || report?.reason || (report ? "完整评测报告不可用" : "正在读取 YCOR 真实评测结果…")}</p>
  return <Card>
    <CardHeader><CardTitle>YCOR：训练后的 LR-ASPP 与现成 B0</CardTitle></CardHeader>
    <CardContent className="grid gap-5">
      <p className="text-sm leading-6">{report.sample_count} 张冻结留出图 · 输出宽×高 {report.frozen.output_wh.join("×")} · 最佳模型来自验证集第 {report.frozen.selected_epoch} 轮。{report.scope}。</p>
      <p className="text-sm leading-6">正类仅为官方 traversable grass；其他有效类别为负类，未标注区域不计入指标。LR-ASPP 使用 YCOR 领域训练，B0 使用 ADE20K 现成权重，训练条件不同。地点可能重叠，内部来源分组为推断分组；这是探索性留出评测，不能证明安全可割。</p>
      <div className="overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm">
        <caption className="sr-only">YCOR 冻结留出集效果对比</caption>
        <thead><tr className="border-b"><th className="py-2">指标</th>{report.models.map(m => <th key={m.id}>{m.name}</th>)}</tr></thead>
        <tbody>{metrics.map(([key, label]) => <tr key={key} className="border-b"><th scope="row" className="py-2 font-normal">{label}</th>{report.models.map(m => <td key={m.id}>{score(m.metrics[key])}</td>)}</tr>)}</tbody>
      </table></div>
      <p className="text-xs leading-6 text-muted-foreground">IoU=TP/(TP+FP+FN)；假阳性率=FP/(FP+TN)；漏检率=FN/(TP+FN)；预测草地错误比例=FP/(TP+FP)。边界为四邻域变化的两侧像素，不含画面外框；相同输出尺寸下容差为{report.frozen.boundary_radius}像素，双方排除未标注区域周围{report.frozen.boundary_radius + 1}像素，避免伪边界。整体按计数累加，不把像素视为独立样本。</p>
      <h3 className="text-sm font-medium">离线速度 · P50 / P95</h3>
      <div className="overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm">
        <caption className="sr-only">同一输出配置下的分阶段推理速度</caption>
        <thead><tr className="border-b"><th className="py-2">阶段</th>{report.models.map(m => <th key={m.id}>{m.name}</th>)}</tr></thead>
        <tbody>{[["preprocess_ms", "预处理（含 H2D）"], ["forward_ms", "模型前向"], ["postprocess_ms", "后处理（含 D2H）"], ["pipeline_ms", "完整已解码图像流水线"]].map(([key, label]) => <tr key={key} className="border-b"><th scope="row" className="py-2 font-normal">{label}</th>{report.models.map(m => <td key={m.id}>{m.latency[key].p50_ms.toFixed(3)} / {m.latency[key].p95_ms.toFixed(3)} ms</td>)}</tr>)}</tbody>
      </table></div>
      <p className="text-xs leading-6 text-muted-foreground">{report.hardware.gpu} · {report.hardware.precision} · {report.hardware.threads}线程 · batch={report.timing_protocol.batch}。预热{report.timing_protocol.warmup_per_model}次/模型；{report.timing_protocol.images}张真实图×{report.timing_protocol.repeats_per_image}次，交替模型顺序。CUDA同步、包含CPU/GPU传输；不含预先完成的读取/解码、网页、图片保存或指标计算。{report.models.map(m => `${m.name}输入高×宽${m.input_nchw.slice(2).join("×")}`).join("；")}。效果与速度采用同一输出配置，不能与另一数据集的历史速度直接比较。</p>
      <details><summary className="cursor-pointer text-sm">分组表现与混淆计数</summary>
        <p className="my-3 text-xs text-muted-foreground">以下分组为每50个IID的诊断分箱，不是已核实的独立庭院或采集序列。整体平均不能掩盖局部误检、漏检。</p>
        <div className="overflow-x-auto"><table className="w-full min-w-[700px] text-left text-xs"><thead><tr><th>诊断组</th><th>模型 / 张数</th><th>IoU</th><th>假阳性率</th><th>漏检率</th><th>边界F1</th></tr></thead>
          <tbody>{report.models.flatMap(m => Object.entries(m.per_group).map(([group, g]) => <tr className="border-b" key={`${m.id}-${group}`}><td className="py-2">{group}</td><td>{m.name} / {g.n}</td>{["grass_iou", "grass_false_positive_rate", "grass_false_negative_rate", "boundary_f1"].map(k => <td key={k}>{score(g.metrics[k])}</td>)}</tr>))}</tbody>
        </table></div>
        {report.models.map(m => <p className="mt-3 text-xs" key={m.id}>{m.name}：{["tp", "fp", "tn", "fn", "ignored"].map(k => `${k.toUpperCase()}=${m.counts[k]}`).join(" · ")}</p>)}
      </details>
      <p className="text-xs leading-6 text-muted-foreground">报告已在模型和分类规则冻结后生成；新增单图样例只是演示，不重新计算或替换上述完整评测。土路高草误检、暗光与稀疏草地漏检等案例可在“单图试验”查看。原始记录：evaluation/ycor/results.json。</p>
    </CardContent>
  </Card>
}
