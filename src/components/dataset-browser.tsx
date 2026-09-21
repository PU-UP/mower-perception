"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

type Sample = { id: string; garden: string }
type Detail = {
  id: string; input: string; truth: string; revision: string | null
  comparisons: { model: string; name: string; image: string; metrics: Record<string, number | null> }[]
}

export function EvaluationMetrics({ metrics }: { metrics: Record<string, number | null> }) {
  return <dl className="flex flex-wrap gap-x-6 gap-y-2 text-xs">
    {[["mowable_iou", "可割 IoU"], ["nonmowable_false_positive_rate", "不可割误判率"], ["predicted_mowable_error_fraction", "预测可割错误比例"], ["boundary_f1", "边界 F1"]].map(([key, label]) => (
      <div key={key}><dt className="text-muted-foreground">{label}</dt><dd>{metrics[key] == null ? "无定义" : `${(metrics[key]! * 100).toFixed(2)}%`}</dd></div>
    ))}
  </dl>
}

export function DatasetBrowser({ busy, onRun }: { busy: boolean; onRun: (id: string) => void }) {
  const [samples, setSamples] = useState<Sample[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [garden, setGarden] = useState("all")
  const [selected, setSelected] = useState("")
  const [detail, setDetail] = useState<Detail | null>(null)
  const [error, setError] = useState("")
  useEffect(() => {
    const controller = new AbortController()
    fetch("/api/grass", { signal: controller.signal }).then(async res => {
      if (!res.ok) throw new Error("无法读取评测集")
      const data = await res.json()
      setSamples(data.samples); setTotal(data.total)
      setSelected(data.samples[0]?.id ?? "")
    }).catch(err => { if (!controller.signal.aborted) setError(String(err)) })
    return () => controller.abort()
  }, [])
  useEffect(() => {
    if (!selected) return
    const controller = new AbortController()
    fetch(`/api/grass/${selected}`, { signal: controller.signal }).then(async res => {
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setDetail(data); setError("")
    }).catch(err => { if (!controller.signal.aborted) setError(String(err)) })
    return () => controller.abort()
  }, [selected])
  const filtered = samples.filter(s => garden === "all" || s.garden === garden)
  const index = filtered.findIndex(s => s.id === selected)
  const current = detail?.id === selected ? detail : null
  const selectClass = "rounded-md border bg-background px-3 py-2 text-sm"
  return <Card>
    <CardHeader>
      <CardTitle>GrassSegHB 标注评测集</CardTitle>
      <CardDescription>已下载 {samples.length} / {total ?? "…"} 张 · 真实人工标注。白色=可割，黑色=不可割；grass 预测仅作可割代理。</CardDescription>
    </CardHeader>
    <CardContent className="grid gap-4">
      {samples.length ? <>
        <div className="flex flex-wrap items-center gap-2">
          <label>庭院 <select aria-label="庭院" className={selectClass} disabled={busy} value={garden} onChange={e => {
            const value = e.target.value; setGarden(value)
            setSelected(samples.find(s => value === "all" || s.garden === value)!.id)
          }}><option value="all">全部庭院</option>{[...new Set(samples.map(s => s.garden))].map(g => <option key={g} value={g}>庭院 {g}</option>)}</select></label>
          <label>样本 <select aria-label="评测样本" className={selectClass} disabled={busy} value={selected} onChange={e => setSelected(e.target.value)}>{filtered.map(s => <option key={s.id} value={s.id}>{s.id}</option>)}</select></label>
          <Button size="sm" variant="outline" disabled={busy || index <= 0} onClick={() => setSelected(filtered[index - 1].id)}>上一张</Button>
          <Button size="sm" variant="outline" disabled={busy || index >= filtered.length - 1} onClick={() => setSelected(filtered[index + 1].id)}>下一张</Button>
          <span className="text-xs text-muted-foreground">{index + 1} / {filtered.length}</span>
          <Button size="sm" disabled={busy || !current || !!error} onClick={() => onRun(selected)}>用当前模型重新推理</Button>
        </div>
        {current ? <>
          <div className="grid gap-4 sm:grid-cols-2">{[[current.input, "原图"], [current.truth, "人工标注"]].map(([src, title]) => <figure key={title}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt={`${selected} ${title}`} className="h-auto w-full rounded-lg" />
            <figcaption className="mt-1 text-sm">{title}</figcaption>
          </figure>)}</div>
          <p className="text-xs text-muted-foreground">以下为已保存的离线评测（版本 {current.revision?.slice(0, 7) ?? "—"}），不会随当前模型按钮变化。逐图指标按原始标注计算，边界容差 3 像素。</p>
          {current.comparisons.length ? current.comparisons.map(c => <div key={c.model} className="grid gap-2 rounded-lg border p-3">
            <h3 className="text-sm font-medium">{c.name}</h3>
            <EvaluationMetrics metrics={c.metrics} />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img loading="lazy" src={c.image} alt={`${c.name}：原图、标注、预测、错误对比`} className="h-auto w-full" />
            <p className="text-xs text-muted-foreground">从左到右：原图 / 标注 / 预测 / 错误。红=误割，蓝=漏割。</p>
          </div>) : <p className="text-sm">尚无完整的本地离线评测结果，可以先重新推理当前图片。</p>}
        </> : <p role="status" className="text-sm">正在加载图片和标注…</p>}
      </> : <p className="text-sm">{total === null ? "正在读取评测集…" : "本地尚无可用标注图片，请按评测文档下载数据。原有样例和上传仍可使用。"}</p>}
      {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
    </CardContent>
  </Card>
}
