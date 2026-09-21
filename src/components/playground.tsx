"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Leaf, LoaderCircle, ShieldAlert, Upload } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { InferResponse, ModelCard, TaxonomyResponse } from "@/lib/types"
import { DatasetBrowser, EvaluationMetrics } from "@/components/dataset-browser"
import { cn } from "@/lib/utils"

function toHex(color: number[]) {
  return `#${color.map((c) => c.toString(16).padStart(2, "0")).join("")}`
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function withModel(url: string, modelId: string) {
  const separator = url.includes("?") ? "&" : "?"
  return `${url}${separator}model=${encodeURIComponent(modelId)}`
}

export function Playground() {
  const inputRef = useRef<HTMLInputElement>(null)
  const lastUploadRef = useRef<File | null>(null)
  const modelIdRef = useRef("segformer_b0_ade20k")
  const [taxonomy, setTaxonomy] = useState<TaxonomyResponse | null>(null)
  const [models, setModels] = useState<ModelCard[]>([])
  const [modelId, setModelId] = useState<string>("segformer_b0_ade20k")
  const [result, setResult] = useState<InferResponse | null>(null)
  const [activeSample, setActiveSample] = useState<string>("lawn_path")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<"overlay" | "mask" | "input">("overlay")

  useEffect(() => {
    modelIdRef.current = modelId
  }, [modelId])

  const runSample = useCallback(async (id: string, selectedModel?: string) => {
    const model = selectedModel ?? modelIdRef.current
    setBusy(true)
    setError(null)
    setResult(null)
    setActiveSample(id)
    lastUploadRef.current = null
    try {
      const res = await fetch(withModel(`/api/infer-sample/${id}`, model), {
        method: "POST",
      })
      if (!res.ok) throw new Error(await res.text())
      setResult(await res.json())
      setView("mask")
    } catch (err) {
      setError(err instanceof Error ? err.message : "推理失败")
    } finally {
      setBusy(false)
    }
  }, [])

  const runUpload = useCallback(async (file: File, selectedModel?: string) => {
    const model = selectedModel ?? modelIdRef.current
    setBusy(true)
    setError(null)
    setResult(null)
    setActiveSample("upload")
    lastUploadRef.current = file
    try {
      const body = new FormData()
      body.append("file", file)
      const res = await fetch(withModel("/api/infer", model), {
        method: "POST",
        body,
      })
      if (!res.ok) throw new Error(await res.text())
      setResult(await res.json())
      setView("overlay")
    } catch (err) {
      setError(err instanceof Error ? err.message : "推理失败")
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    fetch("/api/taxonomy")
      .then((res) => {
        if (!res.ok) throw new Error("taxonomy")
        return res.json()
      })
      .then((data: TaxonomyResponse) => {
        if (cancelled) return
        setTaxonomy(data)
        const available = data.models ?? []
        setModels(available)
        const initial =
          data.default_model || data.model_id || available[0]?.id || "segformer_b0_ade20k"
        modelIdRef.current = initial
        setModelId(initial)
        if (data.samples[0]) {
          void runSample(data.samples[0].id, initial)
        }
      })
      .catch(() => {
        if (!cancelled) setError("后端未就绪，请确认推理服务已启动。")
      })
    return () => {
      cancelled = true
    }
  }, [runSample])

  async function switchModel(nextModel: string) {
    if (nextModel === modelId || busy) return
    modelIdRef.current = nextModel
    setModelId(nextModel)
    setResult(null)
    if (activeSample === "upload" && lastUploadRef.current) {
      await runUpload(lastUploadRef.current, nextModel)
      return
    }
    const sampleId =
      activeSample !== "upload"
        ? activeSample
        : taxonomy?.samples[0]?.id ?? "lawn_path"
    await runSample(sampleId, nextModel)
  }

  const preview = useMemo(() => {
    if (!result) return null
    if (view === "mask") return result.mask
    if (view === "input") return result.input
    return result.overlay
  }, [result, view])

  const activeModel =
    models.find((item) => item.id === modelId) ||
    models.find((item) => item.id === result?.model?.id)

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6 lg:py-10">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="max-w-2xl">
          <p className="mb-2 text-xs font-medium tracking-[0.18em] text-sage uppercase">
            MowerSeg Factory
          </p>
          <h1 className="font-heading text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            割草机可通行语义分割
          </h1>
          <p className="mt-3 text-sm leading-7 text-muted-foreground sm:text-base">
            在同一产品类别契约下手动切换模型对比效果。当前支持 ADE20K /
            PASCAL VOC 预训练映射。ADE20K 的 grass 仅是“可割”的代理预测，不代表可安全通行。
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">闭集 9 类</Badge>
          <Badge variant="outline">可切换模型</Badge>
          <Badge variant="secondary">板端学生网起点</Badge>
        </div>
      </header>

      <DatasetBrowser busy={busy} onRun={(id) => runSample(`grass-${id}`)} />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(280px,0.8fr)]">
        <Card className="overflow-hidden">
          <CardHeader className="border-b">
            <CardTitle>推理结果</CardTitle>
            <CardDescription>
              绿为可割草坪，红/橙为安全类，灰为铺装，深蓝绿为灌木。
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="mb-4 flex flex-col gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                  模型
                </span>
                {(models.length > 0
                  ? models
                  : [
                      {
                        id: modelId,
                        display_name: modelId,
                      } as ModelCard,
                    ]
                ).map((item) => (
                  <Button
                    key={item.id}
                    size="sm"
                    variant={modelId === item.id ? "default" : "outline"}
                    disabled={busy}
                    onClick={() => void switchModel(item.id)}
                  >
                    {item.display_name}
                  </Button>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {taxonomy?.samples.map((sample) => (
                  <Button
                    key={sample.id}
                    size="sm"
                    variant={activeSample === sample.id ? "default" : "outline"}
                    disabled={busy}
                    onClick={() => runSample(sample.id)}
                  >
                    {sample.title}
                  </Button>
                ))}
                <Button
                  size="sm"
                  variant={activeSample === "upload" ? "default" : "outline"}
                  disabled={busy}
                  onClick={() => inputRef.current?.click()}
                >
                  <Upload data-icon="inline-start" />
                  上传图片
                </Button>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (file) void runUpload(file)
                  }}
                />
              </div>
            </div>

            <div className="relative overflow-hidden rounded-xl bg-muted ring-1 ring-foreground/10">
              {preview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={preview}
                  alt="分割结果"
                  className="h-auto w-full object-contain"
                />
              ) : (
                <div className="flex aspect-[16/10] flex-col items-center justify-center gap-3 px-6 text-center text-sm text-muted-foreground">
                  {busy ? (
                    <>
                      <LoaderCircle className="size-6 animate-spin" />
                      正在推理，首次加载权重会稍慢。
                    </>
                  ) : (
                    <>
                      <Leaf className="size-6 text-sage" />
                      选择一张样例，或上传花园 / 草坪照片。
                    </>
                  )}
                </div>
              )}
              {busy && preview ? (
                <div className="absolute inset-0 flex items-center justify-center bg-background/40 text-sm">
                  <LoaderCircle className="mr-2 size-4 animate-spin" />
                  推理中
                </div>
              ) : null}
            </div>

            {result ? (
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <div className="flex gap-2">
                  {(
                    [
                      ["overlay", "叠加"],
                      ["mask", "色块"],
                      ["input", "原图"],
                    ] as const
                  ).map(([key, label]) => (
                    <Button
                      key={key}
                      size="xs"
                      variant={view === key ? "default" : "ghost"}
                      onClick={() => setView(key)}
                    >
                      {label}
                    </Button>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  模型推理 {result.stats.latency_ms.toFixed(0)} ms · {result.stats.device} ·{" "}
                  {result.model?.display_name ||
                    activeModel?.display_name ||
                    result.stats.model.split("/").at(-1)}
                </p>
              </div>
            ) : null}

            {result?.evaluation ? (
              <div className="mt-4"><p className="mb-2 text-sm">本次推理与人工标注比较</p><EvaluationMetrics metrics={result.evaluation} /></div>
            ) : null}
            {activeSample.startsWith("grass-") ? <p className="mt-2 text-xs text-muted-foreground">评测样本：{activeSample.slice(6)}；切换模型保持同一张图。VOC 无草地类别，不计算可割指标。</p> : null}

            {error ? (
              <p className="mt-3 text-sm text-destructive">{error}</p>
            ) : null}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>当前模型</CardTitle>
              <CardDescription>
                逻辑模型与源数据集 taxonomy，切换后会重新推理当前样例。
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2 text-sm">
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">名称</span>
                <span className="text-right">
                  {activeModel?.display_name || modelId}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">权重</span>
                <span className="max-w-[180px] truncate text-right text-xs">
                  {activeModel?.hub_id || result?.stats.model || "—"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">训练数据集</span>
                <span className="text-right">
                  {activeModel?.output_taxonomy ||
                    result?.model?.output_taxonomy ||
                    "—"}
                </span>
              </div>
              <p className="text-xs leading-5 text-muted-foreground">
                {activeModel?.output_taxonomy === "pascal_voc"
                  ? "VOC 无草地类别，不能用于判断 CNN 的割草分割能力。"
                  : "仅将 grass 映射为可割代理；植被与裸地不算可割。二分类评测不验证人、动物或小障碍物检测能力。"}
              </p>
              {result?.stats.input_shape ? (
                <p className="text-xs text-muted-foreground">
                  实际输入：{result.stats.input_shape.slice(2).join(" × ")}（高 × 宽）
                </p>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>产品类别</CardTitle>
              <CardDescription>
                配置在 <code>configs/mower_seg.yaml</code>，换数据时先改这里。
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2">
              {(taxonomy?.classes ?? [])
                .filter((item) => item.name !== "ignore")
                .map((item) => (
                  <div key={item.id} className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span
                        className="size-3 rounded-full ring-1 ring-black/10"
                        style={{ backgroundColor: toHex(item.color) }}
                      />
                      <span className="text-sm">{item.name_zh}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {item.traversable ? "可通行" : item.safety ? "安全" : "禁止"}
                    </span>
                  </div>
                ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>像素占比</CardTitle>
              <CardDescription>预测像素占比，不是准确率；需有标注数据才能评测。</CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <div className="grid gap-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <p className="text-xs text-muted-foreground">可割草坪</p>
                      <p className="font-heading text-xl">
                        {percent(result.stats.traversable_ratio)}
                      </p>
                    </div>
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <p className="flex items-center gap-1 text-xs text-muted-foreground">
                        <ShieldAlert className="size-3" />
                        安全类
                      </p>
                      <p className="font-heading text-xl">
                        {percent(result.stats.safety_ratio)}
                      </p>
                    </div>
                  </div>
                  <div className="grid gap-2">
                    {result.stats.classes.map((row) => (
                      <div key={row.id} className="grid gap-1">
                        <div className="flex items-center justify-between text-xs">
                          <span>{row.name_zh}</span>
                          <span className="text-muted-foreground">
                            {percent(row.ratio)}
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                          <div
                            className={cn("h-full rounded-full")}
                            style={{
                              width: `${Math.max(row.ratio * 100, 1.2)}%`,
                              backgroundColor: toHex(row.color),
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">推理后显示每类占比。</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
