"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Leaf, LoaderCircle, ShieldAlert, Upload } from "lucide-react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
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
  const [mode, setMode] = useState("dataset")
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

  const isYcor = activeModel?.output_taxonomy === "ycor_proxy"

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6 lg:py-10">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="max-w-2xl">
          <p className="mb-2 text-xs font-medium tracking-[0.18em] text-sage uppercase">
            MowerSeg Factory
          </p>
          <h1 className="font-heading text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            割草场景分割评估
          </h1>
          <p className="mt-3 text-sm leading-7 text-muted-foreground sm:text-base">
            用标注数据集比较模型的整体表现，或选择一张图片试验分割效果。草地识别仅作代理；YCOR 可通行草地标签不代表安全可割。
          </p>
        </div>

      </header>

      <Tabs value={mode} onValueChange={(value) => setMode(String(value))} className="gap-6">
        <TabsList aria-label="评估方式" className="w-full sm:w-fit">
          <TabsTrigger value="dataset" className="px-4">数据集横向对比</TabsTrigger>
          <TabsTrigger value="single" className="px-4">单图试验</TabsTrigger>
        </TabsList>
        <TabsContent value="dataset" keepMounted>
          <DatasetBrowser busy={busy} onRun={(id) => {
            setMode("single")
            void runSample(`grass-${id}`)
          }} />
        </TabsContent>
        <TabsContent value="single" keepMounted>
          <p className="mb-4 text-sm text-muted-foreground">选择样例或上传自己的图片，再切换模型比较。没有人工标注的图片只展示分割效果，不计算准确率。</p>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(280px,0.8fr)]">
        <Card className="overflow-hidden">
          <CardHeader className="border-b">
            <CardTitle>推理结果</CardTitle>
            <CardDescription>
              {isYcor ? "绿色为可通行草地代理，灰色为其他有效类别；不是安全可割判断。" : "绿为可割草坪，红/橙为安全类，灰为铺装，深蓝绿为灌木。"}
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
                {isYcor && result.raw_mask ? <a className="text-xs underline" href={result.raw_mask} download="ycor-binary-mask.png">下载原始 0/1 掩码</a> : null}
                <p className="text-xs text-muted-foreground">
                  模型前向（非完整流水线） {result.stats.latency_ms.toFixed(0)} ms · {result.stats.device} ·{" "}
                  {result.model?.display_name ||
                    activeModel?.display_name ||
                    result.stats.model.split("/").at(-1)}
                </p>
              </div>
            ) : null}

            {result?.evaluation ? (
              <div className="mt-4"><p className="mb-2 text-sm">本次推理与人工标注比较（边界容差 {result.evaluation_protocol?.boundary_radius_pixels ?? "未记录"} 像素）</p><EvaluationMetrics metrics={result.evaluation} /></div>
            ) : null}
            {activeSample.startsWith("grass-") ? <p className="mt-2 text-xs text-muted-foreground">评测样本：{activeSample.slice(6)}；切换模型保持同一张图。VOC 无草地类别；YCOR 是可通行草地代理。两者均不在此计算可割指标。</p> : null}

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
                切换模型会重新推理当前图片；训练数据表示模型此前学习的图片来源。
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
                  {isYcor ? "YCOR" : activeModel?.output_taxonomy ||
                    result?.model?.output_taxonomy ||
                    "—"}
                </span>
              </div>
              <p className="text-xs leading-5 text-muted-foreground">
                {isYcor ? "YCOR 二分类训练模型：仅区分可通行草地与其他有效类别。不能识别人、动物或水体等细分类，也不能证明安全可割。" : activeModel?.output_taxonomy === "pascal_voc"
                  ? "PASCAL VOC 是模型训练所用的数据集，其中没有草地类别，不能用于比较可割草地识别能力。"
                  : "ADE20K 是模型训练所用的通用场景分割数据集，包含草地等 150 类。它不是模型名，也不是这次的割草评测集。这里只把其中的 grass（草地）作为可割代理。"}
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
                {isYcor ? "原始标签 0=其他有效类别，1=可通行草地代理；两类都不构成通行许可。" : "模型输出统一显示为以下类别；草地不等于安全通行。"}
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2">
              {(result?.taxonomy ?? (isYcor ? [] : taxonomy?.classes) ?? [])
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
                      {isYcor ? "代理标签" : item.traversable ? "可通行" : item.safety ? "安全" : "禁止"}
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
                      <p className="text-xs text-muted-foreground">{isYcor ? "可通行草地代理" : "可割草坪"}</p>
                      <p className="font-heading text-xl">
                        {percent(isYcor ? (result.stats.classes.find((item) => item.id === 1)?.ratio ?? 0) : result.stats.traversable_ratio)}
                      </p>
                    </div>
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <p className="flex items-center gap-1 text-xs text-muted-foreground">
                        <ShieldAlert className="size-3" />
                        {isYcor ? "其他有效类别" : "安全类"}
                      </p>
                      <p className="font-heading text-xl">
                        {percent(isYcor ? (result.stats.classes.find((item) => item.id === 0)?.ratio ?? 0) : result.stats.safety_ratio)}
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
        </TabsContent>
      </Tabs>
    </div>
  )
}
