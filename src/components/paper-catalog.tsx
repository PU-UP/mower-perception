"use client";

import { useMemo, useState } from "react";
import { ArrowUpRight, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PAPERS, TASKS, type TaskId } from "@/data/briefing";

const TASK_LABEL: Record<TaskId, string> = {
  backbone: "稠密骨干",
  segmentation: "分割",
  depth: "深度",
  geometry: "几何",
  stereo: "立体",
  flow: "光流",
};

export function PaperCatalog() {
  const [task, setTask] = useState<(typeof TASKS)[number]["id"]>("all");
  const [query, setQuery] = useState("");

  const papers = useMemo(() => {
    const q = query.trim().toLowerCase();
    return PAPERS.filter((paper) => {
      const taskOk = task === "all" || paper.tasks.includes(task);
      if (!taskOk) return false;
      if (!q) return true;
      return (
        paper.title.toLowerCase().includes(q) ||
        paper.oneLiner.includes(query.trim()) ||
        paper.venue.toLowerCase().includes(q)
      );
    });
  }, [task, query]);

  return (
    <section id="papers" className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="font-mono text-xs tracking-[0.2em] text-primary uppercase">
            论文卡片
          </p>
          <h2 className="font-heading mt-2 text-3xl font-bold">
            可以直接跟进的工作
          </h2>
          <p className="mt-2 max-w-xl text-sm leading-7 text-muted-foreground">
            按任务筛选。每张卡片写清「为什么值得传统 CV 的人看」，并链到公开论文或项目页。
          </p>
        </div>
        <label className="relative block w-full max-w-sm">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索 SAM、深度、光流…"
            className="h-10 w-full rounded-lg border border-border bg-card pr-3 pl-9 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/40"
          />
        </label>
      </div>

      <div className="mb-6 flex flex-wrap gap-2">
        {TASKS.map((item) => (
          <Button
            key={item.id}
            type="button"
            size="sm"
            variant={task === item.id ? "default" : "outline"}
            onClick={() => setTask(item.id)}
          >
            {item.label}
          </Button>
        ))}
      </div>

      {papers.length === 0 ? (
        <div className="border border-dashed border-border bg-card px-6 py-16 text-center">
          <p className="font-heading text-lg">没有匹配的条目</p>
          <p className="mt-2 text-sm text-muted-foreground">
            换一个任务，或清空搜索词再看。分割、深度、立体、光流都还在出新方法。
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {papers.map((paper) => (
            <Card key={paper.id} className="rounded-none bg-card shadow-none ring-border">
              <CardHeader>
                <div className="flex flex-wrap gap-1.5">
                  {paper.tasks.map((id) => (
                    <Badge key={id} variant="secondary">
                      {TASK_LABEL[id]}
                    </Badge>
                  ))}
                </div>
                <CardTitle className="font-heading mt-2 text-xl">
                  {paper.title}
                </CardTitle>
                <p className="font-mono text-xs text-muted-foreground">
                  {paper.venue}
                </p>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm leading-7">{paper.oneLiner}</p>
                <p className="text-sm leading-7 text-muted-foreground">
                  {paper.whyItMatters}
                </p>
                <ul className="space-y-1.5 border-l-2 border-primary/40 pl-3 text-sm leading-6">
                  {paper.numbers.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
                <div className="flex flex-wrap gap-2 pt-1">
                  {paper.links.map((link) => (
                    <a
                      key={link.href}
                      href={link.href}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                    >
                      {link.label}
                      <ArrowUpRight className="size-3.5" />
                    </a>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </section>
  );
}
