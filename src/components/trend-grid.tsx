import { TRENDS } from "@/data/briefing";

export function TrendGrid() {
  return (
    <section id="trends" className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6">
      <div className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-mono text-xs tracking-[0.2em] text-primary uppercase">
            五条主线
          </p>
          <h2 className="font-heading mt-2 text-3xl font-bold">
            2026 年传统视觉在做什么
          </h2>
        </div>
        <p className="max-w-md text-sm leading-7 text-muted-foreground">
          不是再刷 Cityscapes 零点几个 mIoU。主场已经变成开集、度量几何、以及把基础模型接进经典对应问题。
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {TRENDS.map((trend) => (
          <article
            key={trend.index}
            className="border border-border bg-card p-6 shadow-[0_1px_0_oklch(0.24_0.035_250/0.04)]"
          >
            <div className="font-mono text-xs text-primary">{trend.index}</div>
            <h3 className="font-heading mt-3 text-xl font-semibold leading-snug">
              {trend.title}
            </h3>
            <p className="mt-3 text-sm leading-7 text-muted-foreground">
              {trend.body}
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}
