import { CAVEATS, PLAYBOOK } from "@/data/briefing";

export function Playbook() {
  return (
    <section id="playbook" className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6">
      <p className="font-mono text-xs tracking-[0.2em] text-primary uppercase">
        落地怎么选
      </p>
      <h2 className="font-heading mt-2 text-3xl font-bold">
        2026 年还在做这些任务，默认起点已经变了
      </h2>
      <div className="mt-8 grid gap-4 md:grid-cols-2">
        {PLAYBOOK.map((item) => (
          <article key={item.job} className="border border-border bg-card p-5">
            <h3 className="font-heading text-lg font-semibold">{item.job}</h3>
            <p className="mt-2 text-sm leading-7 text-muted-foreground">
              {item.doThis}
            </p>
          </article>
        ))}
      </div>

      <div className="mt-12 border border-border bg-muted/60 p-6">
        <h3 className="font-heading text-lg font-semibold">阅读边界</h3>
        <ul className="mt-4 space-y-3 text-sm leading-7 text-muted-foreground">
          {CAVEATS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
