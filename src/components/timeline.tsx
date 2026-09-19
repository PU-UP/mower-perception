import { TIMELINE } from "@/data/briefing";

export function Timeline() {
  return (
    <section
      id="timeline"
      className="border-y border-border bg-[#132038] text-[#f6f0e4]"
    >
      <div className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6">
        <p className="font-mono text-xs tracking-[0.2em] text-[#e2b089] uppercase">
          时间线
        </p>
        <h2 className="font-heading mt-2 text-3xl font-bold">
          从 VGGT 到 ECCV 2026
        </h2>
        <ol className="mt-10 grid gap-0 sm:grid-cols-2 lg:grid-cols-5">
          {TIMELINE.map((item, index) => (
            <li
              key={item.date}
              className="border-t border-white/15 py-5 pr-4 sm:border-t-0 sm:border-l sm:pl-4 lg:min-h-56"
            >
              <div className="font-mono text-[11px] text-[#e2b089]">
                {String(index + 1).padStart(2, "0")} · {item.date}
              </div>
              <h3 className="font-heading mt-2 text-lg font-semibold">
                {item.title}
              </h3>
              <p className="mt-2 text-sm leading-6 text-[#cfc6b8]">
                {item.detail}
              </p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
