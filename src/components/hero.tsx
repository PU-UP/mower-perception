export function Hero() {
  return (
    <section
      id="top"
      className="relative overflow-hidden bg-[#132038] text-[#f6f0e4]"
    >
      <div className="hero-grid pointer-events-none absolute inset-0 opacity-70" />
      <div className="relative mx-auto grid w-full max-w-6xl gap-10 px-4 py-16 sm:px-6 sm:py-24 lg:grid-cols-[1.4fr_0.8fr] lg:items-end">
        <div>
          <p className="mb-5 font-mono text-xs tracking-[0.22em] text-[#e2b089] uppercase">
            Field notes · Sep 2026
          </p>
          <h1 className="font-heading text-4xl leading-[1.15] font-bold tracking-tight sm:text-5xl lg:text-6xl">
            传统 CV 还在，
            <br />
            只是换了一套默认方法。
          </h1>
          <p className="mt-6 max-w-2xl text-base leading-8 text-[#d9d1c3] sm:text-lg">
            分割、深度估计、立体匹配、光流没有被生成模型写完。2026
            年真正的变化是：闭集标签换成概念提示，单目深度并进前馈几何，骨干默认变成
            DINOv3 / SAM 3 这类稠密基础模型。
          </p>
        </div>
        <aside className="border border-white/15 bg-white/5 p-5 backdrop-blur-sm">
          <p className="font-mono text-[11px] tracking-widest text-[#e2b089] uppercase">
            这份简报覆盖
          </p>
          <ul className="mt-4 space-y-3 text-sm leading-6 text-[#ece6da]">
            <li>CVPR 2026 丹佛 · 16,092 投稿 / 4,089 接收</li>
            <li>ICLR 2026 Oral：Depth Anything 3</li>
            <li>ECCV 2026：MLLM 骨干对稠密任务仍然偏弱</li>
            <li>SAM 3.1、MoGe-3、π³、PromptStereo</li>
          </ul>
        </aside>
      </div>
    </section>
  );
}
