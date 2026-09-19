import { Hero } from "@/components/hero";
import { PaperCatalog } from "@/components/paper-catalog";
import { Playbook } from "@/components/playbook";
import { SiteHeader } from "@/components/site-header";
import { Timeline } from "@/components/timeline";
import { TrendGrid } from "@/components/trend-grid";

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <main className="flex-1">
        <Hero />
        <TrendGrid />
        <Timeline />
        <PaperCatalog />
        <Playbook />
      </main>
      <footer className="border-t border-border px-4 py-8 text-center text-xs leading-6 text-muted-foreground sm:px-6">
        基于 2025 下半年至 2026 年 9 月公开论文整理。不是会议官方综述。
      </footer>
    </div>
  );
}
