import { ArrowUpRight } from "lucide-react";

const NAV = [
  { href: "#trends", label: "主线" },
  { href: "#timeline", label: "时间线" },
  { href: "#papers", label: "论文" },
  { href: "#playbook", label: "怎么选" },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-[#132038]/90 text-white backdrop-blur-md">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-4 sm:px-6">
        <a href="#top" className="font-heading text-sm tracking-wide">
          2026 传统 CV 简报
        </a>
        <nav className="hidden items-center gap-5 text-sm text-white/75 sm:flex">
          {NAV.map((item) => (
            <a key={item.href} href={item.href} className="hover:text-white">
              {item.label}
            </a>
          ))}
        </nav>
        <a
          href="#papers"
          className="inline-flex items-center gap-1 text-xs text-white/80 hover:text-white sm:text-sm"
        >
          读论文卡片
          <ArrowUpRight className="size-3.5" />
        </a>
      </div>
    </header>
  );
}
