import { Playground } from "@/components/playground"

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <Playground />
      <footer className="border-t border-border px-4 py-6 text-center text-xs leading-6 text-muted-foreground">
        YCOR 模型已完成可通行草地代理训练，真实可割适配尚未完成。草地分割不等于完整的人、动物或障碍物安全检测。
      </footer>
    </div>
  )
}
