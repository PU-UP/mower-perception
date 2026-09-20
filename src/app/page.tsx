import { Playground } from "@/components/playground"

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <Playground />
      <footer className="border-t border-border px-4 py-6 text-center text-xs leading-6 text-muted-foreground">
        当前是源数据集零样本映射（ADE20K / PASCAL VOC），不是割草机域内微调结果。换模型或换数据后保持同一套推理接口。
      </footer>
    </div>
  )
}
