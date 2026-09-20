import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MowerSeg · 割草机可通行分割",
  description:
    "面向智能割草机的轻量语义分割起步框架：SegFormer-B0 零样本推理，映射到草坪产品类别，可输出叠加图和统计。",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="zh-CN"
      className="h-full antialiased"
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
