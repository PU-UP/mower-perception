import type { NextConfig } from "next";

const apiOrigin = process.env.MOWERSEG_API ?? "http://127.0.0.1:43131";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${apiOrigin}/api/:path*` },
      { source: "/samples/:path*", destination: `${apiOrigin}/samples/:path*` },
    ];
  },
};

export default nextConfig;
