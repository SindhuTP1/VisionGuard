import type { Metadata } from "next";
import { MetricsView } from "@/components/MetricsView";

export const metadata: Metadata = { title: "Model metrics · VisionGuard" };

export default function MetricsPage() {
  return <MetricsView />;
}
