import type { Metadata } from "next";
import { Analyzer } from "@/components/Analyzer";

export const metadata: Metadata = { title: "Classic analyzer · VisionGuard" };

// The original single-column analyzer, kept alongside the dashboard.
export default function ClassicPage() {
  return <Analyzer />;
}
