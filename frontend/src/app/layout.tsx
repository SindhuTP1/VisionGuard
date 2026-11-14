import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { HealthBadge } from "@/components/HealthBadge";
import { NavLinks } from "@/components/NavLinks";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "VisionGuard",
  description: "Multi-layer AI image hallucination detector (research prototype).",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <header className="border-b border-line bg-surface-1">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
            <div className="flex items-center gap-2">
              <ShieldIcon />
              <span className="text-[15px] font-semibold tracking-tight">VisionGuard</span>
              <span className="rounded border border-line px-1.5 py-0.5 text-[11px] text-ink-3">research prototype</span>
            </div>
            <NavLinks />
            <div className="ml-auto">
              <HealthBadge />
            </div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">{children}</main>
        <footer className="border-t border-line py-4 text-center text-xs text-ink-3">
          VisionGuard is a research prototype. Its verdicts are signals for a human reviewer, not proof.
        </footer>
      </body>
    </html>
  );
}

function ShieldIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
      <path d="M9 12l2 2 4-4" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
