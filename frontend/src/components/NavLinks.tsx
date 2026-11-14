"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Analyze" },
  { href: "/metrics", label: "Model metrics" },
] as const;

export function NavLinks() {
  const path = usePathname();
  return (
    <nav className="flex gap-1 text-sm">
      {LINKS.map(({ href, label }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={`rounded-md px-3 py-1.5 transition-colors ${
              active ? "bg-surface-2 font-medium text-ink-1" : "text-ink-2 hover:bg-surface-2 hover:text-ink-1"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
