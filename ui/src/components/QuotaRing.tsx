import { formatRelativeTime } from '../api';
import type { QuotaWindow } from '../types';

export function QuotaRing({ window }: { window: QuotaWindow }) {
  const pct = Math.max(0, Math.min(100, Math.round(window.used_percent)));
  const radius = 16;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (pct / 100) * circumference;

  const strokeColor =
    pct >= 90 ? '#f43f5e' : pct >= 70 ? '#f59e0b' : '#10b981';

  return (
    <div className="flex items-center gap-2.5 rounded-lg bg-slate-900/70 px-2.5 py-1.5 border border-white/5">
      <div className="relative h-10 w-10 shrink-0">
        <svg className="h-10 w-10 -rotate-90" viewBox="0 0 40 40">
          <circle
            cx="20"
            cy="20"
            r={radius}
            fill="none"
            stroke="rgba(148,163,184,0.16)"
            strokeWidth="3.5"
          />
          <circle
            cx="20"
            cy="20"
            r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center font-mono text-[10px] font-semibold text-slate-200 tabular-nums">
          {pct}%
        </span>
      </div>
      <div className="min-w-0">
        <div className="text-xs font-medium text-slate-200 truncate">
          {window.label}
        </div>
        <div className="text-[11px] text-slate-400">
          {window.reset_at_ms
            ? `resets ${formatRelativeTime(window.reset_at_ms)}`
            : `${100 - pct}% free`}
        </div>
      </div>
    </div>
  );
}
