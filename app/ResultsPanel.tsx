"use client";

import { Activity } from "lucide-react";

export type SessionResult = {
  reps: number;
  good: number; // reps with no form warning
};

const ar = (n: number) => n.toLocaleString("ar-SA");

// Shown after the 30-second test: total reps, correct and wrong
export function ResultsCard({ result }: { result: SessionResult }) {
  const wrong = Math.max(0, result.reps - result.good);
  const tiles = [
    { label: "مجموع التكرارات", value: result.reps, tone: "text-brand-deep", box: "bg-secondary" },
    { label: "تكرارات صحيحة", value: result.good, tone: "text-primary", box: "bg-secondary" },
    { label: "تكرارات تحتاج تحسين", value: wrong, tone: "text-amber-deep", box: "bg-warmsoft" },
  ];

  return (
    <div className="rounded-3xl border border-surface-edge bg-card/80 p-6 shadow-frost sm:p-8" aria-live="polite">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-2xl font-extrabold">نتيجة التمرين</h3>
        <Activity className="size-7 text-primary" />
      </div>
      <div className="mt-5 grid grid-cols-3 gap-3">
        {tiles.map((t) => (
          <div key={t.label} className={`rounded-2xl ${t.box} px-3 py-5 text-center`}>
            <p className={`font-num text-5xl font-semibold leading-none sm:text-6xl ${t.tone}`}>{ar(t.value)}</p>
            <p className="mt-3 text-base font-semibold text-muted-foreground sm:text-lg">{t.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
