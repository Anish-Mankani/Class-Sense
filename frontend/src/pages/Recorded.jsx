import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { API } from "../hooks/useClassSense";

const clock = (s) => {
  s = Math.round(s || 0);
  return `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const TYPE = {
  "looking away": { label: "Looking away", chip: "bg-amber-100 text-amber-700" },
  "head down": { label: "Head down", chip: "bg-red-100 text-red-700" },
  phone: { label: "Phone", chip: "bg-orange-100 text-orange-700" },
};
const FILTERS = [["all", "All"], ["looking away", "Looking away"], ["head down", "Head down"], ["phone", "Phone"]];

const Card = ({ className = "", children }) => (
  <div className={`rounded-3xl bg-white p-5 shadow-[0_8px_30px_rgba(40,60,140,0.07)] ${className}`}>{children}</div>
);
const Stat = ({ value, label }) => (
  <Card><div className="text-3xl font-bold leading-tight">{value}</div><div className="text-slate-500">{label}</div></Card>
);

export default function Recorded() {
  const [list, setList] = useState(null);
  const [error, setError] = useState(false);
  const [name, setName] = useState(null);
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState("all");
  const [limit, setLimit] = useState(24);

  useEffect(() => {
    fetch(`${API}/api/analyses`).then((r) => r.json())
      .then((l) => { setList(l); if (l.length) setName(l[0].name); })
      .catch(() => setError(true));
  }, []);
  useEffect(() => {
    if (!name) return;
    setData(null); setLimit(24); setFilter("all");
    fetch(`${API}/api/analyses/${encodeURIComponent(name)}`).then((r) => r.json()).then(setData).catch(() => setError(true));
  }, [name]);

  const events = data ? data.events.filter((e) => filter === "all" || e.type === filter) : [];
  const students = data
    ? Object.entries(data.students).map(([id, s]) => ({ id, ...s, off: s["looking away"] + s["head down"] }))
        .sort((a, b) => b.off - a.off).slice(0, 15)
    : [];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold">Recorded Video</h1>
          <p className="mt-1 text-lg text-slate-500">Review a lesson after it has been analysed. Cues are advisory only.</p>
        </div>
        {list && list.length > 0 && (
          <select value={name || ""} onChange={(e) => setName(e.target.value)} aria-label="Choose a recording"
            className="rounded-full bg-white px-5 py-2.5 shadow-sm">
            {list.map((a) => <option key={a.name} value={a.name}>{a.video}</option>)}
          </select>
        )}
      </div>

      {error && <Card>Can’t reach the server on port 8000. Start <code>python3 server.py</code> and reload this page.</Card>}
      {!error && list && list.length === 0 && (
        <Card>
          <p className="font-semibold">No analysed videos yet.</p>
          <p className="mt-1 text-slate-500">Run this in the ClassSense folder, then reload:</p>
          <code className="mt-2 block rounded-xl bg-slate-100 p-3 text-sm">python3 analyze_video.py /path/to/lesson.mp4 --limit-minutes 5</code>
        </Card>
      )}
      {!error && name && !data && <Card>Loading…</Card>}

      {data && (
        <>
          {!data.finished && <Card className="!bg-amber-50 text-amber-800">Partial result: analysed up to {clock(data.processed_until)} of {clock(data.duration_seconds)}.</Card>}
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Stat value={clock(data.duration_seconds)} label="Recording length" />
            <Stat value={data.summary.average_engagement == null ? "–" : `${Math.round(data.summary.average_engagement)}%`} label="Average engagement" />
            <Stat value={data.summary.events} label="Events flagged" />
            <Stat value={data.summary.phone_events} label="Phone events" />
          </div>

          <Card>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-xl font-semibold">Engagement over the lesson</h2>
              <div className="flex gap-4 text-sm text-slate-600">
                <span className="flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-emerald-500" />Engaged</span>
                <span className="flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-orange-400" />Distracted</span>
              </div>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.timeline} margin={{ top: 10, right: 10, left: -12, bottom: 0 }}>
                  <defs>
                    <linearGradient id="rE" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#19B96A" stopOpacity={0.22} /><stop offset="100%" stopColor="#19B96A" stopOpacity={0} /></linearGradient>
                    <linearGradient id="rD" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#FF9F43" stopOpacity={0.22} /><stop offset="100%" stopColor="#FF9F43" stopOpacity={0} /></linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} stroke="#E8ECF8" />
                  <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={clock} tickLine={false} axisLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                  <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                  <Tooltip labelFormatter={clock} formatter={(v) => `${v}%`} />
                  <Area type="monotone" dataKey="engaged" name="Engaged" stroke="#19B96A" strokeWidth={2.5} fill="url(#rE)" dot={false} connectNulls isAnimationActive={false} />
                  <Area type="monotone" dataKey="distracted" name="Distracted" stroke="#FF9F43" strokeWidth={2.5} fill="url(#rD)" dot={false} connectNulls isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <div className="grid gap-5 xl:grid-cols-[1.5fr_1fr]">
            <Card>
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-xl font-semibold">Events</h2>
                <div className="flex flex-wrap gap-2" role="group" aria-label="Filter events">
                  {FILTERS.map(([key, label]) => (
                    <button key={key} onClick={() => { setFilter(key); setLimit(24); }} aria-pressed={filter === key}
                      className={`rounded-full px-3 py-1.5 text-sm font-medium ${filter === key ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-600"}`}>{label}</button>
                  ))}
                </div>
              </div>
              {events.length === 0 && <p className="py-6 text-center text-slate-500">No events of this kind.</p>}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {events.slice(0, limit).map((e) => (
                  <a key={e.snapshot} href={`${API}/analysis-files/${name}/${e.snapshot}`} target="_blank" rel="noreferrer"
                    className="overflow-hidden rounded-2xl border border-slate-100 bg-slate-50">
                    <img loading="lazy" src={`${API}/analysis-files/${name}/${e.snapshot}`} alt={`${TYPE[e.type].label} at ${e.clock}`} className="aspect-video w-full object-cover" />
                    <div className="flex items-center justify-between p-2.5 text-sm">
                      <span className={`rounded-full px-2.5 py-0.5 font-medium ${TYPE[e.type].chip}`}>{TYPE[e.type].label}</span>
                      <span className="text-slate-500">{e.clock} • S{e.student}</span>
                    </div>
                  </a>
                ))}
              </div>
              {events.length > limit && (
                <button onClick={() => setLimit(limit + 24)} className="mt-4 w-full rounded-full bg-slate-100 py-2 font-medium text-slate-600">Show more</button>
              )}
            </Card>

            <Card>
              <h2 className="mb-3 text-xl font-semibold">Students needing a look</h2>
              <table className="w-full text-sm">
                <thead className="text-left text-slate-500"><tr><th className="pb-2">Student</th><th>Away</th><th>Down</th><th>Phone</th></tr></thead>
                <tbody>
                  {students.map((s) => (
                    <tr key={s.id} className="border-t border-slate-100">
                      <td className="py-2 font-medium">Student {s.id}</td>
                      <td>{clock(s["looking away"]).replace(/^0:/, "")}</td>
                      <td>{clock(s["head down"]).replace(/^0:/, "")}</td>
                      <td>{s.phone_events}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-3 text-xs text-slate-500">Times are minutes:seconds. They count only after a cue has lasted the hold time, so they slightly undercount.</p>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
