import { useRef, useState } from "react";
import { Maximize2, ShieldAlert, Smartphone } from "lucide-react";
import { Video } from "./Dashboard.jsx";

const STYLE = {
  attentive: { label: "Attentive", dot: "bg-emerald-500", tile: "border-emerald-200 bg-emerald-50", text: "text-emerald-700" },
  writing: { label: "Writing / reading", dot: "bg-sky-500", tile: "border-sky-200 bg-sky-50", text: "text-sky-700" },
  "looking away": { label: "Looking away", dot: "bg-amber-500", tile: "border-amber-200 bg-amber-50", text: "text-amber-700" },
  "head down": { label: "Head down", dot: "bg-red-500", tile: "border-red-200 bg-red-50", text: "text-red-700" },
  unknown: { label: "Too far to read", dot: "bg-slate-400", tile: "border-slate-200 bg-slate-50", text: "text-slate-600" },
};
const FILTERS = [["all", "All"], ["attention", "Needs attention"], ["phones", "Phones"]];
const initials = (name) => name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
const deg = (v) => (v == null ? "Not measured" : `${Math.round(v)}°`);

export default function LiveClass({ data }) {
  const { faces, live } = data;
  const [filter, setFilter] = useState("all");
  const [selId, setSelId] = useState(null);
  const root = useRef(null);
  const counts = faces.reduce((c, f) => ({ ...c, [f.state]: (c[f.state] || 0) + 1 }), {});
  const shown = faces.filter((f) =>
    filter === "all" ? true : filter === "phones" ? f.phone : f.state === "looking away" || f.state === "head down");
  const sel = faces.find((f) => f.id === selId);

  return (
    <div ref={root} className="space-y-5 bg-page">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold">Live Class</h1>
          <p className="mt-1 text-lg text-slate-500">Class 10-A • {faces.length} students in view</p>
        </div>
        <button onClick={() => root.current?.requestFullscreen?.()}
          className="flex items-center gap-2 rounded-full bg-blue-600 px-5 py-2.5 font-medium text-white">
          <Maximize2 size={18} />Full screen
        </button>
      </div>

      <div className="flex flex-wrap gap-3">
        {Object.entries(STYLE).map(([key, st]) => (
          <span key={key} className="flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm shadow-sm">
            <i className={`h-3 w-3 rounded-full ${st.dot}`} />{st.label}<b>{counts[key] || 0}</b>
          </span>
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.4fr_1fr]">
        <div className="lg:h-[560px]"><Video live={live} /></div>

        <div className="rounded-3xl bg-white p-5 shadow-[0_8px_30px_rgba(40,60,140,0.07)]">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-xl font-semibold">Students</h2>
            <div className="flex gap-2" role="group" aria-label="Filter students">
              {FILTERS.map(([key, label]) => (
                <button key={key} onClick={() => setFilter(key)} aria-pressed={filter === key}
                  className={`rounded-full px-3 py-1.5 text-sm font-medium ${filter === key ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-600"}`}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid max-h-[360px] gap-2.5 overflow-y-auto pr-1 sm:grid-cols-2">
            {shown.map((f) => {
              const st = STYLE[f.state] || STYLE.unknown;
              return (
                <button key={f.id} onClick={() => setSelId(f.id)} aria-pressed={f.id === selId}
                  className={`flex items-center gap-3 rounded-2xl border p-3 text-left ${st.tile} ${f.id === selId ? "ring-2 ring-blue-600" : ""}`}>
                  <span className="relative grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white font-bold">
                    {initials(f.name)}
                    <i className={`absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full border-2 border-white ${st.dot}`} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-semibold">{f.name}</span>
                    <span className={`text-sm font-medium ${st.text}`}>{st.label}</span>
                  </span>
                  {f.phone && <Smartphone size={18} className="text-orange-500" aria-label="Possible phone" />}
                  {f.roster === "unmatched" && <ShieldAlert size={18} className="text-red-500" aria-label="Not on roster" />}
                </button>
              );
            })}
            {shown.length === 0 && <p className="col-span-full py-6 text-center text-slate-500">No students match this filter.</p>}
          </div>

          <div className="mt-4 rounded-2xl bg-slate-50 p-4">
            {sel ? (
              <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-sm">
                <dt className="text-slate-500">Student</dt><dd className="font-semibold">{sel.name}</dd>
                <dt className="text-slate-500">Cue</dt><dd className="font-semibold">{(STYLE[sel.state] || STYLE.unknown).label}</dd>
                <dt className="text-slate-500">Head turn</dt><dd className="font-semibold">{deg(sel.yaw)}</dd>
                <dt className="text-slate-500">Head tilt</dt><dd className="font-semibold">{deg(sel.pitch)}</dd>
                <dt className="text-slate-500">Phone</dt><dd className="font-semibold">{sel.phone ? "Possible phone seen" : "None seen"}</dd>
                <dt className="text-slate-500">Roster</dt><dd className="font-semibold">{sel.roster === "verified" ? "Matched to this class" : "Not matched. Confirm in person."}</dd>
              </dl>
            ) : <p className="text-sm text-slate-500">Select a student to see details. Cues are advisory only.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
