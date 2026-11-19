import { useEffect, useState } from "react";
import { Users, Smartphone, ShieldCheck, Check, Info, ChevronDown, ChevronRight, Maximize2, Lightbulb, ScanFace, ScanSearch, UsersRound, SmartphoneNfc } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { API, summarize } from "../hooks/useClassSense";

const Card = ({ className = "", children }) => (
  <div className={`rounded-3xl bg-white p-5 shadow-[0_8px_30px_rgba(40,60,140,0.07)] ${className}`}>{children}</div>
);

function Ring({ value }) {
  const r = 30, c = 2 * Math.PI * r;
  return (
    <svg width="76" height="76" viewBox="0 0 76 76" className="shrink-0 -rotate-90">
      <circle cx="38" cy="38" r={r} fill="none" stroke="#E4F6EC" strokeWidth="9" />
      <circle cx="38" cy="38" r={r} fill="none" stroke="#19B96A" strokeWidth="9" strokeLinecap="round"
        strokeDasharray={c} strokeDashoffset={c * (1 - value / 100)} />
    </svg>
  );
}

const Stat = ({ icon, value, label }) => (
  <Card className="flex items-center gap-4">
    {icon}
    <div><div className="text-3xl font-bold leading-tight">{value}</div><div className="text-slate-500">{label}</div></div>
  </Card>
);

export function Video({ live }) {
  const [failed, setFailed] = useState(false);
  const [secs, setSecs] = useState(0);
  useEffect(() => { const id = setInterval(() => setSecs((s) => s + 1), 1000); return () => clearInterval(id); }, []);
  const t = [Math.floor(secs / 3600), Math.floor(secs / 60) % 60, secs % 60].map((n) => String(n).padStart(2, "0")).join(":");
  return (
    <div className="relative h-72 overflow-hidden rounded-3xl bg-gradient-to-br from-[#2a3a7a] to-[#16205B] shadow-[0_8px_30px_rgba(40,60,140,0.12)] lg:h-full lg:min-h-80">
      {live && !failed
        ? <img src={`${API}/video`} alt="Live classroom" className="h-full w-full object-cover" onError={() => setFailed(true)} />
        : <div className="grid h-full place-items-center px-8 text-center text-sm text-indigo-100">
            {live ? "Connected. The video stream will appear once the server provides /video." : "Camera preview appears here when the server is running."}
          </div>}
      <span className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-emerald-500 px-3 py-1 text-sm font-semibold text-white">
        <span className="h-2 w-2 rounded-full bg-white" />{live ? "Live" : "Sample"}
      </span>
      <span className="absolute bottom-4 left-4 rounded-xl bg-black/45 px-4 py-2 text-sm text-white backdrop-blur">Classroom Monitor • {t}</span>
      <button aria-label="Full screen" className="absolute bottom-4 right-4 grid h-10 w-10 place-items-center rounded-xl bg-black/45 text-white"
        onClick={(e) => e.currentTarget.parentElement.requestFullscreen?.()}><Maximize2 size={18} /></button>
    </div>
  );
}

const alertIcon = {
  phone: <span className="grid h-10 w-10 place-items-center rounded-xl bg-orange-100 text-orange-500"><Smartphone size={20} /></span>,
  info: <span className="grid h-10 w-10 place-items-center rounded-full bg-blue-100 text-blue-600"><Info size={20} /></span>,
  ok: <span className="grid h-10 w-10 place-items-center rounded-full bg-emerald-100 text-emerald-600"><Check size={20} /></span>,
};

function Mini({ icon, title, ok, text, sub }) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-3 font-semibold"><span className="grid h-9 w-9 place-items-center rounded-full bg-indigo-50 text-indigo-600">{icon}</span>{title}</div>
      <div className="flex items-center gap-3">
        <span className={`grid h-9 w-9 shrink-0 place-items-center rounded-full text-white ${ok ? "bg-emerald-500" : "bg-orange-500"}`}>{ok ? <Check size={18} /> : <Info size={18} />}</span>
        <div className="font-semibold leading-snug">{text}<div className="font-normal text-slate-500">{sub}</div></div>
      </div>
    </Card>
  );
}

export default function Dashboard({ data }) {
  const { faces, history, alerts, live } = data;
  const s = summarize(faces);
  const unmatched = s.total - s.verified;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold">Good Morning, Teacher!</h1>
          <p className="mt-1 text-lg text-slate-500">Here’s what’s happening in your classroom.</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-3 rounded-full bg-white px-5 py-2.5 shadow-sm">Class 10-A <ChevronDown size={18} /></button>
          <span className="grid h-11 w-11 place-items-center rounded-full bg-indigo-200 font-bold text-indigo-800">T</span>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat icon={<span className="grid h-16 w-16 place-items-center rounded-2xl text-blue-600"><Users size={44} /></span>} value={s.total} label="Students" />
        <Stat icon={<Ring value={s.engaged} />} value={`${s.engaged}%`} label="Engaged" />
        <Stat icon={<span className="grid h-16 w-16 shrink-0 place-items-center rounded-full bg-violet-100 text-violet-600"><Smartphone size={30} /></span>} value={s.phones} label="Phones Detected" />
        <Stat icon={<span className="grid h-16 w-16 shrink-0 place-items-center rounded-full bg-emerald-100 text-emerald-500"><ShieldCheck size={36} /></span>} value={`${s.verified} / ${s.total}`} label="Verified" />
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.1fr_1fr]">
        <Video live={live} />
        <Card>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-xl font-semibold">Engagement Trend</h2>
            <div className="flex gap-4 text-sm text-slate-600">
              <span className="flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-emerald-500" />Engaged</span>
              <span className="flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-orange-400" />Distracted</span>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history} margin={{ top: 10, right: 10, left: -12, bottom: 0 }}>
                <defs>
                  <linearGradient id="gE" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#19B96A" stopOpacity={0.22} /><stop offset="100%" stopColor="#19B96A" stopOpacity={0} /></linearGradient>
                  <linearGradient id="gD" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#FF9F43" stopOpacity={0.22} /><stop offset="100%" stopColor="#FF9F43" stopOpacity={0} /></linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#E8ECF8" />
                <XAxis dataKey="t" tickLine={false} axisLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Area type="monotone" dataKey="engaged" name="Engaged" stroke="#19B96A" strokeWidth={3} fill="url(#gE)" dot={{ r: 4, fill: "#19B96A", strokeWidth: 0 }} isAnimationActive={false} />
                <Area type="monotone" dataKey="distracted" name="Distracted" stroke="#FF9F43" strokeWidth={3} fill="url(#gD)" dot={{ r: 4, fill: "#FF9F43", strokeWidth: 0 }} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.7fr_1fr]">
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-3">
            <Mini icon={<ScanFace size={20} />} title="Head Pose" ok={s.engaged >= 70}
              text={s.engaged >= 70 ? "Facing Forward" : "Some Looking Away"} sub={s.engaged >= 70 ? "(Attentive)" : `(${s.distracted}% distracted)`} />
            <Mini icon={<SmartphoneNfc size={20} />} title="Activity" ok={s.phones === 0}
              text={s.phones === 0 ? "No Mobile Phone Detected" : `${s.phones} Possible Phone${s.phones > 1 ? "s" : ""}`} sub={s.phones === 0 ? "" : "Please review"} />
            <Mini icon={<UsersRound size={20} />} title="Student Check" ok={unmatched === 0}
              text={unmatched === 0 ? "All Registered" : `${unmatched} Not On Roster`} sub={unmatched === 0 ? "Students Present" : "Confirm in person"} />
          </div>
          <div className="flex items-center gap-4 rounded-2xl bg-indigo-100/70 px-5 py-4 text-indigo-900">
            <Lightbulb className="text-amber-500" /><span className="flex-1">Tip: Keep encouraging participation for better engagement!</span><ChevronRight size={18} />
          </div>
        </div>
        <Card>
          <div className="mb-3 flex items-center justify-between"><h2 className="text-xl font-semibold">Recent Alerts</h2><a className="text-sm font-medium text-blue-600" href="#all">View All →</a></div>
          <ul className="divide-y divide-slate-100">
            {alerts.map((a, i) => (
              <li key={i} className="flex items-center gap-3 py-3">
                {alertIcon[a.kind]}
                <div className="flex-1 font-medium leading-snug">{a.title}<div className="font-normal text-slate-500">{a.sub}</div></div>
                <span className="text-sm text-slate-400">{a.time}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
