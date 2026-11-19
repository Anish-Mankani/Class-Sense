import { useState } from "react";
import { BarChart3, FileText, Film, GraduationCap, Home, MonitorPlay, Settings } from "lucide-react";
import Dashboard from "./pages/Dashboard.jsx";
import LiveClass from "./pages/LiveClass.jsx";
import Recorded from "./pages/Recorded.jsx";
import { useClassSense } from "./hooks/useClassSense";

const NAV = [["Dashboard", Home], ["Live Class", MonitorPlay], ["Recorded Video", Film], ["Analytics", BarChart3], ["Reports", FileText], ["Settings", Settings]];

export default function App() {
  const [page, setPage] = useState("Dashboard");
  const data = useClassSense();
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col bg-gradient-to-b from-[#1B2559] to-[#141C47] p-5 text-white md:flex">
        <div className="mb-8 mt-4 text-center">
          <GraduationCap className="mx-auto text-blue-400" size={44} />
          <div className="mt-2 text-2xl font-bold">ClassSense</div>
          <div className="text-sm text-indigo-200">Smarter Classrooms</div>
        </div>
        <nav className="space-y-1">
          {NAV.map(([name, Icon]) => (
            <button key={name} onClick={() => setPage(name)}
              className={`flex w-full items-center gap-4 rounded-xl px-4 py-3 text-left font-medium ${page === name ? "bg-blue-600" : "text-indigo-100 hover:bg-white/10"}`}>
              <Icon size={22} />{name}
            </button>
          ))}
        </nav>
        <div className="mt-auto flex items-center gap-3 rounded-full bg-white/10 px-4 py-2.5 text-sm">
          <span className={`h-3 w-3 rounded-full ${data.live ? "bg-emerald-400" : "bg-amber-400"}`} />{data.live ? "Online" : "Sample data"}
        </div>
      </aside>
      <main className="min-w-0 flex-1 p-5 md:p-8">
        {page === "Recorded Video" ? <Recorded /> : page === "Live Class" ? <LiveClass data={data} /> : page === "Dashboard"
          ? <Dashboard data={data} />
          : <div className="rounded-3xl bg-white p-10 text-center text-slate-500 shadow-sm"><h1 className="mb-2 text-2xl font-bold text-ink">{page}</h1>This page is next. The Dashboard is the finished screen for now.</div>}
      </main>
    </div>
  );
}
