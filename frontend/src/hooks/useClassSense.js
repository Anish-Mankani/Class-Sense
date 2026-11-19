import { useCallback, useEffect, useRef, useState } from "react";

const HOST = window.location.hostname || "localhost";
export const API = `http://${HOST}:8000`;
const WS_URL = `ws://${HOST}:8000/ws`;
const now = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

export function summarize(faces) {
  const usable = faces.filter((f) => f.state !== "unknown");
  const n = usable.length || 1;
  const pct = (fn) => Math.round((100 * usable.filter(fn).length) / n);
  return {
    total: faces.length,
    engaged: pct((f) => f.state === "attentive" || f.state === "writing"),
    distracted: pct((f) => f.state === "looking away" || f.state === "head down"),
    phones: faces.filter((f) => f.phone).length,
    verified: faces.filter((f) => f.roster === "verified").length,
  };
}

// Sample students so the design works before the server is running.
function simStep(prev) {
  return Array.from({ length: 24 }, (_, i) => {
    const p = prev[i] || { state: "attentive", phone: false };
    const r = Math.random();
    let state = p.state;
    if (state === "attentive") state = r < 0.02 ? "looking away" : r < 0.03 ? "head down" : "attentive";
    else if (r < 0.15) state = "attentive";
    const phone = p.phone ? Math.random() > 0.1 : i < 23 && Math.random() < 0.003;
    return { id: i + 1, name: `Student ${i + 1}`, state, phone, roster: i === 23 ? "unmatched" : "verified",
      yaw: state === "looking away" ? 35 : 2, pitch: state === "head down" ? 20 : 0 };
  });
}

const seedHistory = () => Array.from({ length: 12 }, (_, i) => ({
  t: new Date(Date.now() - (11 - i) * 300000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
  engaged: Math.round(84 + Math.random() * 12), distracted: Math.round(10 + Math.random() * 8) }));

const seedAlerts = () => [
  { kind: "phone", title: "Mobile Phone Detected", sub: "in Row 3", time: "9:32 AM" },
  { kind: "info", title: "Engagement Improved", sub: "(+12%)", time: "9:28 AM" },
  { kind: "ok", title: "Class Roster Match", sub: "(23/24)", time: "9:25 AM" },
];

export function useClassSense() {
  const [faces, setFaces] = useState(() => simStep([]));
  const [history, setHistory] = useState(seedHistory);
  const [alerts, setAlerts] = useState(seedAlerts);
  const [live, setLive] = useState(false);
  const facesRef = useRef(faces);
  const liveRef = useRef(false);

  const apply = useCallback((next) => {
    const prev = facesRef.current;
    const added = [];
    next.forEach((f) => {
      const o = prev.find((p) => p.id === f.id);
      if (f.phone && !(o && o.phone)) added.push({ kind: "phone", title: "Mobile Phone Detected", sub: f.name, time: now() });
      else if (o && o.state !== f.state && (f.state === "head down" || f.state === "looking away"))
        added.push({ kind: "info", title: `${f.name} is ${f.state}`, sub: "Advisory cue", time: now() });
    });
    facesRef.current = next;
    setFaces(next);
    if (added.length) setAlerts((a) => [...added, ...a].slice(0, 6));
  }, []);

  useEffect(() => {
    const id = setInterval(() => { if (!liveRef.current) apply(simStep(facesRef.current)); }, 1000);
    return () => clearInterval(id);
  }, [apply]);

  useEffect(() => {
    const id = setInterval(() => {
      const s = summarize(facesRef.current);
      setHistory((h) => [...h.slice(-11), { t: now(), engaged: s.engaged, distracted: s.distracted }]);
    }, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let ws, retry, closed = false;
    const connect = () => {
      try { ws = new WebSocket(WS_URL); } catch { return; }
      ws.onopen = () => { liveRef.current = true; setLive(true); apply([]); };
      ws.onmessage = (e) => apply(JSON.parse(e.data).faces);
      ws.onclose = () => { liveRef.current = false; setLive(false); if (!closed) retry = setTimeout(connect, 2000); };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); if (ws) ws.close(); };
  }, [apply]);

  return { faces, history, alerts, live };
}
