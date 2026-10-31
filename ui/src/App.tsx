import { useEffect, useState } from "react";
import { OperatorDashboard } from "./components/OperatorDashboard";
import { parseDemoData, type DemoData } from "./data/adapter";

export function App() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/active-run.json", { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }).then(input => setData(parseDemoData(input))).catch(reason => {
      if (reason?.name !== "AbortError") setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => controller.abort();
  }, []);
  if (error) return <main className="load-state"><div role="alert"><h1>Could not load operator run</h1><p>Refresh the page and confirm <code>/active-run.json</code> is available.</p><small>{error}</small></div></main>;
  if (!data) return <main className="load-state"><div role="status">Loading run evidence…</div></main>;
  return <OperatorDashboard data={data} />;
}
