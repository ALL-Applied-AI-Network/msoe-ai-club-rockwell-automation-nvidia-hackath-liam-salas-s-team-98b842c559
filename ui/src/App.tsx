import pipelinePin from "../config/pipeline-pin.json";
import { AppShell } from "./app/AppShell";
import { HttpRunnerClient } from "./runner/client";

const expectedPipelineCommit = pipelinePin.expected_pipeline_commit;
/**
 * Development-only escape hatch: comma-separated full commits that may start sessions despite the
 * pin. The shell shows an "Unpinned development service" banner whenever it is used.
 */
const devAllowedCommits = (import.meta.env.DEV ? import.meta.env.VITE_DEV_ALLOWED_COMMITS ?? "" : "").split(",").map((c: string) => c.trim()).filter((c: string) => /^[0-9a-f]{40}$/.test(c));
const runnerClient = new HttpRunnerClient(import.meta.env.VITE_RUNNER_BASE_URL, [expectedPipelineCommit, ...devAllowedCommits]);

export function App() {
  return <AppShell client={runnerClient} expectedCommit={expectedPipelineCommit} devAllowedCommits={devAllowedCommits} />;
}
