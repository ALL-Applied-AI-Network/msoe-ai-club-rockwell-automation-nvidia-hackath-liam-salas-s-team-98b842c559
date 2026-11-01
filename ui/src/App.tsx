import { LiveHmi } from "./components/LiveHmi";
import { HttpRunnerClient } from "./runner/client";

const runnerClient = new HttpRunnerClient(import.meta.env.VITE_RUNNER_BASE_URL);

export function App() {
  return <LiveHmi client={runnerClient} />;
}
