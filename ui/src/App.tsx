import pipelinePin from "../config/pipeline-pin.json";
import { LiveHmi } from "./components/LiveHmi";
import { HttpRunnerClient } from "./runner/client";

const expectedPipelineCommit = pipelinePin.expected_pipeline_commit;
const runnerClient = new HttpRunnerClient(import.meta.env.VITE_RUNNER_BASE_URL, expectedPipelineCommit);

export function App() {
  return <LiveHmi client={runnerClient} expectedPipelineCommit={expectedPipelineCommit} />;
}
