import { afterEach, describe, expect, it } from "vitest";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const temporaryDirectories: string[] = [];
const repositoryRoot = resolve(process.cwd(), "..");

function temporaryDirectory(): string {
  const directory = mkdtempSync(resolve(tmpdir(), "fluidmix-generator-test-"));
  temporaryDirectories.push(directory);
  return directory;
}

function score(successCount: number) {
  return {
    scope: "limited_pre_access_all_abstention_no_accuracy_or_winner_claim",
    denominators: { combined: 16, ordered: 8, shuffled: 8 },
    abstention_correctness: { denominator: 16, success_count: successCount },
    attempts: Array.from({ length: 16 }, (_, index) => ({
      variant: index < 8 ? "ordered" : "shuffled"
    }))
  };
}

function runPython(source: string, args: string[] = []) {
  return spawnSync("python3", ["-c", source, ...args], {
    cwd: repositoryRoot,
    encoding: "utf8"
  });
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("mock-run generator contracts", () => {
  it("parses all four WP6 score files and fails closed on source-count drift", () => {
    const directory = temporaryDirectory();
    for (const [filename, successes] of [
      ["template-score.json", 16],
      ["qwen-score.json", 16],
      ["anomalythink-score.json", 0],
      ["cosmos-score.json", 0]
    ] as const) {
      writeFileSync(resolve(directory, filename), JSON.stringify(score(successes)));
    }
    const source = [
      "import json, pathlib, sys",
      `sys.path.insert(0, ${JSON.stringify(repositoryRoot)})`,
      "from scripts.generate_mock_run import load_wp6_rows",
      "print(json.dumps(load_wp6_rows(pathlib.Path(sys.argv[1]))))"
    ].join(";");
    const valid = runPython(source, [directory]);
    expect(valid.status, valid.stderr).toBe(0);
    const rows = JSON.parse(valid.stdout);
    expect(rows.map((row: { method: string; passed: number }) => [row.method, row.passed])).toEqual([
      ["template", 16], ["qwen3_vl", 16], ["anomalythink_free_form", 0], ["cosmos_strict_parse", 0]
    ]);
    expect(rows.map((row: { provenance: { source_id: string } }) => row.provenance.source_id)).toEqual([
      expect.stringMatching(/^template-score\.json#sha256=[0-9a-f]{64}$/),
      expect.stringMatching(/^qwen-score\.json#sha256=[0-9a-f]{64}$/),
      expect.stringMatching(/^anomalythink-score\.json#sha256=[0-9a-f]{64}$/),
      expect.stringMatching(/^cosmos-score\.json#sha256=[0-9a-f]{64}$/)
    ]);

    const drifted = score(0);
    drifted.attempts[15].variant = "ordered";
    writeFileSync(resolve(directory, "cosmos-score.json"), JSON.stringify(drifted));
    const invalid = runPython(source, [directory]);
    expect(invalid.status).not.toBe(0);
    expect(invalid.stderr).toContain("8 ordered and 8 shuffled");
  });

  it("parses ffmpeg Duration output instead of assuming cue lengths", () => {
    const directory = temporaryDirectory();
    const fakeFfmpeg = resolve(directory, "fake-ffmpeg");
    const media = resolve(directory, "section.mp4");
    writeFileSync(fakeFfmpeg, "#!/bin/sh\necho 'Duration: 00:00:03.25, start: 0.000000' >&2\nexit 0\n");
    writeFileSync(media, "fixture");
    chmodSync(fakeFfmpeg, 0o755);
    const source = [
      "import pathlib, sys",
      `sys.path.insert(0, ${JSON.stringify(repositoryRoot)})`,
      "from scripts.generate_mock_run import probe_duration",
      "print(probe_duration(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])))"
    ].join(";");
    const result = runPython(source, [fakeFfmpeg, media]);
    expect(result.status, result.stderr).toBe(0);
    expect(Number(result.stdout.trim())).toBe(3.25);
  });
});
