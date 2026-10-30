import { afterEach, describe, expect, it } from "vitest";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";

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
    encoding: "utf8",
    env: { ...process.env, PYTHONPATH: "/tmp/frontend-pydeps" }
  });
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("mock-run generator contracts", () => {
  it("rejects structurally compatible but unapproved WP1 and WP5 top-level inputs", () => {
    const directory = temporaryDirectory();
    const wp1Manifest = resolve(directory, "simple-production-manifest.json");
    const wp1Receipt = resolve(directory, "preexec-receipt.json");
    const wp5Measurements = resolve(directory, "measurements.json");
    const wp5Locators = resolve(directory, "locators.json");
    writeFileSync(wp1Manifest, JSON.stringify({ jobs: [] }));
    writeFileSync(wp1Receipt, JSON.stringify({
      artifact_type: "fluidmix_simple_transport_preexec_v1",
      status: "ready",
      manifest_sha256: "90418a6e9a98d456de29bcebe497f5e0db3ab2309c6942f428318534dda14a76"
    }));
    writeFileSync(wp5Measurements, JSON.stringify({
      status: "measured", data_origin: "twin", real_data_used: false
    }));
    writeFileSync(wp5Locators, JSON.stringify({ data_origin: "twin", images: {} }));
    const source = [
      "import pathlib,sys",
      `sys.path.insert(0, ${JSON.stringify(repositoryRoot)})`,
      "import scripts.generate_mock_run as g",
      "mode=sys.argv[1]",
      "root=pathlib.Path(sys.argv[2])",
      "g.manifest_fill_jobs(root) if mode=='manifest' else g.validate_wp1_batch(root) if mode=='receipt' else g.validate_wp5(root/'measurements.json',root/'locators.json')"
    ].join(";");
    for (const [mode, label] of [["manifest", "WP1 manifest input"], ["receipt", "WP1 pre-execution receipt"], ["wp5", "WP5 measurements input"]]) {
      const result = runPython(source, [mode, directory]);
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain(`${label} does not match the approved SHA-256`);
    }
  });

  it("rejects structurally valid but unapproved WP6 score files before parsing", () => {
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
    const result = runPython(source, [directory]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("approved SHA-256");
    expect(result.stdout).toBe("");
  });

  it("refuses transport states that are not bound by a pinned render receipt", () => {
    const batch = temporaryDirectory();
    const png = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64"
    );
    for (let segment = 0; segment < 10; segment += 1) {
      const mode = segment === 0 ? "entry" : segment === 9 ? "exit" : "between";
      const stateDirectory = resolve(batch, `segment-${String(segment).padStart(2, "0")}-${mode}`, "render", "states");
      mkdirSync(stateDirectory, { recursive: true });
      writeFileSync(resolve(stateDirectory, "state-00.png"), png);
    }
    const destination = resolve(batch, "composite.mp4");
    const source = [
      "import pathlib,sys",
      `sys.path.insert(0, ${JSON.stringify(repositoryRoot)})`,
      "import scripts.generate_mock_run as g",
      "g.run_ffmpeg=lambda command,label: pathlib.Path(command[-1]).write_bytes(b'mp4')",
      "g.probe_duration=lambda ffmpeg,path: 1.0",
      "g.build_media(pathlib.Path('ffmpeg'),pathlib.Path(sys.argv[1]),[pathlib.Path('fill')]*9,pathlib.Path(sys.argv[2]))"
    ].join(";");
    const result = runPython(source, [batch, destination]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("render receipt");
  });

  it("rejects same-count transport state byte drift against its immutable inventory", () => {
    const renderDirectory = temporaryDirectory();
    const states = resolve(renderDirectory, "states");
    mkdirSync(states);
    const state = resolve(states, "state-00.png");
    const original = Buffer.from("approved-state-bytes");
    writeFileSync(state, original);
    const receipt = {
      artifact_type: "fluidmix_transport_stage_render_receipt_v1",
      mode: "entry",
      state_indices: [0],
      captures: [{
        state_index: 0,
        rgb_path: "states/state-00.png",
        rgb_sha256: createHash("sha256").update(original).digest("hex")
      }]
    };
    const receiptPath = resolve(renderDirectory, "receipt.json");
    writeFileSync(receiptPath, JSON.stringify(receipt));
    const source = [
      "import json,pathlib,sys",
      `sys.path.insert(0, ${JSON.stringify(repositoryRoot)})`,
      "from scripts.generate_mock_run import validate_transport_capture_files",
      "root=pathlib.Path(sys.argv[1])",
      "receipt=json.loads(pathlib.Path(sys.argv[2]).read_text())",
      "validate_transport_capture_files(root,receipt,'entry',[0])"
    ].join(";");
    const valid = runPython(source, [renderDirectory, receiptPath]);
    expect(valid.status, valid.stderr).toBe(0);

    writeFileSync(state, Buffer.from("drifted-state-byte"));
    const drifted = runPython(source, [renderDirectory, receiptPath]);
    expect(drifted.status).not.toBe(0);
    expect(drifted.stderr).toContain("transport state states/state-00.png SHA-256 drift");
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
