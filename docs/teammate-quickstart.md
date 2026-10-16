# Independent teammate quickstart receipt

**Template only: no teammate run is recorded here.** INN-24 requires a teammate to follow
the [quickstart](quickstart.md) from a clean checkout and record corrections. An agent,
author, automated clone or independent code review does not substitute for that trial.

Copy this template into the teammate's run record. Retain failed commands and actual
outputs. Do not include credentials, private host details or raw data. A local record
does not require posting a message; attach it to the issue only within authorization.

| Field | Teammate records |
|---|---|
| Teammate and UTC date/time | NOT RUN |
| Repository URL and exact commit (`git rev-parse HEAD`) | NOT RUN |
| Clean source status before setup (`git status --short`) | NOT RUN |
| OS/architecture and `python3 --version` | NOT RUN |
| Environment creation, activation and dependency install commands/results | NOT RUN |
| `python -m pip freeze` (package versions only) | NOT RUN |
| `python -m pytest tests -q` output and exit status | NOT RUN |
| `python scripts/check_agent_docs.py` output and exit status | NOT RUN |
| `python scripts/smoke_absorbance.py` output and exit status | NOT RUN |
| `python scripts/fit_absorbance_video5.py` output and warning interpretation | NOT RUN |
| Fresh host/version, startup skill catalog and selected workflow (if tested) | NOT RUN |
| Steps that failed or needed undocumented knowledge | NOT RUN |
| Exact corrections, files/lines and recheck results | NOT RUN |
| Final status and remaining limits | NOT RUN |

The synthetic smoke should recover fraction 0.150000 with residual below 1e-6. The real
CSV diagnostic is expected to warn about negative absorbance; a successful exit does not
make that calibration physically valid. Optional USD/render tools are outside the CPU
setup. Record them as untested if unused. Record any host not tested honestly.
