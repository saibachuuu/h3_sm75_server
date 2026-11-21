# Explicit API implementation notes

## Baseline and current scope

Implementation authorized on 2026-11-21. The user allows the listed agents, dedicated worktrees, phase commits, and primary-agent-only review/integration.

- Baseline: `5caf1bf`, original worktree clean at implementation start.
- Integration branch: `feat/explicit-api`, separate worktree; original TE branch preserved.
- Authoritative design: `../plan_20261121_152726/`.
- Local interpreter: Python 3.10.11; torch `2.8.0+cpu`, CUDA unavailable; `psutil` present, `safetensors` initially absent.
- Existing `_remote_probe` snapshots remain historical clues; fresh source copies and hashes were obtained read-only from the GPU environment.

### Read-only remote inventory (P0)

- Server repository: `5caf1bf`, branch `feat/te-proj-swap`.
- Existing remote untracked files preserved: `QQ20261006-160626.png`, `TEST_PROMPT.txt`, `_submit_veda_task.py`.
- ComfyUI: `5f415e31a`; Python 3.12.7; torch `2.13.0+cu132`; comfy-kitchen `0.2.40`; safetensors `0.7.0`; psutil `7.1.3`; gguf `0.17.1`.
- GPU inventory: RTX 2080 Ti, 22528 MiB total, approximately 272 MiB used at inspection, driver 610.88. This is inventory, not a GPU loading/inference test.
- Four checkpoint headers read without tensor payloads or CUDA imports: DiT 13,997,668,774 bytes; video VAE 2,811,065,184; audio VAE 605,254,808; VEDA 275,415,752.
- Full header/source-hash report: `C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_sources_20261121\remote_environment.json` (UTF-8; use utf-8-sig when reading).
- Fresh Comfy/ClipProj/minimax/attention/nested-tensor sources are in the same approved temporary directory. Source copies are read-only development references, not imported into the service.
- Local isolated test interpreter: `C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe`, inherits CPU torch, safetensors `0.8.0` installed only in that environment.
- `python tests/_selftest_environment_probe.py`: 3 tests passed. Probe reads a bounded safetensors header and never imports torch or constructs a model.

## Phase ledger

| Phase | Status | Evidence / pending gate |
|---|---|---|
| P0 feasibility/version baseline | In progress | Selective materialization, resident Comfy bookkeeping, VEDA lifecycle, actual dependency signatures |
| P1 empty shell/executor | Local foundation integrated | Process-only CLI, no startup loader, environment/compute split, one worker, operation HTTP views; real empty boot pending |
| P2 partial weight lifecycle | Not integrated | Real selective read/materialize/purge/reload, quant storage, error recovery |
| P3 resident inference/artifacts | Not integrated | Strict weight-device invariants and complete AV latent roundtrip |
| P4 attention/VEDA | Not integrated | Correct rollback/clear and no implicit predictor copies |
| P5 compatibility/UI/routes | Not integrated | Explicit migration, public state sanitation, safe downloads |
| P6 GPU acceptance | Not run | Same-input latent/frames/resources and real sparse-call evidence |

No GPU load, inference baseline, or attention kernel verification has been performed at this point. CPU mocks will not be reported as real CUDA acceptance.

## Important execution boundaries

- No default auto_load/offload/preload or implicit attention installation.
- Partial disk2ram must not materialize the entire model then delete unselected weights.
- Comfy function overloads are service-owned, version-checked, restorable wrappers/facades; not a global no-op or an edit to installed third-party source.
- Routine environment inspection is read-only. Do not terminate/restart existing GPU services or run stress probes without an agreed safe device window.
- Preserve user's files/worktree changes; do not push/merge to the original branch without authorization.

## Reviews and test results

### Reviewed executor delivery

- Child B initial `6ca7535`: primary review identified readiness, admission revalidation, retained callback/payload/exception frames, unsafe public JSON, and asynchronous GPU completion boundaries.
- Child B corrective `0d101db`: primary inspected the diff and reran the 26-case stdlib suite; passed, including weak-reference cleanup and synchronization-before-terminal assertions.
- Personally integrated as `6de0165` and `e3bfcac`; no merge/push into the original TE branch.
- Operations expose one worker and queue, detached errors, bounded receipts and atomic control admission. Model-specific readiness is still refined by subsequent P2/P3 adapters.

### Primary empty startup / HTTP foundation

- `tests/_selftest_empty_startup.py`: 5 passed; default CLI contains no model/attention actions, relative paths fixed before runner.chdir, no runtime/kernel loading during environment setup, retriable environment failure.
- `tests/_selftest_control_http.py`: passed against a real local ThreadingHTTPServer and QueueManager with empty fake environment; status/weights/operation query and cancel errors are JSON and responsive.
- `tests/_check_call_sites.py`: passed after the Runtime changes.
- `git diff --check`: no whitespace errors.
- `tests/_selftest_execution_boundary.py`: 2 passed; subsystem error categories
  survive the worker boundary, and legacy jobs retain their reservation until
  the synchronization hook completes, just like operations.
- The four weight POSTs and resident inference are not yet wired; a working empty HTTP foundation is not full explicit-weight acceptance.

### Other reviews still open

- A `9353342`/`66467fe`: CPU catalog/storage/fake lifecycle evidence, but real H3 materialization is not yet proven. Primary requested recipe-option conflict, deep snapshot, catalog replacement, header bounds and recovery fixes; real adapter work continues in the same session.
- D `083b885`/`b3169d6`: primary found stage-root symlink, reference/delete race, quota/manifest rollback, batch_index and artifact validation edge cases. Primary reran `e3b95f7`/`5dddcbf`; all three suites passed, but concrete near-quota, no-clobber publication, invalid builder and failed-cleanup counterexamples still require correction in the same session. Existing green tests do not constitute merge approval.
- C: resident Comfy and VEDA lifecycle work continues; no result has been accepted yet.

Four fixed children are recorded in `agent_work_ledger.md`; slot E remains unused because its exact model was not listed by the provider.
