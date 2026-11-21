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
| P1 empty shell/executor | Not integrated | One worker, atomic admission, empty model references, callable control APIs |
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

Pending child deliveries. Fixed child IDs and owned scopes are in `agent_work_ledger.md`.
