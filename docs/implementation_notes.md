# Explicit API implementation notes

## Baseline and current scope

Implementation authorized on 2026-11-21. The user allows the listed agents, dedicated worktrees, phase commits, and primary-agent-only review/integration.

- Baseline: `5caf1bf`, original worktree clean at implementation start.
- Integration branch: `feat/explicit-api`, separate worktree; original TE branch preserved.
- Authoritative design: `../plan_20261121_152726/`.
- Local interpreter: Python 3.10.11; torch `2.8.0+cpu`, CUDA unavailable; `psutil` present, `safetensors` initially absent.
- Existing `_remote_probe` source snapshots are clues, not yet verified against the running GPU environment.

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
