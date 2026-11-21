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

### Real CPU/meta feasibility probes (primary execution)

- Primary reviewed and locally reran A's `f4e758e` seven suites; passed. Ran
  its immutable probe bundle in a unique remote temporary directory, not the
  server checkout: `C:\Users\sheepserver\AppData\Local\Temp\explicit_api_probe_f4e758e_20261121_1650`.
- Probe 00 passed: all three package versions and nine source hashes match.
- Probe 10 initially failed because Comfy imports initialized CUDA despite CPU
  CLI flags. On Windows PowerShell an empty environment value removes the
  variable; `CUDA_VISIBLE_DEVICES=-1` correctly isolated the CPU-only process.
  With that isolation, the real H3 class built successfully: 50 blocks, 2 refiner
  blocks, 56 compute units, 2 explicitly noncompute keys, 268 meta tensors and
  zero CPU/CUDA resident storage. The successful full JSON is copied locally
  to `explicit_api_sources_20261121\probe_10_cpu_isolated.json` under approved temp.
- Probe 20 read only `dit.blocks.0` on CPU, then correctly failed `META_REMAINS`:
  all 18 keys bound, but only 1,763,840 CPU bytes and 4 meta tensors remained.
  Initial suspicion was the mixed-precision construction-device cache. A direct
  physical-leaf diagnostic disproved that explanation: all four QuantizedTensor
  parameters and their `_qdata` are non-meta CPU tensors, and factory devices
  are CPU. Their `_qdata` byte counts are 115,605,504 / 38,535,168 / 154,140,672 /
  77,070,336. The actual issue is `_walk_tensors`: the real quant wrapper inherits
  `torch.Tensor`, so its early tensor-return skips `_qdata` and typed parameters;
  the outer wrapper's unavailable storage is wrongly classified as meta. The
  earlier non-Tensor fake did not expose this. Corrective work and an actual
  Tensor-subclass regression were returned to session A. No GPU weight move is
  accepted until physical accounting and CPU selective purge/reload pass.
- No installed third-party source or existing server worktree was modified.

### Real empty-boot HTTP gate

- A reviewed `85f6de5` archive was extracted to a separate remote temporary
  directory. The bounded `tests/probes/empty_boot.py` started a new loopback
  server on an ephemeral port and stopped only the PID it created.
- Passed: `api_version=2`, `env_ready=true`, `ready=false`,
  `compute_ready=false`, `attn=none`, idle executor and all five model records
  absent with no catalogs or units. The status reported zero allocated and
  reserved PyTorch VRAM; no model/attention/load API actions were requested.
- Full report and log are retained locally in approved temp under
  `explicit_api_sources_20261121\empty_boot_report.json` / `.log`.

## Phase ledger

| Phase | Status | Evidence / pending gate |
|---|---|---|
| P0 feasibility/version baseline | In progress | Selective materialization, resident Comfy bookkeeping, VEDA lifecycle, actual dependency signatures |
| P1 empty shell/executor | Empty-boot gate passed | Process-only CLI, environment/compute split, one worker, operation HTTP views; real remote empty HTTP startup verified |
| P2 partial weight lifecycle | Not integrated | Real selective read/materialize/purge/reload, quant storage, error recovery |
| P3 resident inference/artifacts | Artifact component locally integrated | CPU safetensors roundtrip and safe downloads; resident inference and real GPU invariants pending |
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

### Reviewed artifact component

- D initial `083b885`/`b3169d6`, corrections `e3b95f7`/`5dddcbf`/`5651128`:
  personally reviewed and reran storage, serialization and API-validation suites.
  Integrated as `4a364cd`, `b1b02ab`, `56de65b`, `2a74dcd`, `b5076bd`.
- Primary additionally retained the configured disk reserve after actual writes,
  extended the writer lease through hash/manifest publication, and pinned artifact
  downloads against cleanup. Primary regressions pass (2 storage, 3 HTTP).
- `GET /api/artifacts/<id>` and safe `/outputs/stage/<name>` are wired; invalid
  paths are rejected rather than flattened with basename. Suffix ranges and 416
  responses are covered. These CPU/HTTP tests are not GPU inference acceptance.
- B staged runner `a08e30e` was personally reviewed/tested and integrated as
  `f1c9c6b`; the primary narrowed weights-review discovery to its subsystem.

### Other reviews still open

- A `9353342`/`66467fe`: CPU catalog/storage/fake lifecycle evidence, but real H3 materialization is not yet proven. Primary requested recipe-option conflict, deep snapshot, catalog replacement, header bounds and recovery fixes; real adapter work continues in the same session.
- C: resident Comfy and VEDA lifecycle work continues; no result has been accepted yet.

Four fixed children are recorded in `agent_work_ledger.md`; slot E remains unused because its exact model was not listed by the provider.

## 2026-11-21 continuation: unified weight API + real remote verification

- Integrated `agent/explicit-weights` (A) and B's `975fd64` (`h3_patcher`). C branch
  (`agent/explicit-bindings`) remains unmerged; attention/VEDA resident bindings are
  not part of this round.
- New `WeightService` with five backends: `dit` (block-level via A's `H3ComfyAdapter`),
  `vae.video` / `vae.audio` (whole-group Comfy VAE), `te` (whole-group CLIP + ClipProj),
  `veda.bundle` (whole-group `LoadedBundle`). One HTTP contract:
  `POST /api/weights/<model>/<action>` -> 202 operation; `dry_run` never constructs.
- `POST /api/infer/encode` is resident-only: `te.model` must already be `vram`; the cond
  artifact is committed to `ArtifactStore` (`h3.cond.v1`). Legacy `/api/encode` keeps a
  bounded synchronous wait on the same worker. `/api/te/step` and `/api/te/probe` remain
  legacy debug endpoints (not yet migrated to operations).
- Local selftests: `--phase all` = 15/15 (p0 7, p1 5, p2 3). New coverage:
  `_selftest_te_manager.py` (13 cases) and `_selftest_weight_api.py` (HTTP contract).

### Remote deployment

- Remote worktree: `C:\Users\sheepserver\Desktop\h3_sm75_server_explicit_api`,
  branch `feat/explicit-api`; defaults file `Desktop\h3_explicit_defaults.json`
  uses the user-profile weights (DiT w4a8, **light** video VAE, TE 4B + ClipProj)
  and D-drive audio VAE / VEDA bundle.
- GPU idle at start: 272 MiB / 22528 MiB used. The probes start and stop only their
  own server PID; no user service was touched.

### Real remote results (2026-11-21)

| model | action | result |
|---|---|---|
| te | disk2ram | done 0.51s; 4,858,391,108 B CPU |
| te | ram2vram | done 2.73s; 4,858,391,108 B CUDA; proj 304 MB stays CPU (partial, expected) |
| te | infer.encode | done 3.02s; 24 tokens; cond `[1,24,5120]`; artifact 492,096 B, download bytes match |
| te | vram2ram / purge | 2.02s / 0.63s; final state `absent`, 0 resident bytes |
| dit | catalog | 56 units; `disk2ram` dry_run lists all |
| dit | disk2ram `dit.blocks.0` | done 0.52s; 387,359,232 B (matches probe 20) |
| dit | ram2vram / vram2ram / purge | 1.01s / 0.51s / 0.52s; `partial` -> `absent` |
| vae.video | disk2ram (light 2.14 GB) | done 8.17s; official 2.8 GB file took 135.26s |
| vae.audio | disk2ram (605 MB fp32) | done 18.58s |
| veda.bundle | disk2ram | done 1.01s; 550,502,400 B bf16 resident (fp8 source 275 MB) |

All four actions for all five models passed on the real GPU host. This proves the
API/state machine and physical accounting; it does **not** prove DiT sampling or
VAE decode results, nor attention/VEDA sparse execution (C not merged).