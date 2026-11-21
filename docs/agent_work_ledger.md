# Explicit API agent ledger

## User constraints

- Maximum five child agents, each using a distinct model from the user's list.
- Reuse the recorded session ID for follow-up work; do not create replacement agents when slots are exhausted.
- Each child has its own Git worktree and makes phase-scoped commits.
- The primary agent personally reviews and integrates commits. Children must not merge, cherry-pick, push, or create further children.

## Integration worktree

- Branch: `feat/explicit-api`
- Worktree: `C:\Users\88888888\work-on-comfyui\h3_sm75_server_explicit_api`
- Baseline: `5caf1bfbfb4df0b4972d8b7bebc7188886bf33f8`
- Original `feat/te-proj-swap` worktree is preserved.
- Plan: `../plan_20261121_152726/README.md`

## Fixed agent slots

| Slot | Model | Session ID | Branch / worktree suffix | Owned scope |
|---|---|---|---|---|
| A | `opencode-go/muse-spark-1.3-contributor` | `ses_e02ec8c81ffeupPJgZKuPpoDEU` | `agent/explicit-weights` / `_wt_weights` | weights, model adapters, selective-load/storage tests |
| B | `opencode-go/deepseek-v4.1-flash` | `ses_e02ec3b95ffe0Y8dejYGcn3GKl` | `agent/explicit-executor` / `_wt_executor` | operations, sole queue worker/admission, concurrency tests |
| C | `opendesign-01/deepseek-v4.1-flash` | `ses_e02ebebf4ffedN3o4e3a6av5t5` | `agent/explicit-bindings` / `_wt_bindings` | Comfy bindings, attention controller, VEDA bridge/lifecycle tests |
| D | `opendesign-01/deepseek-v4-flash` | `ses_e02eba516ffe7hxUqXOynrkeRk` | `agent/explicit-artifacts` / `_wt_artifacts` | artifacts, API validation, path/serializer tests |
| E | `opendesign-01/kimi-k2.7-code` | Not created | Reserved | Model absent from provider catalog at initial lookup |

All child worktree names have prefix `C:\Users\88888888\work-on-comfyui\h3_sm75_server`.
Four children created; no model duplicated. Slot E remains unused, not replaced by an unlisted model.

## Integration rules

- Primary-owned files: entry, runtime, HTTP routes, infer/pipeline integration, environment probes, implementation notes, final review/test fixes.
- Child scopes are non-overlapping; shared interface changes are communicated using their existing session IDs.
- Before integrating: inspect each commit/diff, run focused tests, resolve interface assumptions, then cherry-pick reviewed commits.
- Record review findings and integrated commit IDs in `implementation_notes.md`; a phase is not complete merely because a child committed code.
