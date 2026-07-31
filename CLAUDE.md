# CLAUDE.md

Project-level guidance for Claude Code in this repository.

## Documentation / Markdown file placement

Keep the repository root clean. **Only `README.md` and `CLAUDE.md` may live at the top level.**

All other Markdown documents — design notes, analyses, porting notes, submission
guides, regression write-ups, etc. — must be organized into a secondary folder,
**`docs/`** (or an appropriate sub-folder of it). Do **not** drop general `.md`
docs in the main folder.

- New notes/analyses → create them under `docs/` from the start.
- Note: `docs/` is also the Jupyter-book source (`_toc.yml`, `_config.yml`,
  `index.md`, …). Stray `.md` files placed there are harmless — the book build
  only includes files listed in `docs/_toc.yml` — but if a doc should appear in
  the published book, add it to `_toc.yml`.

## Evaluation (Matbench) sync — `docs/EVAL_REGISTRY.md`

`docs/EVAL_REGISTRY.md` is the **single source of truth** shared with the Matbench
evaluation side. It is a git-tracked table; the diff is the sync log. All κ_SRME /
F1 / RMSD / CPS numbers flow through this file — do **not** carry eval results only
in chat, memory, or other docs.

**Four rules (both sides obey):**

1. **Immutable key = `ckpt_id`** — the `--identifier` string we set at submit time
   (also the suffix of the checkpoint dir). Once written it is **never renamed or
   deleted**. The runtime timestamp prefix is not the key; it just fills `ckpt_path`.
2. **One recipe = one row.** A changed recipe / re-train = new ckpt = a **new row**.
   Never overwrite an existing row's key or its results.
3. **Columns have owners.** *This project* writes `status / ckpt_id / ckpt_path /
   stage / recipe / notes`. *Eval side* writes `κ_SRME / F1 / RMSD / CPS /
   eval_date` and flips `status` `REQUESTED → DONE`. Neither edits the other's
   columns. The eval side **makes no judgments** — it only back-fills results
   against the immutable key.
4. **Corrections never mutate old rows.** An old ckpt's measured numbers stay as-is;
   a changed interpretation goes in `notes` or the "结论修正区" section, or a new row
   with `superseded-by: <ckpt_id>`. This guarantees the eval side's key always
   resolves to the same row.

**Workflow.** On submit, append a `REQUESTED` row (result cells blank). The eval
side watches `REQUESTED` rows, fills numbers, flips to `DONE`. We read `DONE` rows
to pull numbers. Recipe organization / interpretation is done **only in this
project**; the eval side just appends results in chronological order.

The onboarding prompt handed to the eval-side Claude Code lives at
`docs/EVAL_SIDE_ONBOARDING_PROMPT.md`.
