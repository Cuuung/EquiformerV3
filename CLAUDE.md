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
