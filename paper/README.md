# Paper Draft

This directory contains a CVPR-style paper draft derived from the current `VGGT-NBV` codebase.

Structure:
- `main.tex`: paper entrypoint.
- `preamble.tex`: package imports and shared macros.
- `sections/`: per-section source files.
- `references.bib`: bibliography for the current draft.
- `cvpr.sty`, `ieeenat_fullname.bst`: copied template assets.

Compile with Tectonic:

```bash
cd paper
~/.local/bin/tectonic main.tex
```

Or simply run:

```bash
cd paper
make
```

Current status:
- The manuscript is written to match the released repository architecture.
- The experiments section includes protocol-faithful placeholder tables instead of fabricated numbers.
- The appendix includes a repository-to-paper mapping for future revision.
