# VTC2 + WhisperX vs Human — Word-Count Agreement Figures

A single Python pipeline that regenerates **every figure, Excel table, and PDF bundle**
for the ASR meeting experiment (VTC2+WhisperX vs Human reference, plus a 5-model
English/Spanish token analysis).

## Folder layout

```
.
├── code/
│   └── make_all_figures.py     # the whole pipeline — one file, one command
├── excel/
│   └── VTC2_HUman.xlsx          # INPUT data (the only file the code reads)
└── output/                      # ALL generated results (created on run; git-ignored)
    ├── excel/                   # cleaned data tables + correlations
    │   ├── VTC2_vs_Human_168clips.xlsx
    │   ├── VTC2_vs_Human_14recordings.xlsx
    │   ├── analysis2_data_tidy.xlsx
    │   └── correlations_model_vs_human.csv
    └── figures/                 # 42 PNGs + 2 PDF bundles
        ├── fig5_*, figA–D, figR_*            # clip- and recording-level scatter/agreement
        ├── VTC2_vs_Human_figures_bundle.pdf
        ├── analysis2_figures_bundle.pdf
        └── analysis2/                        # correlation heatmaps + stacked-bar figures
```

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 code/make_all_figures.py
```

Everything is written under `output/`. The script anchors all paths to its own
location, so it works from any working directory, and it recreates any missing
output folders — you can delete `output/` and re-run for a clean regeneration.

## What the pipeline does (in order)

1. **168-clip clean Excel + fig5** — per-clip scatter, colored by recording.
2. **figA–D** — clip-level scatter with regression stats, Bland–Altman agreement,
   per-recording small multiples, and color+shape scatter.
3. **14-recording Excel + figR_*** — recording-level versions (sum of 12 clips each).
4. **Experiment PDF bundle** — all of the above, captioned.
5. **analysis2** — English vs Spanish token analysis across 5 models
   (Human, Human-TS+Whisper, VTC1+Whisper, VTC2+Whisper, Only Whisper):
   correlation heatmaps + stacked-bar figures at clip, recording, and per-model level.
6. **analysis2 PDF bundle** — all analysis2 figures, captioned.

## Notes

- `N/A` values (pipeline produced no output for near-silent clips) are treated as `0`.
- The `output/` folder is git-ignored because it is fully regenerable. To version the
  results too, delete the `output/` line from `.gitignore`.
