"""
Individual, publication-style figures for the ASR benchmarking paper.

Reads the same input workbook as make_all_figures.py (excel/VTC2_HUman.xlsx) and
writes ONE figure per file (no multi-panel grids) into
    output/figures_individual/
Each figure is deliberately plain: a single accent colour, an identity line, an
OLS fit, and a small text box with r and n. Nothing decorative.

Run:  python3 code/make_paper_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IN_XLSX = os.path.join(ROOT, "excel", "VTC2_HUman.xlsx")
OUTDIR = os.path.join(ROOT, "output", "figures_individual")
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.90",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})
ACCENT = "#3f6da3"      # muted blue for markers
FIT = "#33475b"         # dashed OLS fit
IDENT = "0.55"          # identity line
ENG_C = "#4c78a8"       # english
SPA_C = "#e0913c"       # spanish

# 5 pipelines, each a block of 5 columns: Total, English, Spanish, English%, Spanish%
MODELS = ["Human", "HumanTS", "VTC1", "VTC2", "OnlyWhisper"]
BLOCK0 = {"Human": 2, "HumanTS": 7, "VTC1": 12, "VTC2": 17, "OnlyWhisper": 22}


def load_speaker(sheet):
    """Return a tidy per-recording DataFrame for one speaker sheet."""
    raw = pd.read_excel(IN_XLSX, sheet_name=sheet, header=None)
    data = raw.iloc[3:].copy()          # rows 0-2 are title/header/subheader
    # Recording id lives in a merged cell: only the first clip of each recording
    # carries it, so forward-fill down the column. The clip id is in column 1.
    rec = data.iloc[:, 0].ffill().astype(str)
    clip = data.iloc[:, 1].astype(str)
    out = pd.DataFrame({"recording": rec.values, "clip": clip.values})
    for m in MODELS:
        c = BLOCK0[m]
        out[f"{m}_total"] = pd.to_numeric(data.iloc[:, c].values, errors="coerce")
        out[f"{m}_en"] = pd.to_numeric(data.iloc[:, c + 1].values, errors="coerce")
        out[f"{m}_es"] = pd.to_numeric(data.iloc[:, c + 2].values, errors="coerce")
    # keep real clip rows only; drop the per-recording "_COMBINED" subtotal row
    keep = out["clip"].str.startswith("DL-") & ~out["clip"].str.contains("COMBINED")
    out = out[keep].copy()
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].fillna(0)         # N/A = pipeline produced no output
    # collapse 12 clips -> 1 row per recording
    return out.groupby("recording", as_index=False).sum(numeric_only=True)


def ols(x, y):
    b, a = np.polyfit(x, y, 1)          # slope, intercept
    return b, a


def scatter(x, y, xlabel, ylabel, fname, title=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = np.corrcoef(x, y)[0, 1]
    lim = max(x.max(), y.max()) * 1.08
    fig, ax = plt.subplots(figsize=(4.1, 4.0))
    ax.plot([0, lim], [0, lim], color=IDENT, lw=1.0, zorder=1)
    b, a = ols(x, y)
    xs = np.array([0, lim])
    ax.plot(xs, a + b * xs, "--", color=FIT, lw=1.3, zorder=2)
    ax.scatter(x, y, s=70, color=ACCENT, edgecolor="white", linewidth=0.8,
               alpha=0.9, zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.text(0.04, 0.95, f"r = {r:.2f}\nn = {len(x)}", transform=ax.transAxes,
            va="top", ha="left", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), bbox_inches="tight")
    plt.close(fig)


def bland_altman(x, y, unit, fname, title=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mean = (x + y) / 2.0
    diff = y - x                        # VTC2 - Human
    bias = diff.mean(); sd = diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    fig, ax = plt.subplots(figsize=(4.3, 3.9))
    ax.axhline(0, color="0.8", lw=0.9)
    ax.axhline(bias, color=FIT, lw=1.3)
    ax.axhline(hi, color="#b5493b", lw=1.0, ls="--")
    ax.axhline(lo, color="#b5493b", lw=1.0, ls="--")
    ax.scatter(mean, diff, s=70, color=ACCENT, edgecolor="white",
               linewidth=0.8, alpha=0.9, zorder=3)
    ax.set_xlabel(f"Mean of human and VTC2 ({unit})")
    ax.set_ylabel("VTC2 − human (words)")
    if title:
        ax.set_title(title, fontsize=11)
    xr = ax.get_xlim()[1]
    ax.text(xr, bias, f" bias = {bias:+.0f}", va="center", ha="left",
            fontsize=9, color=FIT, clip_on=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), bbox_inches="tight")
    plt.close(fig)


def scatter_combined(df, spk, fname):
    """One scatter per speaker: total / English / Spanish as three colours."""
    series = [
        ("total", "Total", "#5a6b7b"),
        ("en", "English", ENG_C),
        ("es", "Spanish", SPA_C),
    ]
    x_all = np.concatenate([df[f"Human_{m}"].values for m, _, _ in series])
    y_all = np.concatenate([df[f"VTC2_{m}"].values for m, _, _ in series])
    lim = max(x_all.max(), y_all.max()) * 1.08
    fig, ax = plt.subplots(figsize=(4.6, 4.5))
    ax.plot([0, lim], [0, lim], color=IDENT, lw=1.0, zorder=1)
    for m, lab, col in series:
        x = df[f"Human_{m}"].values.astype(float)
        y = df[f"VTC2_{m}"].values.astype(float)
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=62, color=col, edgecolor="white", linewidth=0.8,
                   alpha=0.9, zorder=3, label=f"{lab}  (r = {r:.2f})")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Human reference (words)")
    ax.set_ylabel("VTC2 + WhisperX (words)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, columnspacing=1.0, handletextpad=0.3, fontsize=9)
    ax.text(0.97, 0.05, "above line = overcount\nbelow line = undercount",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
            color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- load
adult = load_speaker("ADULT")
kchi = load_speaker("KCHI")

scatter_combined(adult, "Adult", "scatter_combined_adult.png")
scatter_combined(kchi, "Key child", "scatter_combined_kchild.png")

pairs = [
    (adult, "Adult", "total", "adult_total", "Adult — total words"),
    (adult, "Adult", "en", "adult_english", "Adult — English words"),
    (adult, "Adult", "es", "adult_spanish", "Adult — Spanish words"),
    (kchi, "Key child", "total", "kchild_total", "Key child — total words"),
    (kchi, "Key child", "en", "kchild_english", "Key child — English words"),
    (kchi, "Key child", "es", "kchild_spanish", "Key child — Spanish words"),
]

for df, spk, meas, tag, title in pairs:
    x = df[f"Human_{meas}"]; y = df[f"VTC2_{meas}"]
    scatter(x, y, "Human reference (words)", "VTC2 + WhisperX (words)",
            f"scatter_{tag}.png", title=title)
    bland_altman(x, y, "words", f"ba_{tag}.png", title=title)

# ----------------------------------------------------------------------------- language balance (overall = adult + kchi + och etc. -> use OVERALL sheet)
overall = load_speaker("OVERALL")
tot_en = {m: overall[f"{m}_en"].sum() for m in MODELS}
tot_es = {m: overall[f"{m}_es"].sum() for m in MODELS}
labels = ["Human", "Human-TS\n+Whisper", "VTC1\n+Whisper", "VTC2\n+Whisper", "Only\nWhisper"]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
en = [tot_en[m] for m in MODELS]; es = [tot_es[m] for m in MODELS]
xpos = np.arange(len(MODELS))
ax.bar(xpos, en, color=ENG_C, label="English tokens", edgecolor="white")
ax.bar(xpos, es, bottom=en, color=SPA_C, label="Spanish tokens", edgecolor="white")
ax.axhline(tot_en["Human"] + tot_es["Human"], color="0.5", ls=":", lw=1.1)
for i, m in enumerate(MODELS):
    ax.text(i, en[i] + es[i] + 400, f"{en[i]+es[i]:,.0f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Tokens (all 168 clips)")
ax.set_ylim(0, max(en[i] + es[i] for i in range(len(MODELS))) * 1.13)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=2, columnspacing=1.6, handletextpad=0.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "lang_balance.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- percent Spanish: expert vs VTC2, pooled, by speaker
def pct_es(df, model):
    return 100 * df[f"{model}_es"].sum() / df[f"{model}_total"].sum()

spk_names = ["Adult", "Key child"]
exp = [pct_es(adult, "Human"), pct_es(kchi, "Human")]
vtc = [pct_es(adult, "VTC2"), pct_es(kchi, "VTC2")]
xpos = np.arange(len(spk_names)); w = 0.36
fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.bar(xpos - w/2, exp, w, color="#5a6b7b", label="Expert reference", edgecolor="white")
ax.bar(xpos + w/2, vtc, w, color=ACCENT, label="VTC2 + WhisperX", edgecolor="white")
for i in range(len(spk_names)):
    ax.text(i - w/2, exp[i] + 1, f"{exp[i]:.0f}%", ha="center", fontsize=9.5)
    ax.text(i + w/2, vtc[i] + 1, f"{vtc[i]:.0f}%", ha="center", fontsize=9.5)
ax.set_xticks(xpos); ax.set_xticklabels(spk_names)
ax.set_ylabel("% Spanish (pooled)"); ax.set_ylim(0, 108)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=2, columnspacing=1.6, handletextpad=0.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "pct_spanish.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- per-language correlation by model (grouped bars, recording level)
def rec_corr(sheet_df, model, lang):
    x = sheet_df[f"Human_{lang}"]; y = sheet_df[f"{model}_{lang}"]
    return np.corrcoef(x, y)[0, 1]

comp = [m for m in MODELS if m != "Human"]
comp_lab = ["Human-TS+Whisper", "VTC1+Whisper", "VTC2+Whisper", "Only Whisper"]
en_r = [rec_corr(overall, m, "en") for m in comp]
es_r = [rec_corr(overall, m, "es") for m in comp]
xpos = np.arange(len(comp)); w = 0.36
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.bar(xpos - w/2, en_r, w, color=ENG_C, label="English tokens", edgecolor="white")
ax.bar(xpos + w/2, es_r, w, color=SPA_C, label="Spanish tokens", edgecolor="white")
for i in range(len(comp)):
    ax.text(i - w/2, en_r[i] + 0.01, f"{en_r[i]:.2f}", ha="center", fontsize=8.5)
    ax.text(i + w/2, es_r[i] + 0.01, f"{es_r[i]:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(xpos); ax.set_xticklabels(comp_lab, fontsize=9, rotation=12)
ax.set_ylabel("Pearson r with human tokens"); ax.set_ylim(0, 1.14)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=2, columnspacing=1.6, handletextpad=0.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "corr_by_model.png"), bbox_inches="tight")
plt.close(fig)

print("wrote figures to", OUTDIR)
for f in sorted(os.listdir(OUTDIR)):
    print("  ", f)
