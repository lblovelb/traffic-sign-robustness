"""Turn results.csv into a summary table and robustness plot.

python -m tsr.report results/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fmt(mean: float, std: float) -> str:
    # std is NaN when there is only one seed; then show the mean alone.
    return f"{mean:.3f} ± {std:.3f}" if pd.notna(std) else f"{mean:.3f}"


def summarize(df: pd.DataFrame, seen: list[str] | None = None) -> pd.DataFrame:
    """One row per (model, train): clean vs corrupted accuracy, mean ± std over seeds.

    ``seen`` = corruptions used for augmentation. Grouping every model by the
    same seen/unseen split makes the regimes directly comparable.
    """
    if seen is None:
        seen = sorted(df.loc[df["seen_in_training"].astype(bool), "corruption"].unique())
    df = df.copy()
    # Label every row as clean, seen (used in augmentation) or unseen.
    df["group"] = "unseen"
    df.loc[df["corruption"] == "clean", "group"] = "clean"
    df.loc[df["corruption"].isin(seen), "group"] = "seen"

    # Mean per seed first, then mean/std across seeds. The std therefore measures
    # how much results vary between training runs, not between corruption types.
    per_seed = df.groupby(["model", "train", "seed", "group"])["accuracy"].mean().unstack("group")
    per_seed["all_corrupted"] = (
        df[df["group"] != "clean"].groupby(["model", "train", "seed"])["accuracy"].mean()
    )
    agg = per_seed.groupby(["model", "train"]).agg(["mean", "std"])

    rows = []
    for (model, train), r in agg.iterrows():
        row = {"model": model, "train": train}
        for col in ["clean", "all_corrupted", "seen", "unseen"]:
            if (col, "mean") in r and pd.notna(r[(col, "mean")]):
                row[col] = _fmt(r[(col, "mean")], r[(col, "std")])
        lat = df[(df["model"] == model) & (df["train"] == train)]
        row["latency_ms"] = f"{lat['latency_ms'].mean():.3f}"
        row["size_mb"] = f"{lat['size_mb'].mean():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def per_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per corruption, averaged over severities and seeds."""
    sub = df[df["corruption"] != "clean"]
    table = sub.groupby(["model", "train", "corruption"])["accuracy"].mean().unstack("corruption")
    return table.round(3).reset_index()


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One subplot per corruption, one line per (model, training data).
    names = [c for c in df["corruption"].unique() if c != "clean"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), sharey=True)
    clean = df[df["corruption"] == "clean"]
    for ax, name in zip(axes, names, strict=True):
        # Add the clean result as severity 0 so every line starts from the same baseline.
        sub = pd.concat([clean.assign(corruption=name), df[df["corruption"] == name]])
        for (model, train), g in sub.groupby(["model", "train"]):
            s = g.groupby("severity")["accuracy"].agg(["mean", "std"]).fillna(0)
            ax.plot(s.index, s["mean"], marker="o", label=f"{model} ({train})")
            # Shaded band = ± one standard deviation across seeds.
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
        ax.set_title(name)
        ax.set_xlabel("severity (0 = clean)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(csv_path: str | Path) -> Path:
    """Write summary.md and robustness.png next to the CSV. Called by experiment.py."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    out_dir = csv_path.parent
    plot(df, out_dir / "robustness.png")
    md = [
        "# Results summary",
        "",
        "Accuracy, mean ± std over seeds. *seen* = corruptions used in augmentation, "
        "*unseen* = never shown during training.",
        "",
        summarize(df).to_markdown(index=False),
        "",
        "## Accuracy per corruption (mean over severities)",
        "",
        per_corruption(df).to_markdown(index=False),
        "",
        "![robustness](robustness.png)",
        "",
    ]
    out = out_dir / "summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out} and {out_dir / 'robustness.png'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/results.csv")
    write_report(parser.parse_args().csv)


if __name__ == "__main__":
    main()
