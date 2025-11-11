import argparse, re, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sanitize(name: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", name.strip())
    return re.sub(r"_+", "_", s).strip("_") or "param"

def tensor_to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
    except Exception:
        pass
    return np.asarray(x)

def title_from_name(name: str) -> str:
    # Preferred names: layer{L}_{tag}
    m = re.match(r"layer(\d+)_(.+)$", name)
    if m:
        return f"Layer {m.group(1)}  {m.group(2)}"
    # Fallback: mod.blocks.L.attn.qkvo_w_I  (I: 0=q,1=k,2=v,3=o)
    m = re.search(r"blocks\.(\d+)\.attn\.qkvo_w_(\d+)", name)
    if m:
        tag = ["q","k","v","o"][int(m.group(2))]
        return f"Layer {m.group(1)}  {tag}"
    # Fallback: mod.blocks.L.mlp.(fc_w|proj_w)
    m = re.search(r"blocks\.(\d+)\.mlp\.(fc_w|proj_w)", name)
    if m:
        tag = "mlp_fc" if m.group(2) == "fc_w" else "mlp_proj"
        return f"Layer {m.group(1)}  {tag}"
    # Otherwise, just show the name
    return name

def plot_hist_singlepanel(fig_title: str, values: np.ndarray, bins: int, out_pdf: Path):
    fig = plt.figure(figsize=(8.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    vals = values[np.isfinite(values)]
    ax.hist(vals, bins=bins, density=False, color="indianred", edgecolor="none")
    ax.set_xlabel("singular value"); ax.set_ylabel("count")
    ax.set_title(f"{fig_title}")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="logs_spectral/muon_5960/singular_values.pkl",
                    help="Path to pickle {name -> 1D tensor of singular values}")
    ap.add_argument("--outdir", default="logs_spectral/muon_5960/sv_hists",
                    help="Directory to write per-weight PDFs")
    ap.add_argument("--bins", type=int, default=100, help="Histogram bins")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, "rb") as f:
        sv_map = pickle.load(f)

    count = 0
    for raw_name, s in sv_map.items():
        arr = tensor_to_numpy(s).ravel().astype(np.float64)
        safe = sanitize(raw_name)
        title = title_from_name(raw_name)  # e.g., "Layer 0 · q"
        plot_hist_singlepanel(title, arr, args.bins, outdir / f"{safe}.png")
        count += 1

    print(f"Wrote {count} per-weight PDFs to: {outdir}")

if __name__ == "__main__":
    main()
