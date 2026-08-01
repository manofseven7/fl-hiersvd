import os, json, glob, numpy as np
from scipy import stats

RES = "results"
STD = ["fedavg","fedprox","scaffold","fednova","fedkd","powersgd","qsgd",
       "fedpaq","fedcom","topk","adaptive_topk","sign"]

# (dataset, method, tag, table-label)
MAIN = ([(d, m, "", m) for d in ("fashion","har") for m in STD]
        + [("fashion","proposed","nofs","proposed_unmasked")]
        + [("har","proposed","nofs","proposed_nofs")]
        + [("har","proposed","fs","proposed_fs")])
CIFAR = [("cifar", m, "", m) for m in ("fedavg","powersgd","qsgd","proposed")]

def files(ds, m, tag):
    suf = f"_{tag}" if tag else ""
    return sorted(glob.glob(f"{RES}/{ds}_{m}{suf}_s*.json"))

def seed_map(ds, m, tag):
    out = {}
    for f in files(ds, m, tag):
        s = int(f.rsplit("_s", 1)[1].split(".")[0]); out[s] = json.load(open(f))
    return out

def mean_sd(ds, m, tag, key="final_acc"):
    recs = [json.load(open(f)) for f in files(ds, m, tag)]
    if not recs: return None, None
    v = np.array([r[key] for r in recs])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

def paired(ds, m1, t1, m2, t2, key="final_acc"):
    """diff = method1 - method2, paired by seed; returns (n, diff, ci95_half, p_two)."""
    a, b = seed_map(ds, m1, t1), seed_map(ds, m2, t2)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2: return None
    x = np.array([a[s][key] for s in seeds]); y = np.array([b[s][key] for s in seeds])
    d = x - y; md = d.mean(); sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, len(seeds)-1) * sd / np.sqrt(len(seeds))
    _, p = stats.ttest_rel(x, y)
    return dict(n=len(seeds), diff=md, ci=half, p=p)

def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m-rank)*p[i]); adj[i] = min(run, 1.0)
    return adj

def table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'dataset':9s}{'method':18s}{'acc':>13s}{'macro-F1':>13s}{'MB':>9s}")
    for ds, m, tag, lab in rows:
        am, asd = mean_sd(ds, m, tag, "final_acc")
        fm, fsd = mean_sd(ds, m, tag, "final_f1")
        cm, _ = mean_sd(ds, m, tag, "comm_mb")
        if am is None:
            print(f"{ds:9s}{lab:18s}{'(missing)':>13s}{'--':>13s}{'--':>9s}"); continue
        f1txt = f"{fm:7.2f}±{fsd:4.2f}" if fm is not None else "--"
        print(f"{ds:9s}{lab:18s}{am:7.2f}±{asd:4.2f}{f1txt:>13s}{cm:9.2f}")

def paired_block():
    print("\n=== Paired statistical comparisons (per-seed; diff = Proposed - X) ===")
    # confirmatory family (Holm-corrected): 4 tests
    conf = [("fashion","proposed","nofs","fedavg",""),
            ("fashion","proposed","nofs","powersgd",""),
            ("har","proposed","nofs","fedavg",""),
            ("har","proposed","nofs","powersgd","")]
    rs = [paired(*c) for c in conf]; ps = [r["p"] for r in rs if r]
    adj = holm(ps) if ps else []
    k = 0
    for c, r in zip(conf, rs):
        if r is None:
            print(f"  {c[0]:8s} Prop-{c[3]:9s} MISSING"); continue
        print(f"  {c[0]:8s} Prop-{c[3]:9s} diff={r['diff']:+.3f}  "
              f"95%CI[{r['diff']-r['ci']:+.3f},{r['diff']+r['ci']:+.3f}]  "
              f"p={r['p']:.4f}  Holm={adj[k]:.4f}"); k += 1
    # feature-selection component test (HAR fs vs nofs)
    r = paired("har","proposed","fs","proposed","nofs")
    if r:
        print(f"  {'har':8s} FS-vs-noFS   diff={r['diff']:+.3f}  "
              f"95%CI[{r['diff']-r['ci']:+.3f},{r['diff']+r['ci']:+.3f}]  p={r['p']:.4f}")
    # CIFAR (exploratory, NOT Holm-corrected)
    print("  --- CIFAR-10 (exploratory, unadjusted) ---")
    for x in ("fedavg","powersgd","qsgd"):
        r = paired("cifar","proposed","",x,"")
        if r:
            print(f"  {'cifar':8s} Prop-{x:9s} diff={r['diff']:+.3f}  "
                  f"95%CI[{r['diff']-r['ci']:+.3f},{r['diff']+r['ci']:+.3f}]  p={r['p']:.4f}")

if __name__ == "__main__":
    table(MAIN, "Table 5 (Fashion + HAR, 30 rounds)")
    table(CIFAR, "Table 7 (CIFAR-10, 100 clients, 400 rounds)")
    paired_block()
