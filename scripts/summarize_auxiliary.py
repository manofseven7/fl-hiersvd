import glob, json, sys
import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "profile"
root = f"results/{mode}"
files = sorted(glob.glob(f"{root}/*.json"))
if not files:
    raise SystemExit(f"no JSON logs found under {root}")

groups = {}
for path in files:
    rec = json.load(open(path)); cfg = rec["cfg"]
    if mode == "profile": key = (cfg["dataset"], cfg["method"])
    else: key = (cfg["clients"], cfg["edges"])
    groups.setdefault(key, []).append(rec)

if mode == "profile": print("dataset method n time_s_mean time_s_sd peak_rss_mb comm_mb")
else: print("clients edges n time_s_mean time_s_sd peak_rss_mb comm_mb")
for key, recs in sorted(groups.items()):
    times=np.array([r["wall_time_s"] for r in recs]); rss=np.array([r["peak_rss_mb"] for r in recs]); comm=np.array([r["comm_mb"] for r in recs])
    sd=times.std(ddof=1) if len(times)>1 else 0.0
    print(*key, len(recs), f"{times.mean():.2f}", f"{sd:.2f}", f"{rss.mean():.2f}", f"{comm.mean():.2f}")
