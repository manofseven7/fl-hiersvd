import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
ORDER = ["fedavg","fedprox","scaffold","fednova","fedkd","powersgd","qsgd",
         "fedpaq","fedcom","topk","adaptive_topk","sign","proposed"]
COL = dict(fedavg="#4477AA", fedprox="#77AADD", scaffold="#DD5544", fednova="#228833",
           fedkd="#AA3377", powersgd="#C99A2E", qsgd="#0077BB", fedpaq="#EE8866",
           fedcom="#999999", topk="#009988", adaptive_topk="#44AA99",
           sign="#B4B4BC", proposed="#15151A")
LABEL = dict(adaptive_topk="Adaptive-TopK-EF", sign="SignSGD-MV",
             powersgd="PowerSGD-EF", fedkd="FedKD-style")

def _tag(ds, method):
    if method != "proposed": return ""
    return "nofs" if ds == "fashion" else "fs"

def _files(ds, method, tag=None):
    tag = _tag(ds, method) if tag is None else tag
    suffix = f"_{tag}" if tag else ""
    return sorted(glob.glob(f"results/{ds}_{method}{suffix}_s*.json"))

def trajectories(ds, method, tag=None, metric="acc"):
    recs = [json.load(open(f)) for f in _files(ds, method, tag)]
    if not recs: return None
    lengths = {len(r["traj"]) for r in recs}
    if len(lengths) != 1: raise ValueError(f"trajectory-length mismatch: {ds}/{method}")
    val = np.array([[p[metric] for p in r["traj"]] for r in recs], float)
    comm = np.array([[p["comm_mb"] for p in r["traj"]] for r in recs], float)
    return val.mean(0), val.std(0, ddof=1) if len(recs) > 1 else np.zeros(val.shape[1]), comm.mean(0)

def two_dataset_trajectory(metric, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ds, title in zip(axes, ("fashion", "har"), ("Fashion-MNIST", "UCI HAR")):
        for method in ORDER:
            z = trajectories(ds, method, metric=metric)
            if z is None: continue
            mean, sd, comm = z; hero = method == "proposed"
            ax.plot(comm, mean, color=COL[method], lw=2.5 if hero else 1.4,
                    label=LABEL.get(method, method))
            ax.fill_between(comm, mean-sd, mean+sd, color=COL[method], alpha=.14 if hero else .06)
        ax.set_xscale("log"); ax.set_title(title)
        ax.set_xlabel("Cumulative 3-link communication (MB)")
        ax.set_ylabel("Test accuracy (%)" if metric == "acc" else "Macro-F1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7)
    fig.tight_layout(rect=(0, .10, 1, 1)); fig.savefig(f"{OUT}/{filename}")
    plt.close(fig); print("saved", filename)

def final_communication():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ds, title in zip(axes, ("fashion", "har"), ("Fashion-MNIST", "UCI HAR")):
        vals=[]; labs=[]; colors=[]
        for method in ORDER:
            recs=[json.load(open(f)) for f in _files(ds, method)]
            if not recs: continue
            vals.append(np.mean([r["comm_mb"] for r in recs])); labs.append(LABEL.get(method, method)); colors.append(COL[method])
        x=np.arange(len(vals)); ax.bar(x, vals, color=colors); ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Cumulative communication (MB)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(f"{OUT}/all_13_methods_final_communication.pdf")
    plt.close(fig); print("saved all_13_methods_final_communication.pdf")

def har_ablation():
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for tag, label, color in (("nofs","Without mask","#333333"),("fs","Shared mask","#D62728")):
        z=trajectories("har","proposed",tag=tag,metric="acc")
        if z is None: continue
        mean,sd,comm=z; rounds=np.arange(1,len(mean)+1)
        axes[0].plot(rounds,mean,label=label,color=color); axes[0].fill_between(rounds,mean-sd,mean+sd,color=color,alpha=.12)
        axes[1].plot(comm,mean,label=label,color=color); axes[1].fill_between(comm,mean-sd,mean+sd,color=color,alpha=.12)
    axes[0].set(xlabel="Communication round",ylabel="Test accuracy (%)")
    axes[1].set(xlabel="Cumulative 3-link communication (MB)",ylabel="Test accuracy (%)")
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/HAR_ablation_accuracy.pdf"); plt.close(fig)
    print("saved HAR_ablation_accuracy.pdf")

def cifar_figures():
    methods=("fedavg","powersgd","qsgd","proposed")
    fig,ax=plt.subplots(figsize=(6,4))
    for method in methods:
        z=trajectories("cifar",method,tag="",metric="acc")
        if z is None: continue
        mean,sd,comm=z
        ax.plot(comm,mean,color=COL[method],label=LABEL.get(method,method))
        ax.fill_between(comm,mean-sd,mean+sd,color=COL[method],alpha=.10)
    ax.set_xscale("log"); ax.set(xlabel="Cumulative 3-link communication (MB)",ylabel="Test accuracy (%)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/cifar10_accuracy_vs_communication.pdf"); plt.close(fig)

    means=[]; sds=[]; comms=[]; labs=[]
    for method in methods:
        recs=[json.load(open(f)) for f in _files("cifar",method,tag="")]
        if not recs: continue
        means.append(np.mean([r["final_acc"] for r in recs])); sds.append(np.std([r["final_acc"] for r in recs],ddof=1) if len(recs)>1 else 0)
        comms.append(np.mean([r["comm_mb"] for r in recs])); labs.append(LABEL.get(method,method))
    fig,axes=plt.subplots(1,2,figsize=(9,4)); x=np.arange(len(labs))
    axes[0].bar(x,means,yerr=sds,color=[COL[m] for m in methods[:len(labs)]],capsize=3); axes[0].set_ylabel("Test accuracy (%)")
    axes[1].bar(x,comms,color=[COL[m] for m in methods[:len(labs)]]); axes[1].set_yscale("log"); axes[1].set_ylabel("Cumulative communication (MB)")
    for ax in axes: ax.set_xticks(x); ax.set_xticklabels(labs,rotation=30,ha="right")
    fig.tight_layout(); fig.savefig(f"{OUT}/cifar10_four_methods.pdf"); plt.close(fig)
    print("saved CIFAR-10 figures")

if __name__ == "__main__":
    two_dataset_trajectory("acc", "all_13_methods_accuracy_vs_communication.pdf")
    two_dataset_trajectory("f1", "all_13_methods_f1_vs_communication.pdf")
    final_communication(); har_ablation(); cifar_figures()
