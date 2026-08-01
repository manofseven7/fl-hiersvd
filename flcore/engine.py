import copy, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score
from . import utils as U, feature_selection as FS
from .methods import FLMethod, FedKDMethod

class CommTracker:
    def __init__(self): self.c2e = 0; self.e2c = 0; self.down = 0; self.setup = 0
    def total(self): return self.c2e + self.e2c + self.down + self.setup
    def mb(self): return self.total() / (1024 ** 2)

def _evaluate(model, loader, device):
    model.eval(); cor = tot = 0; ys = []; ps = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x).argmax(1)
            cor += (pred == y).sum().item(); tot += y.size(0)
            ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return 100.0 * cor / tot, 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)

def train(cfg, client_loader, test_loader, d, C_out, is_img, client_idxs, log):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    from .models import build_model
    tmpl = build_model(cfg, d, C_out).to(device)
    view = U.to_np(tmpl.state_dict())
    method = (FedKDMethod(cfg, d, C_out) if cfg.method == "fedkd"
              else FLMethod(cfg, tmpl))

    M, N = cfg.edges, cfg.clients
    groups = [list(map(int, g)) for g in np.array_split(np.arange(N), M)]
    comm = CommTracker(); traj = []; mask = None

    # SCAFFOLD transmission model (single source of truth; see configs/presets.py)
    full_cv = (cfg.method == "scaffold" and
               getattr(cfg, "scaffold_variant", "full") == "full")

    for r in range(1, cfg.rounds + 1):
        round_start = copy.deepcopy(view)
        sel = (sorted(np.random.choice(N, cfg.fedpaq_k, replace=False).tolist())
               if cfg.method == "fedpaq" else list(range(N)))

        # ---- client train + uplink (model part) ----
        uplinks, sizes, nsteps, cv_deltas = {}, {}, {}, {}
        round_c2e_model = 0; round_power_p_broadcast = 0
        for i in sel:
            loader, ni = client_loader(i)
            if cfg.method == "fedkd":
                method.ensure_teacher(i, loader, device); upd = method.client_train(view, loader, i)
            else:
                upd, ns, _ = method.client_train(view, loader, i, mask=mask); nsteps[i] = ns
                if cfg.method == "scaffold":
                    cv_deltas[i] = method.scaffold_update_cv(upd, ns, i)
            pay, nb = method.uplink(upd, i, mask=mask)
            if cfg.method == "powersgd":
                from .compression import powersgd_p_bytes
                round_power_p_broadcast += powersgd_p_bytes(pay)
            uplinks[i] = (pay, upd); sizes[i] = ni; round_c2e_model += nb
        if full_cv:                       # client also transmits Δc_i (== model shape)
            round_c2e_model *= 2
        comm.c2e += round_c2e_model
        comm.down += round_power_p_broadcast

        fednova_tau = None
        if cfg.method == "fednova":       # normalise by tau, restore weighted effective tau globally
            total_n = sum(sizes.values())
            fednova_tau = sum((sizes[i] / total_n) * max(nsteps[i], 1) for i in sel)
            for i in sel:
                pay, upd = uplinks[i]
                tau = max(nsteps[i], 1)
                uplinks[i] = (_scale_payload(pay, cfg.method, 1.0 / tau), upd)

        # ---- edge aggregate + recompress (model part) ----
        edge_pay, edge_sizes, round_e2c_model = [], [], 0
        for edge_id, g in enumerate(groups):
            gsel = [i for i in g if i in sel]
            if not gsel: continue
            ep, eb = method.edge_aggregate([uplinks[i][0] for i in gsel],
                                           [sizes[i] for i in gsel], edge_id=edge_id)
            edge_pay.append(ep); edge_sizes.append(sum(sizes[i] for i in gsel))
            round_e2c_model += eb
        if full_cv:                       # edge also forwards aggregated Δc (== agg shape)
            round_e2c_model *= 2
        comm.e2c += round_e2c_model

        # ---- central aggregate + downlink (scaffold: nb already = model + server cv) ----
        dpay, db = method.central_aggregate(edge_pay, edge_sizes)
        if fednova_tau is not None:
            dpay = _scale_payload(dpay, "fednova", fednova_tau)
        # The paper counts the complete central-to-client delivery, not a
        # single logical broadcast object. FedPAQ sends only to sampled clients.
        comm.down += db * len(sel)

        view = method.apply_downlink(view, dpay)

        # ---- SCAFFOLD control-variate bookkeeping (NO comm accounting here) ----
        if cfg.method == "scaffold":
            method.scaffold_server_cv([cv_deltas[i] for i in sel],
                                      [sizes[i] for i in sel])

        # ---- coordinated feature selection at T_w (HAR only) ----
        if cfg.feature_selection and mask is None and r == cfg.Tw:
            stats, szs = []; first_key = next(iter(view))
            for i in range(N):
                loader, ni = client_loader(i); Xb = _grab_inputs(loader, 512)
                if i not in uplinks:
                    raise RuntimeError("Feature-mask construction requires full participation at T_w")
                warmup_update = uplinks[i][1]
                G = FS.model_importance(round_start[first_key] + warmup_update[first_key],
                                        round_start[first_key])
                stats.append((G, FS.marginal_entropy(Xb),
                              FS.correlation_redundancy(Xb, seed=cfg.seed * 1000 + i)))
                szs.append(ni)
            mask = FS.build_shared_mask(stats, szs, cfg.rho, w=cfg.fs_weights)
            comm.setup += FS.setup_bytes(N, len(mask))
            view[first_key] = view[first_key] * mask[None, :]
            method.active_mask = mask

        # ---- evaluate ----
        tmpl.load_state_dict(U.from_np(view, tmpl.state_dict()))
        acc, f1 = _evaluate(tmpl, test_loader, device)
        traj.append(dict(round=r, acc=acc, f1=f1, comm_mb=comm.mb()))
        if r % 5 == 0 or r == cfg.rounds:
            log.info(f"[{cfg.method:14s}|{cfg.dataset:7s}|s{cfg.seed}] r={r:3d} "
                     f"acc={acc:5.2f}  f1={f1:5.2f}  comm={comm.mb():8.2f} MB")
    return traj, comm

def _scale_payload(pay, method, s):
    out = {}
    for n, it in pay.items():
        if it[0] == "dense": out[n] = ("dense", it[1] * s)
        elif it[0] == "svd": out[n] = ("svd", it[1], it[2]*s, it[3], it[4])
        else: out[n] = it
    return out

def _grab_inputs(loader, max_samples):
    xs = []
    for b, (x, y) in enumerate(loader):
        xs.append(x.numpy())
        if sum(len(z) for z in xs) >= max_samples: break
    return np.concatenate(xs, 0)[:max_samples]
