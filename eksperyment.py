"""
Składanie nauczonych modeli cząstek (wersja PyTorch, pod maszynę GitHuba).

Pytanie: czy model ELEKTRONU (uczony tylko na samotnym elektronie w losowych polach)
i model PROTONU (uczony tylko na polu pojedynczego protonu), złożone razem,
odtwarzają atom i jon H2+, których żaden z nich nie widział?

Świat: 1D, równanie Schrödingera, jednostki atomowe. Dokładny solver służy
wyłącznie jako nauczyciel (generuje przykłady) i sędzia (sprawdza wynik).

Warianty modelu elektronu (żaden nie widzi pola protonu):
  A - 30k przykładów, sieć 256-256-128
  C - 120k przykładów, sieć 512-512-256
  D - jak C, ale bardziej różnorodny świat (częściej 2-4 rozdzielone studnie o różnych kształtach)
  E - jak D, 300k przykładów, sieć 1024-1024-512

Użycie: python eksperyment.py C
"""
import json, sys, time
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(max(1, torch.get_num_threads()))
N, L, K = 128, 12.0, 4
x = np.linspace(-L, L, N); dx = x[1] - x[0]
T = (np.eye(N) - 0.5 * np.eye(N, k=1) - 0.5 * np.eye(N, k=-1)) / dx**2
Tt = torch.tensor(T)

def exact_batch(V):  # V: (B, N) numpy -> (B, K) poziomy energii
    H = Tt.unsqueeze(0) + torch.diag_embed(torch.tensor(V))
    return torch.linalg.eigvalsh(H)[:, :K].numpy()
exact = lambda V: exact_batch(np.atleast_2d(V))[0]
proton_field = lambda x0: -1.0 / np.sqrt((x - x0) ** 2 + 1.0)

def noise(rng):
    return sum(rng.normal(0, .05) * np.sin(k * np.pi * (x + L) / (2 * L) + rng.uniform(0, 2 * np.pi)) for k in range(1, 4))

def field_v1(rng):
    V = np.zeros(N)
    for _ in range(rng.integers(1, 4)):
        V += rng.uniform(-2, .6) * np.exp(-((x - rng.uniform(-5, 5)) ** 2) / (2 * rng.uniform(.4, 4) ** 2))
    return V + noise(rng)

def field_v2(rng):
    if rng.random() < 0.5:
        return field_v1(rng)
    V = np.zeros(N)
    for c in rng.uniform(-6, 6, rng.integers(2, 5)):
        w, d, p = rng.uniform(.4, 3), rng.uniform(-1.5, -.1), rng.uniform(1, 2.5)
        V += d * np.exp(-np.abs((x - c) / w) ** p)
    return V + noise(rng)

def make_data(gen, n, seed):
    rng = np.random.default_rng(seed)
    V = np.array([gen(rng) for _ in range(n)])
    E = np.concatenate([exact_batch(V[i:i + 2000]) for i in range(0, n, 2000)])
    return V.astype(np.float32), E.astype(np.float32)

def mlp(sizes, act=nn.ReLU):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])

def fit(model, X, Y, epochs, lr=1e-3, bs=256, log=""):
    """Trening z walidacją, spadkiem LR i zapamiętaniem najlepszych wag."""
    X, Y = torch.tensor(X), torch.tensor(Y)
    n = len(X); nv = max(1, n // 10); perm = torch.randperm(n)
    Xv, Yv, Xt, Yt = X[perm[:nv]], Y[perm[:nv]], X[perm[nv:]], Y[perm[nv:]]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = idx[i:i + bs]
            loss = nn.functional.mse_loss(model(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): v = nn.functional.mse_loss(model(Xv), Yv).item()
        if v < best: best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1: print(f"{log} epoka {ep} walidacja {v:.2e}", flush=True)
    model.load_state_dict(best_state)
    return model

CFG = {"A": (field_v1, 30_000, [256, 256, 128], 300),
       "C": (field_v1, 120_000, [512, 512, 256], 300),
       "D": (field_v2, 120_000, [512, 512, 256], 300),
       "E": (field_v2, 300_000, [1024, 1024, 512], 200)}

def main(variant):
    torch.manual_seed(0)
    gen, n, hidden, epochs = CFG[variant]
    t0 = time.time(); V, E = make_data(gen, n, 0); t_data = time.time() - t0
    Vte, Ete = make_data(gen, 2000, 99)
    mx, sx = V.mean(0), V.std(0) + 1e-6; my, sy = E.mean(0), E.std(0)

    t0 = time.time()
    el = fit(mlp([N, *hidden, K]), (V - mx) / sx, (E - my) / sy, epochs, log=f"[{variant}] elektron")
    t_train = time.time() - t0
    def electron(VV):
        with torch.no_grad():
            z = el(torch.tensor(((np.atleast_2d(VV) - mx) / sx).astype(np.float32))).numpy()
        return z * sy + my

    # model protonu: uczy się wyłącznie pola pojedynczego protonu
    rng = np.random.default_rng(1)
    P = rng.uniform(-6, 6, (4000, 1)).astype(np.float32)
    F = np.array([proton_field(p[0]) for p in P], dtype=np.float32)
    pr = fit(mlp([1, 128, 256, N], nn.Tanh), P / 6, F, 400, log=f"[{variant}] proton")
    def proton(x0):
        with torch.no_grad(): return pr(torch.tensor([[x0 / 6]], dtype=torch.float32)).numpy()[0]

    res = {"wariant": variant, "przyklady": n, "siec": hidden, "epoki": epochs,
           "czas_danych_s": round(t_data, 1), "czas_treningu_s": round(t_train, 1),
           "blad_na_wlasnym_swiecie": float(np.abs(electron(Vte) - Ete).mean()),
           "atom_MAE": float(np.mean([np.abs(electron(proton(x0))[0] - exact(proton_field(x0))) for x0 in [-2, 0, 1.5]]))}
    Rs = np.arange(0, 8.01, 0.5); rows = []
    for R in Rs:  # emulator: pola protonów się dodają
        Em = electron(proton(-R / 2) + proton(R / 2))[0]
        Ee = exact(proton_field(-R / 2) + proton_field(R / 2))
        rep = 1 / np.sqrt(R**2 + 1)
        rows.append([R, Em[0] + rep, Ee[0] + rep, Em[1] - Em[0], Ee[1] - Ee[0], *(Em - Ee)])
    rows = np.array(rows); far = Rs >= 5
    res.update({
        "H2_E0_MAE": float(np.abs(rows[:, 5]).mean()), "H2_E1_MAE": float(np.abs(rows[:, 6]).mean()),
        "R_min_model": float(Rs[rows[:, 1].argmin()]), "R_min_dokl": float(Rs[rows[:, 2].argmin()]),
        "glebokosc_wiazania_model": float(rows[-1, 1] - rows[:, 1].min()),
        "glebokosc_wiazania_dokl": float(rows[-1, 2] - rows[:, 2].min()),
        "tunel_MAE_R5_8": float(np.abs(rows[far, 3] - rows[far, 4]).mean()),
        "tunel_model_R5_8": rows[far, 3].round(3).tolist(), "tunel_dokl_R5_8": rows[far, 4].round(3).tolist(),
        "R": Rs.tolist(), "krzywa_model": rows[:, 1].round(4).tolist(), "krzywa_dokl": rows[:, 2].round(4).tolist()})
    json.dump(res, open(f"wynik_{variant}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith(("krzywa", "R"))}, indent=1))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
