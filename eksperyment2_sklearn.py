"""
Eksperyment 2: czy problemem był za krótki trening, czy za mało różnorodne doświadczenie elektronu?

Warianty modelu elektronu (wszystkie BEZ pola protonu w treningu):
  A  - jak w eksp. 1: 30k przykładów, sieć 256-256-128, krótki trening (wczesne zatrzymanie)
  B  - te same dane, ta sama sieć, trening ~10x dłuższy
  C  - 120k przykładów, większa sieć 512-512-256, długi trening
  D  - jak C, ale bardziej różnorodny świat: częściej 2-4 rozdzielone studnie
       (gaussy o losowych kształtach - nadal żadnego kształtu Coulomba/protonu)
Model protonu i emulator (dodawanie pól) bez zmian.
"""
import numpy as np, json, time, sys
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

N, L, K = 128, 12.0, 4
x = np.linspace(-L, L, N); dx = x[1] - x[0]
T = (np.eye(N) - 0.5 * np.eye(N, k=1) - 0.5 * np.eye(N, k=-1)) / dx**2
exact = lambda V: np.linalg.eigvalsh(T + np.diag(V))[:K]
proton_field = lambda x0: -1.0 / np.sqrt((x - x0) ** 2 + 1.0)

def field_v1(rng):
    V = np.zeros(N)
    for _ in range(rng.integers(1, 4)):
        V += rng.uniform(-2, .6) * np.exp(-((x - rng.uniform(-5, 5)) ** 2) / (2 * rng.uniform(.4, 4) ** 2))
    for k in range(1, 4):
        V += rng.normal(0, .05) * np.sin(k * np.pi * (x + L) / (2 * L) + rng.uniform(0, 2 * np.pi))
    return V

def field_v2(rng):
    if rng.random() < 0.5:
        return field_v1(rng)
    # kilka rozdzielonych studni o losowych (nie-coulombowskich) profilach
    V = np.zeros(N)
    n = rng.integers(2, 5)
    centers = rng.uniform(-6, 6, n)
    for c in centers:
        w = rng.uniform(.4, 3); d = rng.uniform(-1.5, -.1); p = rng.uniform(1, 2.5)  # p: kształt (gauss p=2, inne)
        V += d * np.exp(-np.abs((x - c) / w) ** p)
    for k in range(1, 4):
        V += rng.normal(0, .05) * np.sin(k * np.pi * (x + L) / (2 * L) + rng.uniform(0, 2 * np.pi))
    return V

def data(gen, n, seed):
    rng = np.random.default_rng(seed)
    V = np.array([gen(rng) for _ in range(n)])
    return V, np.array([exact(v) for v in V])

def train(V, E, hidden, max_iter, patience):
    sx, sy = StandardScaler().fit(V), StandardScaler().fit(E)
    m = MLPRegressor(hidden_layer_sizes=hidden, learning_rate_init=1e-3, batch_size=256,
                     max_iter=max_iter, early_stopping=True, n_iter_no_change=patience,
                     tol=1e-6, random_state=0)
    t0 = time.time(); m.fit(sx.transform(V), sy.transform(E))
    pred = lambda VV: sy.inverse_transform(m.predict(sx.transform(np.atleast_2d(VV))))
    return pred, time.time() - t0, m.n_iter_

# model protonu
rng = np.random.default_rng(1)
P = rng.uniform(-6, 6, 4000)[:, None]
proton = MLPRegressor(hidden_layer_sizes=(128, 256), activation="tanh", max_iter=800,
                      early_stopping=True, n_iter_no_change=40, random_state=0).fit(P / 6, [proton_field(p[0]) for p in P])
pp = lambda x0: proton.predict([[x0 / 6]])[0]

def evaluate(pred):
    out = {}
    out["atom_MAE"] = float(np.mean([np.abs(pred(pp(x0))[0] - exact(proton_field(x0))) for x0 in [-2, 0, 1.5]]))
    Rs = np.arange(0, 8.01, 0.5); rows = []
    for R in Rs:
        Em = pred(pp(-R / 2) + pp(R / 2))[0]; Ee = exact(proton_field(-R / 2) + proton_field(R / 2))
        rep = 1 / np.sqrt(R**2 + 1)
        rows.append([R, Em[0] + rep, Ee[0] + rep, Em[1] - Em[0], Ee[1] - Ee[0], *(Em - Ee)])
    rows = np.array(rows)
    out["H2_E0_MAE"] = float(np.mean(np.abs(rows[:, 5])))
    out["H2_E1_MAE"] = float(np.mean(np.abs(rows[:, 6])))
    out["R_min_model"] = float(Rs[rows[:, 1].argmin()]); out["R_min_dokl"] = float(Rs[rows[:, 2].argmin()])
    out["glebokosc_wiazania_model"] = float(rows[-1, 1] - rows[:, 1].min())
    out["glebokosc_wiazania_dokl"] = float(rows[-1, 2] - rows[:, 2].min())
    far = Rs >= 5
    out["tunel_rozszczepienie_R5-8_model"] = rows[far, 3].round(3).tolist()
    out["tunel_rozszczepienie_R5-8_dokl"] = rows[far, 4].round(3).tolist()
    out["krzywa_model"] = rows[:, 1].round(3).tolist(); out["krzywa_dokl"] = rows[:, 2].round(3).tolist()
    return out

variant = sys.argv[1]
cfg = {"A": ("v1", 30000, (256, 256, 128), 120, 12),
       "B": ("v1", 30000, (256, 256, 128), 1500, 60),
       "C": ("v1", 120000, (512, 512, 256), 600, 40),
       "D": ("v2", 120000, (512, 512, 256), 600, 40)}[variant]
gen = field_v1 if cfg[0] == "v1" else field_v2
t0 = time.time(); V, E = data(gen, cfg[1], 0); tg = time.time() - t0
Vt, Et = data(gen, 2000, 99)
pred, tt, it = train(V, E, cfg[2], cfg[3], cfg[4])
res = {"wariant": variant, "przyklady": cfg[1], "siec": cfg[2], "czas_danych_s": round(tg, 1),
       "czas_treningu_s": round(tt, 1), "epoki": it,
       "blad_na_wlasnym_swiecie": float(np.mean(np.abs(pred(Vt) - Et))), **evaluate(pred)}
json.dump(res, open(f"/home/claude/wynik_{variant}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("krzywa")}, indent=1))
