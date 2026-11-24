"""
Eksperyment: czy dwa osobno wytrenowane modele (elektron i proton),
złożone razem, poprawnie przewidują stany układu, którego żaden z nich nie widział?

Świat: 1 wymiar, mechanika kwantowa (równanie Schrödingera, jednostki atomowe).
Nauczyciel (tylko do generowania przykładów i sprawdzania): dokładny solver.

Model ELEKTRONU: sieć neuronowa, wejście = pole V(x) w 128 punktach,
                 wyjście = 4 najniższe poziomy energii elektronu w tym polu.
                 Trening WYŁĄCZNIE na losowych polach (sumy gaussów + gładki szum).
                 Nigdy nie widzi pola protonu ani atomu.
Model PROTONU:   sieć neuronowa, wejście = położenie protonu,
                 wyjście = pole, które wytwarza (128 punktów).
                 Trening wyłącznie na pojedynczym protonie.
Emulator:        pola protonów się dodają (zasada superpozycji), wynik idzie do elektronu.
Test:            atom (1 proton) i jon H2+ (2 protony w różnych odległościach) -
                 porównanie z dokładnym rozwiązaniem.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import json, time

rng = np.random.default_rng(0)
N, L = 128, 12.0
x = np.linspace(-L, L, N)
dx = x[1] - x[0]
K = 4  # liczba poziomów energii

# ---------- nauczyciel: dokładny solver ----------
T = (np.diag(np.full(N, 1.0)) - 0.5 * np.diag(np.ones(N - 1), 1) - 0.5 * np.diag(np.ones(N - 1), -1)) / dx**2
def exact_levels(V):
    return np.linalg.eigvalsh(T + np.diag(V))[:K]

def proton_field(x0, a=1.0):  # prawdziwe pole protonu (miękki Coulomb)
    return -1.0 / np.sqrt((x - x0) ** 2 + a**2)

# ---------- dane dla elektronu: losowe pola, BEZ kształtu Coulomba ----------
def random_field():
    V = np.zeros(N)
    for _ in range(rng.integers(1, 4)):
        depth = rng.uniform(-2.0, 0.6)
        width = rng.uniform(0.4, 4.0)
        c = rng.uniform(-5, 5)
        V += depth * np.exp(-((x - c) ** 2) / (2 * width**2))
    for k in range(1, 4):  # gładki szum
        V += rng.normal(0, 0.05) * np.sin(k * np.pi * (x + L) / (2 * L) + rng.uniform(0, 2 * np.pi))
    return V

t0 = time.time()
n_train = 30000
Vtr = np.array([random_field() for _ in range(n_train)])
Etr = np.array([exact_levels(V) for V in Vtr])
Vte = np.array([random_field() for _ in range(2000)])
Ete = np.array([exact_levels(V) for V in Vte])
print(f"dane elektronu: {time.time()-t0:.1f}s")

sx, sy = StandardScaler().fit(Vtr), StandardScaler().fit(Etr)
electron = MLPRegressor(hidden_layer_sizes=(256, 256, 128), activation="relu",
                        learning_rate_init=1e-3, batch_size=256, max_iter=120,
                        early_stopping=True, n_iter_no_change=12, random_state=0)
t0 = time.time()
electron.fit(sx.transform(Vtr), sy.transform(Etr))
print(f"trening elektronu: {time.time()-t0:.1f}s, iteracje {electron.n_iter_}")
def electron_predict(V):
    return sy.inverse_transform(electron.predict(sx.transform(np.atleast_2d(V))))

# ---------- dane dla protonu: tylko pojedynczy proton ----------
P_pos = rng.uniform(-6, 6, 4000)[:, None]
P_field = np.array([proton_field(p[0]) for p in P_pos])
proton = MLPRegressor(hidden_layer_sizes=(128, 256), activation="tanh",
                      learning_rate_init=1e-3, max_iter=600, random_state=0,
                      early_stopping=True, n_iter_no_change=30)
proton.fit(P_pos / 6.0, P_field)
def proton_predict(x0):
    return proton.predict(np.array([[x0 / 6.0]]))[0]

# ---------- ocena ----------
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
res = {}
res["elektron_losowe_pola_MAE"] = mae(electron_predict(Vte), Ete)
res["proton_pole_MAE"] = mae([proton_predict(p) for p in [-3, 0, 2.5]],
                             [proton_field(p) for p in [-3, 0, 2.5]])

# test 1: atom wodoru (1 proton), nigdy niewidziany przez elektron
atom = {}
for x0 in [-2.0, 0.0, 1.5]:
    Vm = proton_predict(x0)
    atom[x0] = {"model": electron_predict(Vm)[0].round(4).tolist(),
                "dokladnie": exact_levels(proton_field(x0)).round(4).tolist()}
res["atom"] = atom

# test 2: jon H2+ (2 protony) - złożenie trzech modeli
R_list = np.round(np.arange(0.0, 8.01, 0.5), 2)
h2 = []
for R in R_list:
    Vm = proton_predict(-R / 2) + proton_predict(R / 2)        # emulator: pola się dodają
    Ve = proton_field(-R / 2) + proton_field(R / 2)
    Em, Ee = electron_predict(Vm)[0], exact_levels(Ve)
    rep = 1.0 / np.sqrt(R**2 + 1.0)                             # odpychanie protonów
    h2.append({"R": float(R), "E0_model": float(Em[0]), "E0_dokl": float(Ee[0]),
               "E1_model": float(Em[1]), "E1_dokl": float(Ee[1]),
               "calk_model": float(Em[0] + rep), "calk_dokl": float(Ee[0] + rep)})
res["h2plus"] = h2

# kontrola: czy elektron nie "rozpoznaje" pola protonu jako czegoś znanego?
# odległość pola protonu od najbliższego pola treningowego
Vp = proton_field(0.0)
d = np.min(np.sqrt(np.mean((Vtr - Vp) ** 2, axis=1)))
d_typ = np.median([np.min(np.sqrt(np.mean((Vtr[:3000] - v) ** 2, axis=1))) for v in Vte[:200]])
res["odl_pola_protonu_od_treningu"] = float(d)
res["typowa_odl_testu_od_treningu"] = float(d_typ)

json.dump(res, open("/home/claude/wyniki.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "h2plus"}, indent=1, ensure_ascii=False))
for r in h2: print(r)
