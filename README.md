# composable-particle-models

**Pytanie:** czy model, który nauczył się tylko zachowania elektronu, i model, który nauczył się tylko pola protonu, złożone razem, odtwarzają atom i cząsteczkę, których żaden z nich nigdy nie widział?

To test zasady „budowania z nauczonych klocków zamiast uczenia całości”.

## Jak to działa

- **Świat:** 1 wymiar, mechanika kwantowa (równanie Schrödingera, jednostki atomowe).
- **Nauczyciel i sędzia:** dokładny solver. Generuje przykłady do nauki i sprawdza wynik końcowy. Nie jest częścią modeli.
- **Model elektronu:** sieć neuronowa `pole V(x) → 4 najniższe poziomy energii`. Uczy się wyłącznie samotnego elektronu w losowych polach i nigdy nie widzi pola protonu.
- **Model protonu:** sieć neuronowa `położenie → pole, które wytwarza`. Uczy się tylko pojedynczego protonu.
- **Emulator:** dodaje pola protonów (zasada superpozycji). To jedyna reguła fizyki wpisana ręcznie.
- **Test:** atom (1 proton) i jon H2+ (2 protony w odległości 0–8). Sprawdzamy poziomy energii, krzywą wiązania i tunelowanie.

## Dotychczasowe wyniki (lokalnie, scikit-learn)

| | Eksp. 1 (trening 20 s) | Trening do końca |
|---|---|---|
| Błąd poziomów atomu | ~0,010 | 0,004 |
| H2+, poziom podstawowy | 0,028 | 0,011 |
| H2+, 1. poziom wzbudzony | 0,045 | 0,010 |
| Odległość wiązania (prawdziwa 2,0) | 3,5 | 2,0 |

Otwarty problem: **tunelowanie** przy dalekich protonach jest niestabilne. Warianty C–E sprawdzają, czy rozwiązuje je więcej danych, większa sieć albo bardziej różnorodne doświadczenie elektronu.

## Uruchomienie

- Na GitHubie: zakładka **Actions → eksperyment → Run workflow**. Uruchamia się też sam po zmianie `eksperyment.py`. Wyniki są w artefaktach `wynik_*.json`.
- Lokalnie: `pip install numpy torch`, potem `python eksperyment.py C`.

## Pliki

- `eksperyment.py`: wersja PyTorch, warianty A, C, D, E
- `eksperyment1_sklearn.py`, `eksperyment2_sklearn.py`: pierwotne wersje lokalne
- `wyniki_lokalne/`: wyniki z tych wersji
