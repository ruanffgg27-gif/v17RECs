"""Argila estimada pela CTC quando o laudo não traz a análise granulométrica.

A argila só entra na conta do gesso (S alvo por classe de argila e fator argila/100). Sem granulometria,
estima-se a argila pela CTC a pH 7, que nos solos da região acompanha bem o teor de argila:

    Argila (g/kg) = −185,96 + 6,572 × CTC (mmolc/dm³)

Ajuste com 2.085 amostras da base da equipe (planilha de relação calcário × atributos; CTC = Ca + Mg +
H+Al + 2,5): R² = 0,78; erro típico ≈ 100 g/kg; a classe de argila usada no gesso (< 200 / 200–400 /
> 400 g/kg) é acertada em ~75% das amostras (validação cruzada). É menos preciso que a análise — por isso
cada valor estimado fica marcado na planilha e no book.

Quando o laudo tem argila em parte das amostras (≥ 10), a relação é recalibrada com essas amostras
(regressão linear local), desde que o ajuste local seja razoável (R² ≥ 0,4 e inclinação positiva); se não
for, a equação regional é deslocada pelo viés médio dessas amostras (≥ 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COEF_REGIONAL = (-185.96, 6.572)
LIMITES = (60.0, 750.0)
K_PADRAO = 2.5          # mmolc/dm³, quando a CTC é recomposta sem K no laudo

GRANULOMETRIA = ["argila", "areia", "silte"]
MICRONUTRIENTES = ["b", "zn", "cu", "fe", "mn"]
NOMES = {"argila": "Argila", "areia": "Areia", "silte": "Silte", "b": "B", "zn": "Zn", "cu": "Cu", "fe": "Fe",
         "mn": "Mn", "p": "P", "k": "K", "s": "S-SO₄", "al": "Al", "mo": "M.O.", "ph": "pH", "ctc": "CTC",
         "ca": "Ca", "mg": "Mg", "hal": "H+Al"}


def ctc_amostras(d: pd.DataFrame) -> pd.Series:
    """CTC do laudo; se faltar, recompõe Ca + Mg + H+Al + K."""
    k = d["k"] if "k" in d else pd.Series(np.nan, index=d.index)
    rec = d["ca"] + d["mg"] + d["hal"] + k.fillna(K_PADRAO)
    if "ctc" in d:
        return d["ctc"].where(d["ctc"].notna() & (d["ctc"] > 0), rec)
    return rec


def completar_argila(dados: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Preenche a argila que falta pela CTC. Devolve (dados, descrição do método ou None)."""
    d = dados
    if "argila" not in d:
        d["argila"] = np.nan
    falta = d["argila"].isna()
    d["argila_estimada"] = False
    if not falta.any():
        return d, None
    ctc = ctc_amostras(d)
    a, b = COEF_REGIONAL
    metodo = "equação regional: Argila = −186 + 6,57 × CTC"
    medida = ~falta & ctc.notna()
    if medida.sum() >= 10 and ctc[medida].std() > 0:
        bl, al = np.polyfit(ctc[medida], d.loc[medida, "argila"], 1)
        pred = al + bl * ctc[medida]
        ss = ((d.loc[medida, "argila"] - d.loc[medida, "argila"].mean()) ** 2).sum()
        r2 = 1 - ((d.loc[medida, "argila"] - pred) ** 2).sum() / ss if ss > 0 else 0
        if bl > 0 and r2 >= 0.4:
            a, b = al, bl
            metodo = f"regressão local com o próprio laudo ({int(medida.sum())} amostras com argila; R² = {r2:.2f})"
    if metodo.startswith("equação") and medida.sum() >= 5:
        # sem boa regressão local: corrige o viés da equação regional com as amostras medidas
        vies = float((d.loc[medida, "argila"] - (a + b * ctc[medida])).mean())
        a += vies
        metodo = f"equação regional corrigida pelas {int(medida.sum())} amostras com argila ({vies:+.0f} g/kg)"
    est = (a + b * ctc).clip(*LIMITES)
    preencher = falta & est.notna()
    d.loc[preencher, "argila"] = est[preencher].round(0)
    d.loc[preencher, "argila_estimada"] = True
    return d, metodo


def ausentes(dados: pd.DataFrame, mapa: dict) -> list[str]:
    """Atributos que não vieram no laudo (sem coluna ou coluna vazia)."""
    return [c for c in NOMES if c not in mapa or c not in dados or dados[c].isna().all()]


def mensagem_ausentes(faltam: list[str], metodo_argila: str | None, n_est: int, n_total: int) -> list[str]:
    """Avisos legíveis sobre o que falta no laudo e o que o sistema fez."""
    msgs = []
    gran = [NOMES[c] for c in GRANULOMETRIA if c in faltam]
    micro = [NOMES[c] for c in MICRONUTRIENTES if c in faltam]
    outros = [NOMES[c] for c in faltam if c not in GRANULOMETRIA + MICRONUTRIENTES]
    if metodo_argila and n_est:
        parte = "todas as amostras" if n_est == n_total else f"{n_est} de {n_total} amostras"
        msgs.append(f"Argila ausente em {parte}: estimada pela CTC ({metodo_argila}). Usada só no cálculo do "
                    "gesso; valores marcados como estimados na planilha e no book.")
    elif gran:
        msgs.append(f"Granulometria incompleta no laudo: sem {', '.join(gran)}.")
    if micro:
        msgs.append(f"Micronutrientes ausentes no laudo ({', '.join(micro)}): ficam fora da interpretação e do book.")
    if outros:
        msgs.append(f"Também não constam no laudo: {', '.join(outros)}.")
    return msgs
