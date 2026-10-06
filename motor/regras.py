"""Regras de recomendação — funções puras e determinísticas.

Cada função recebe os atributos de UMA amostra (já em unidades padrão:
Ca, Mg, K, H+Al, Al em mmolc/dm³; P e S em mg/dm³; argila em g/kg) e
devolve um Resultado com a dose, o critério que a definiu e observações.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .parametros import Calcario, Gesso, Fosforo, Potassio

@dataclass
class Resultado:
    dose: float | None
    criterio: str = ""
    obs: list[str] = field(default_factory=list)
    bruto: float | None = None   # valor da equação antes de limites/arredondamento


def _ok(*vals) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


def arredondar(x: float | None, passo: int) -> float | None:
    if x is None or passo <= 0:
        return x
    return float(round(x / passo) * passo)


# ----------------------------------------------------------------- calcário
def fatores_calcario(p: Calcario) -> dict[str, float]:
    """Fatores de correção da dose quando o calcário difere do de referência.

    Segue a conta usual de equilíbrio de bases:
        dose = necessidade do nutriente ÷ teor do óxido × 100/PRNT
    Logo, em relação ao calcário de referência:
    - Critério Ca: × (CaO_ref / CaO) × (PRNT_ref / PRNT)
    - Critério Mg: × (MgO_ref / MgO) × (PRNT_ref / PRNT)
    - Termo do Al (neutralização): × (PRNT_ref / PRNT)
    (O coeficiente 345 do Mg equivale a ≈ 24 kg Mg/ha por mmolc ÷ 14,9% MgO
    ÷ 0,82, o que confirma essa forma de cálculo.)
    Com o calcário de referência (33% CaO, 14,9% MgO, PRNT 82) todos valem 1.
    """
    f_prnt = p.ref_prnt / p.prnt if p.prnt > 0 else float("nan")
    return {
        "ca": p.ref_cao / p.cao if p.cao > 0 else float("inf"),
        "mg": p.ref_mgo / p.mgo if p.mgo > 0 else float("inf"),
        "prnt": f_prnt,
    }


def calcario(ca, mg, hal, al, k=None, abertura: bool = False,
             p: Calcario | None = None) -> Resultado:
    p = p or Calcario()
    if not _ok(ca, mg, hal):
        return Resultado(None, "sem dados", ["Calcário: falta Ca, Mg ou H+Al no laudo"])
    al = al if _ok(al) else 0.0
    obs: list[str] = []

    k_ctc = p.k_ctc
    if p.usar_k_medido:
        if _ok(k):
            k_ctc = k
        else:
            obs.append(f"Calcário: K ausente, usado K = {p.k_ctc} na CTC*")
    ctc = ca + mg + hal + k_ctc

    f = fatores_calcario(p)
    termo_al = p.c_al * al * f["prnt"]
    cand = {}
    if p.cao > 0:
        cand["Ca"] = p.a_ca * (p.alvo_ca * ctc - ca) * f["ca"] * f["prnt"] + termo_al
    if p.mgo > 0:
        cand["Mg"] = p.b_mg * (p.alvo_mg * ctc - mg) * f["mg"] * f["prnt"] + termo_al
    else:
        obs.append("Calcário sem MgO: critério Mg desconsiderado")

    criterio, bruto = max(cand.items(), key=lambda kv: kv[1])
    dose = bruto

    def limitar(x):
        nonlocal criterio
        if x < p.dose_min:
            criterio = f"mínimo ({criterio})" if bruto > 0 else "mínimo"
            return p.dose_min
        if x > p.dose_max:
            criterio = f"máximo ({criterio})"
            obs.append(f"Calcário: equação indicou {bruto:.0f} kg/ha, limitado a {p.dose_max:.0f}")
            return p.dose_max
        return x

    if abertura and p.abertura_antes_dos_limites:
        dose = limitar(dose * p.fator_abertura)
    else:
        dose = limitar(dose)
        if abertura:
            dose *= p.fator_abertura
    if abertura:
        criterio += f" ×{p.fator_abertura:g} abertura"
    return Resultado(dose, criterio, obs, bruto)


# -------------------------------------------------------------------- gesso
def s_alvo(argila: float, p: Gesso) -> float:
    if argila < p.limite_arenoso:
        return p.s_alvo_arenoso
    if argila <= p.limite_medio:
        return p.s_alvo_medio
    return p.s_alvo_argiloso


def gesso(s, argila, p: Gesso | None = None) -> Resultado:
    p = p or Gesso()
    if not _ok(s, argila):
        return Resultado(None, "sem dados", ["Gesso: falta S ou argila no laudo"])
    alvo = s_alvo(argila, p)
    bruto = ((alvo - s) * 1000 / 75) * argila / 100
    obs, crit = [], f"S alvo {alvo:g}"
    dose = bruto
    if bruto < p.dose_min:
        dose, crit = p.dose_min, crit + " → mínimo operacional"
        if s >= alvo:
            obs.append(f"Gesso: S ({s:g}) já ≥ alvo ({alvo:g}); mantida dose mínima de {p.dose_min:.0f}")
    elif bruto > p.dose_max:
        dose, crit = p.dose_max, crit + " → limitado ao máximo"
        obs.append(f"Gesso: equação indicou {bruto:.0f} kg/ha, limitado a {p.dose_max:.0f}")
    return Resultado(dose, crit, obs, bruto)


# ------------------------------------------------------------------ fósforo
def p2o5(p_resina, p: Fosforo | None = None) -> Resultado:
    p = p or Fosforo()
    if not _ok(p_resina):
        return Resultado(None, "sem dados", ["P2O5: falta P no laudo"])
    if p_resina <= 0:
        return Resultado(p.dose_max, "máximo", [f"P2O5: P = {p_resina} (≤ 0); aplicada dose máxima"], None)
    bruto = p.coef * p_resina ** p.expoente
    dose, crit = bruto, "equação"
    if bruto < p.dose_min:
        dose, crit = p.dose_min, "mínimo"
    elif bruto > p.dose_max:
        dose, crit = p.dose_max, "máximo"
    return Resultado(dose, crit, [], bruto)


# ----------------------------------------------------------------- potássio
def kcl(k, p: Potassio | None = None) -> Resultado:
    p = p or Potassio()
    if not _ok(k):
        return Resultado(None, "sem dados", ["KCl: falta K no laudo"])
    if k <= 0:
        return Resultado(None, "K inválido", [f"KCl: K = {k} (≤ 0); verifique o laudo"])
    bruto = p.coef * k ** p.expoente
    dose, crit = bruto, "equação"
    if bruto < p.dose_min:
        dose, crit = p.dose_min, "mínimo"
    elif p.dose_max is not None and bruto > p.dose_max:
        dose, crit = p.dose_max, "máximo"
    return Resultado(dose, crit, [], bruto)


# ------------------------------------------------------- S elementar e fórmulas
def s_elementar(dose_gesso, coef: float = 4.1904, expoente: float = 0.3754) -> float | None:
    """Dose de S elementar que substitui a dose de gesso recomendada."""
    if not _ok(dose_gesso) or dose_gesso <= 0:
        return None
    return coef * dose_gesso ** expoente


def ler_formula(txt: str) -> tuple[float, float, float] | None:
    """'11-52-00' → (11, 52, 0). Aceita espaços, pontos, vírgulas decimais e '/'."""
    nums = re.findall(r"\d+(?:[.,]\d+)?", str(txt or ""))
    if len(nums) < 2:
        return None
    vals = [float(n.replace(",", ".")) for n in nums[:3]] + [0.0] * (3 - len(nums[:3]))
    return tuple(vals)  # type: ignore[return-value]


def kcl_equivalente(dose_produto_p, k2o_pct: float, k2o_kcl: float = 60.0):
    """KCl (kg/ha) equivalente ao K2O que a formulação de P já fornece.

    Ex.: 200 kg/ha de 00-30-10 → 20 kg/ha de K2O → 20 / 0,60 = 33,3 kg/ha de KCl a descontar."""
    return dose_produto_p * (k2o_pct / 100) / (k2o_kcl / 100)
