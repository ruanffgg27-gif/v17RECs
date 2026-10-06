"""Vento e janelas de pulverização no local (NASA POWER, grade de ~50 km, hora local).

O vento vem de reanálise (MERRA-2) numa grade regional: descreve o regime típico da região — direção
predominante, meses e horários mais ventosos — e é o mesmo para talhões vizinhos. Não existe variação de
vento DENTRO do talhão nessa escala; quebra-ventos, matas e relevo alteram o vento localmente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from . import fontes
from .base import Area

SETORES = ["N", "NNE", "NE", "ENE", "L", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
NOME_SETOR = {"N": "norte", "NE": "nordeste", "L": "leste", "SE": "sudeste", "S": "sul", "SO": "sudoeste",
              "O": "oeste", "NO": "noroeste"}
CLASSES = [("< 3 km/h", 0, 3, "#C6DBEF"), ("3–10 km/h", 3, 10, "#41AB5D"), ("10–15 km/h", 10, 15, "#FDAE61"),
           ("> 15 km/h", 15, 999, "#D7301F")]
CRITERIOS = {"vento_min": 3.0, "vento_max": 10.0, "temp_max": 30.0, "ur_min": 55.0}
HORAS = list(range(5, 21))
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


@dataclass
class Vento:
    fonte: str
    periodo: tuple[int, int]
    horario: bool                                                     # True = dados horários (janelas disponíveis)
    rosa: pd.DataFrame = field(default_factory=pd.DataFrame)          # setor × classe (% do tempo)
    mensal: pd.DataFrame = field(default_factory=pd.DataFrame)        # mês: média, p90 (km/h), % ventoso
    janela: pd.DataFrame | None = None                                # mês × hora: % das horas adequadas
    predominante: str = ""
    predominante_pct: float = 0.0
    graus_predominante: float = 0.0
    media_kmh: float = 0.0
    calmaria_pct: float = 0.0
    avisos: list[str] = field(default_factory=list)


def setor_de(graus) -> np.ndarray:
    return ((np.asarray(graus, float) % 360 + 11.25) // 22.5).astype(int) % 16


def por_extenso(setor: str) -> str:
    return NOME_SETOR.get(setor, setor)


def obter(area: Area, anos: int = 5) -> Vento | None:
    lon, lat = area.centro
    fim = fontes.hoje() - timedelta(days=5)
    lista = list(range(fim.year - anos, fim.year))                    # anos civis completos
    h = fontes.power_horario(lon, lat, lista)
    if h is not None and h["WS2M"].notna().sum() > 24 * 365:
        df = pd.DataFrame({"vel": h["WS2M"] * 3.6, "dir": h.get("WD10M"), "t": h.get("T2M"), "ur": h.get("RH2M")})
        return analisar(df, horario=True, fonte="NASA POWER horário (MERRA-2, ~50 km), hora local")
    d = fontes.power_diario(lon, lat, fontes.anos_atras(anos, fim), fim, parametros=("WS2M", "WD10M"))
    if d is None or d["WS2M"].notna().sum() < 365:
        return None
    df = pd.DataFrame({"vel": d["WS2M"] * 3.6, "dir": d.get("WD10M")})
    v = analisar(df, horario=False, fonte="NASA POWER diário (MERRA-2, ~50 km)")
    v.avisos.append("Vento horário indisponível; usadas médias diárias (sem janelas de pulverização)")
    return v


def analisar(df: pd.DataFrame, horario: bool = True, fonte: str = "") -> Vento:
    df = df.dropna(subset=["vel"])
    v = Vento(fonte, (int(df.index.year.min()), int(df.index.year.max())), horario)
    v.media_kmh = float(df["vel"].mean())
    v.calmaria_pct = float(100 * (df["vel"] < 3).mean())
    # rosa dos ventos: % do tempo por setor e classe de velocidade
    if "dir" in df and df["dir"].notna().any():
        r = df.dropna(subset=["dir"])
        sec = setor_de(r["dir"].values)
        tab = np.zeros((16, len(CLASSES)))
        for j, (_, lo, hi, _) in enumerate(CLASSES):
            m = (r["vel"].values >= lo) & (r["vel"].values < hi)
            tab[:, j] = np.bincount(sec[m], minlength=16)
        tab = 100 * tab / max(len(r), 1)
        v.rosa = pd.DataFrame(tab, index=SETORES, columns=[c[0] for c in CLASSES])
        tot = v.rosa.sum(axis=1)
        v.predominante = str(tot.idxmax())
        v.predominante_pct = float(tot.max())
        # direção média vetorial (de onde sopra) — para a seta no mapa
        rad = np.deg2rad(r["dir"].values)
        v.graus_predominante = float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
        if tot.max() - tot.median() < 2:
            v.avisos.append("Vento sem direção claramente predominante")
    # regime mensal
    g = df.groupby(df.index.month)["vel"]
    v.mensal = pd.DataFrame({"Mês": [MESES[m - 1] for m in g.mean().index],
                             "Vento médio (km/h)": g.mean().values, "P90 (km/h)": g.quantile(0.9).values,
                             "Acima de 10 km/h (%)": (g.apply(lambda s: (s > 10).mean() * 100)).values})
    # janelas de pulverização (só com dados horários)
    if horario and {"t", "ur"} <= set(df.columns) and df["t"].notna().any():
        c = CRITERIOS
        ok = (df["vel"].between(c["vento_min"], c["vento_max"]) & (df["t"] < c["temp_max"]) & (df["ur"] > c["ur_min"]))
        valido = df[["vel", "t", "ur"]].notna().all(axis=1)
        x = pd.DataFrame({"ok": ok[valido].astype(float), "mes": df.index[valido].month, "hora": df.index[valido].hour})
        jan = x[x["hora"].isin(HORAS)].pivot_table(index="mes", columns="hora", values="ok", aggfunc="mean") * 100
        v.janela = jan.reindex(index=range(1, 13), columns=HORAS)
    return v


def melhores_horas(v: Vento, meses=(10, 11, 12, 1, 2, 3), limiar: float = 50.0) -> str:
    """Faixas de horário com ≥ limiar% das horas adequadas, na média dos meses indicados."""
    if v.janela is None:
        return ""
    m = v.janela.loc[[k for k in meses if k in v.janela.index]].mean()
    boas = [h for h in m.index if m[h] >= limiar]
    if not boas:
        h = int(m.idxmax())
        return f"nenhum horário passa de {limiar:.0f}%; o melhor é por volta das {h}h ({m.max():.0f}%)"
    faixas, ini, ant = [], boas[0], boas[0]
    for h in boas[1:] + [None]:
        if h is not None and h == ant + 1:
            ant = h
            continue
        faixas.append(f"{ini}h–{ant + 1}h")
        if h is not None:
            ini = ant = h
    return " e ".join(faixas)


def tabelas(v: Vento) -> dict[str, pd.DataFrame]:
    t = {"Vento mensal": v.mensal.round(1)}
    if not v.rosa.empty:
        r = v.rosa.round(2)
        r.index.name = "Direção (de onde sopra)"
        t["Rosa dos ventos (%)"] = r.reset_index()
    if v.janela is not None:
        j = v.janela.round(0)
        j.index = [MESES[m - 1] for m in j.index]
        j.index.name = "Mês"
        j.columns = [f"{h}h" for h in j.columns]
        t["Janelas pulverização (%)"] = j.reset_index()
    return t
