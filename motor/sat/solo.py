"""Contexto do solo por SoilGrids 2.0 (ISRIC): granulometria PREDITA em grade de 250 m.

Só argila, silte e areia são usados: são os atributos mais estáveis e com melhor desempenho do modelo global.
Carbono, CTC, pH, N e densidade dependem muito do manejo e as predições ficavam longe da realidade das
lavouras. Um pixel de 250 × 250 m corresponde a 6,25 ha. A estimativa ajuda a contextualizar e a planejar a
amostragem; NÃO substitui a análise granulométrica do laboratório.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import Point

from . import fontes
from .base import Area

NOMES = {"clay": ("Argila", "%"), "silt": ("Silte", "%"), "sand": ("Areia", "%")}
# a API já devolve (valor ÷ d_factor) nas unidades-alvo (%)
CONVERTE = {k: 1.0 for k in NOMES}
ESPESSURA = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}            # para a média ponderada 0–30 cm
GRUPOS = [("Arenosa", 0, 15, "#F3DFA2"), ("Média", 15, 35, "#D9A95B"), ("Argilosa", 35, 60, "#A0522D"),
          ("Muito argilosa", 60, 100, "#6B2E14")]                  # grupos texturais (Embrapa) pelo teor de argila


def grupo_textural(argila: float) -> str:
    return next(n for n, lo, hi, _ in GRUPOS if argila < hi or hi == 100)


@dataclass
class Solo:
    tabela: pd.DataFrame = field(default_factory=pd.DataFrame)   # propriedade × profundidade: média, q05, q95
    pontos: list[tuple[float, float]] = field(default_factory=list)
    textura: str = ""
    camada: dict = field(default_factory=dict)      # 0–30 cm: {prop: (média, p5, p95)} ponderado pela espessura
    avisos: list[str] = field(default_factory=list)


def pontos_amostra(area: Area, maximo: int = 4) -> list[tuple[float, float]]:
    """Centro do talhão e, em áreas maiores, pontos internos afastados (≥ 1 pixel de 250 m)."""
    from ..geo import projetar
    pts = [area.poligono.representative_point()]
    if area.area_ha > 60 and maximo > 1:
        x0, y0, x1, y1 = area.poligono.bounds
        cand = [Point(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy) for fx, fy in
                ((0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25))]
        for p in cand:
            if area.poligono.contains(p) and all(p.distance(q) > 300 for q in pts):
                pts.append(p)
            if len(pts) >= maximo:
                break
    ll = [projetar(p, area.epsg, inverso=True) for p in pts]
    return [(p.x, p.y) for p in ll]


def obter(area: Area, consultar=None) -> Solo | None:
    consultar = consultar or fontes.soilgrids_ponto
    pts = pontos_amostra(area)
    linhas = []
    for k, (lon, lat) in enumerate(pts):
        if k:
            time.sleep(1.5)                           # respeita o limite de uso da API
        for r in consultar(lon, lat):
            linhas.append({**r, "ponto": k})
    if not linhas:
        return None
    return resumir(pd.DataFrame(linhas), pts)


def resumir(df: pd.DataFrame, pts) -> Solo:
    df = df[df["propriedade"].isin(NOMES)].copy()
    for c in ("media", "q05", "q95"):
        df[c] = df.apply(lambda r: None if r[c] is None or pd.isna(r[c]) else r[c] * CONVERTE[r["propriedade"]],
                         axis=1).astype(float)
    g = df.groupby(["propriedade", "profundidade"], sort=False).agg(
        media=("media", "mean"), q05=("q05", "min"), q95=("q95", "max"), pontos=("ponto", "nunique")).reset_index()
    g["Atributo"] = g["propriedade"].map(lambda p: NOMES[p][0])
    g["Unidade"] = g["propriedade"].map(lambda p: NOMES[p][1])
    s = Solo(g, list(pts))
    for prop in NOMES:
        sub = g[(g["propriedade"] == prop) & (g["profundidade"].isin(list(ESPESSURA)))]
        if len(sub):
            w = sub["profundidade"].map(ESPESSURA).astype(float).values
            s.camada[prop] = tuple(float(np.average(sub[c].fillna(sub["media"]).values, weights=w))
                                   for c in ("media", "q05", "q95"))
    if "clay" in s.camada:
        a = s.camada["clay"][0]
        s.textura = grupo_textural(a).lower() + f" (argila estimada ≈ {a:.0f}% na camada 0–30 cm)"
    return s
