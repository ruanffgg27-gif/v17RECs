"""Temperatura de superfície (Landsat 8/9, Coleção 2 Nível 2, banda térmica ST_B10, ~30 m reamostrada de 100 m).

Mostra onde o talhão fica RELATIVAMENTE mais quente na passagem do satélite (~10h30). O sensor térmico enxerga
em 100 m; para o mapa, a média das passagens é suavizada (~45 m) e interpolada em 10 m, o que representa
melhor a resolução real do dado do que os blocos de 30 m. Temperatura elevada, sozinha, não comprova falta de água: solo exposto, palhada, relevo e
cobertura também aquecem a superfície. Evapotranspiração exigiria modelagem com dados auxiliares e não
é estimada aqui.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import fontes
from .base import Area, Grade, grade_da_area, ler_na_grade

ESCALA, DESLOC = 0.00341802, 149.0          # ST_B10 → Kelvin


@dataclass
class CenaTermica:
    data: date
    lst: np.ndarray = field(repr=False)      # °C (nan = nuvem/sem dado)
    limpo: float = 1.0


@dataclass
class Termico:
    grade: Grade
    cenas: list[CenaTermica] = field(repr=False)
    relativo_medio: np.ndarray | None = field(default=None, repr=False)   # °C acima/abaixo da mediana do talhão
    recorrencia_quente: np.ndarray | None = field(default=None, repr=False)  # fração das cenas no 20% mais quente
    tabela: pd.DataFrame = field(default_factory=pd.DataFrame)
    quente_recorrente_ha: float = 0.0
    media_c: float = float("nan")                                          # média do talhão nas passagens (°C)
    avisos: list[str] = field(default_factory=list)


def mascara_qa(qa: np.ndarray) -> np.ndarray:
    """Pixel limpo pelo QA_PIXEL: bit 6 (claro) ligado e sem nuvem/sombra/neve (bits 1, 3, 4, 5)."""
    q = np.nan_to_num(qa, nan=1).astype(np.uint16)
    ruim = (q & (1 << 0)) | (q & (1 << 1)) | (q & (1 << 3)) | (q & (1 << 4)) | (q & (1 << 5))
    return ((q & (1 << 6)) > 0) & (ruim == 0)


def ler_cena(item: dict, grade: Grade, assinar=fontes.assinar_pc) -> CenaTermica | None:
    a = item["assets"]
    st = ler_na_grade(assinar(a["lwir11"]["href"]), grade, "bilinear", nodata_origem=0)
    qa = ler_na_grade(assinar(a["qa_pixel"]["href"]), grade, "vizinho", nodata_origem=1)
    ok = mascara_qa(qa) & np.isfinite(st)
    lst = np.where(ok, st * ESCALA + DESLOC - 273.15, np.nan)
    limpo = float(np.isfinite(lst[grade.mascara]).mean())
    return CenaTermica(fontes.data_item(item), lst, limpo)


def obter(area: Area, anos: int = 3, fim: date | None = None, trabalhadores: int = 8, buscar=None,
          ler=None, min_limpo: float = 0.7, max_cenas: int = 40) -> Termico | None:
    fim = fim or fontes.hoje()
    grade = grade_da_area(area, 30.0, buffer_m=90)
    itens = (buscar or fontes.landsat)(area.bbox_ll(100), fontes.anos_atras(anos, fim), fim)
    if not itens:
        return None
    itens = sorted(itens, key=lambda i: i["properties"].get("eo:cloud_cover", 100))[:max_cenas]
    ler = ler or ler_cena

    def um(it):
        try:
            return ler(it, grade)
        except Exception:  # noqa: BLE001
            return None

    from .base import mapear_com_prazo
    cenas = [c for c in mapear_com_prazo(um, itens, trabalhadores, prazo_s=300)
             if c is not None and c.limpo >= min_limpo]
    if len(cenas) < 3:
        return None
    return analisar(grade, cenas)


def analisar(grade: Grade, cenas: list[CenaTermica]) -> Termico:
    cenas = sorted(cenas, key=lambda c: c.data)
    m = grade.mascara
    rels, quentes, ns = [], np.zeros(m.shape), np.zeros(m.shape)
    linhas = []
    for c in cenas:
        v = c.lst[m]
        v = v[np.isfinite(v)]
        med = np.median(v)
        rel = c.lst - med
        rels.append(rel)
        p80 = np.percentile(v, 80)
        ok = np.isfinite(c.lst)
        quentes += ok & (c.lst >= p80)
        ns += ok
        linhas.append({"Data": c.data, "Média (°C)": float(v.mean()), "Mín (°C)": float(v.min()),
                       "Máx (°C)": float(v.max()), "Amplitude (°C)": float(np.percentile(v, 95) - np.percentile(v, 5)),
                       "Área limpa (%)": 100 * c.limpo})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rel_med = np.nanmean(np.stack(rels), axis=0)
    with np.errstate(invalid="ignore"):
        rec = np.where(ns > 0, quentes / ns, np.nan)
    rec[~m] = np.nan
    T = Termico(grade, cenas, rel_med, rec, pd.DataFrame(linhas))   # relativo_medio inclui o entorno (suavização)
    T.quente_recorrente_ha = float(np.nansum(rec >= 0.5) * grade.area_pixel_ha)
    T.media_c = float(np.mean([l["Média (°C)"] for l in linhas]))
    return T


def superficie(T: Termico, fator: int = 3, sigma_px: float = 1.5):
    """Média relativa suavizada (gaussiana com tratamento de nan) e interpolada (cúbica) numa grade `fator`×
    mais fina. Devolve (valores °C relativos, extent UTM, resolução)."""
    from scipy import ndimage
    v = T.relativo_medio
    ok = np.isfinite(v)
    num = ndimage.gaussian_filter(np.where(ok, v, 0.0), sigma_px, mode="nearest")
    den = ndimage.gaussian_filter(ok.astype(float), sigma_px, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(den > 0.2, num / den, np.nan)
    s = np.where(np.isfinite(s), s, np.nanmedian(s))
    fino = ndimage.zoom(s, fator, order=3, mode="nearest")
    g = T.grade
    return fino, g.extent, g.res / fator


def resumo_superficie(T: Termico, fino, extent, res, area) -> dict:
    """Estatísticas do mapa suavizado dentro do talhão."""
    from shapely import contains_xy
    x0, x1, y0, y1 = extent
    h, w = fino.shape
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
    ys = y1 - (np.arange(h) + 0.5) * (y1 - y0) / h
    X, Y = np.meshgrid(xs, ys)
    dentro = contains_xy(area.poligono, X, Y)
    v = fino[dentro]
    px_ha = (x1 - x0) / w * (y1 - y0) / h / 1e4
    return {"p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "amplitude": float(np.percentile(v, 95) - np.percentile(v, 5)),
            "quente_ha": float((v >= 1.0).sum() * px_ha), "frio_ha": float((v <= -1.0).sum() * px_ha)}
