"""Executa as frentes do módulo Satélites em paralelo e junta os resultados."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import agua, clima, historico, lavoura, relevo, solo, termico
from .base import Area, Grade

MODULOS = {"historico": "Histórico de uso (MapBiomas/PRODES)", "lavoura": "Comportamento da lavoura (Sentinel-2)",
           "relevo": "Relevo e água (Copernicus DEM)", "clima": "Clima e vento (CHIRPS/NASA POWER)",
           "solo": "Contexto do solo (SoilGrids)", "termico": "Temperatura de superfície (Landsat)",
           "agua": "Água no solo (balanço hídrico e NISAR)"}


@dataclass
class Diagnostico:
    area: Area
    historico: historico.Historico | None = None
    lavoura: lavoura.Lavoura | None = None
    relevo: relevo.Relevo | None = None
    clima: clima.Clima | None = None
    solo: solo.Solo | None = None
    termico: termico.Termico | None = None
    agua: agua.Agua | None = None
    avisos: list[str] = field(default_factory=list)
    tempos: dict[str, float] = field(default_factory=dict)


def reamostrar(valores: np.ndarray, origem: Grade, destino: Grade) -> np.ndarray:
    """Leva uma camada de uma grade UTM para outra (bilinear; nan fora da origem)."""
    X, Y = destino.malha()
    col = (X - origem.x0) / origem.res - 0.5
    lin = (origem.y1 - Y) / origem.res - 0.5
    v = np.where(np.isfinite(valores), valores, 0.0)
    w = np.isfinite(valores).astype(float)
    num = ndimage.map_coordinates(v, [lin, col], order=1, mode="constant", cval=0.0)
    den = ndimage.map_coordinates(w, [lin, col], order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore"):
        return np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)


def executar(area: Area, modulos=tuple(MODULOS), anos_lavoura: int = 4, anos_termico: int = 3,
             progresso=None, fontes_sinteticas: dict | None = None, par_agua=None,
             earthdata: dict | None = None) -> Diagnostico:
    """Roda as frentes escolhidas. `fontes_sinteticas` (testes): {modulo: função(area) → resultado}.

    `par_agua`: agua.ParametrosAgua com o que o usuário informou (opcional); `earthdata`: credencial do login
    Earthdata para ler o NISAR ({"token": …} ou {"usuario": …, "senha": …})."""
    d = Diagnostico(area)
    fs = fontes_sinteticas or {}
    tarefas = {
        "historico": lambda: historico.obter(area),
        "lavoura": lambda: lavoura.obter(area, anos=anos_lavoura),
        "clima": lambda: clima.obter(area),
        "solo": lambda: solo.obter(area),
        "termico": lambda: termico.obter(area, anos=anos_termico),
        "relevo": lambda: relevo.obter(area),
        "agua": lambda: agua.obter(area, par_agua, earthdata),
    }

    def rodar(nome):
        t0 = time.time()
        try:
            r = fs[nome](area) if nome in fs else tarefas[nome]()
            erro = None
        except Exception as e:  # noqa: BLE001
            r, erro = None, e
        return nome, r, erro, time.time() - t0

    escolhidos = [m for m in MODULOS if m in modulos]
    with ThreadPoolExecutor(min(7, max(1, len(escolhidos)))) as ex:
        futuros = [ex.submit(rodar, m) for m in escolhidos]
        for fut in futuros:
            nome, r, erro, dt = fut.result()
            d.tempos[nome] = dt
            setattr(d, nome, r)
            if progresso:
                progresso(f"{'✅' if r is not None else '⚠️'} {MODULOS[nome]} ({dt:.0f} s)")
            if r is None:
                d.avisos.append(f"{MODULOS[nome]}: sem dados" + (f" ({type(erro).__name__}: {erro})" if erro else
                                                                 " (fonte indisponível ou imagens insuficientes)"))
            else:
                d.avisos += [f"{MODULOS[nome]}: {a}" for a in getattr(r, "avisos", [])]
    # erosão com a cobertura observada pela lavoura (NDVI médio de todas as imagens)
    if d.relevo is not None and d.lavoura is not None and d.lavoura.cenas:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ndvi_medio = np.nanmean(np.stack([c.indices["ndvi"] for c in d.lavoura.cenas]), axis=0)
        cob = reamostrar(ndvi_medio, d.lavoura.grade, d.relevo.grade)
        relevo.aplicar_cobertura(d.relevo, cob)
    return d
