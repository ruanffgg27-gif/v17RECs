"""Comportamento da lavoura a partir de séries Sentinel-2 (NDVI, NDRE, NDMI, MSAVI2).

Procedimento padronizado:
- Uma imagem por mês (a menos nublada) nos últimos N anos; nuvens e sombras removidas pela SCL.
- Uma imagem entra se ≥ 60% do talhão estiver limpo.
- Ano agrícola de julho a junho (ex.: 2024/25). O "pico" de cada ano é o maior NDVI de cada pixel.
- Vigor relativo: posição do pixel em relação à mediana do próprio talhão naquele ano (escore robusto),
  o que permite comparar anos com culturas e estágios diferentes.

Baixo vigor, sozinho, não identifica a causa (nutrição, compactação, doença, falha de plantio etc.);
os mapas indicam onde investigar.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import ndimage

from . import fontes
from .base import Area, Grade, grade_da_area, ler_na_grade

INDICES = {"ndvi": "NDVI", "ndre": "NDRE", "ndmi": "NDMI", "msavi2": "MSAVI2"}
SCL_VALIDAS = (4, 5, 6, 7)
CLASSES_ESTAB = ["Alto e estável", "Médio", "Instável", "Baixo e estável"]
COR_ESTAB = {"Alto e estável": "#1A9850", "Médio": "#D9EF8B", "Instável": "#FDAE61", "Baixo e estável": "#D73027"}


@dataclass
class Cena:
    data: date
    indices: dict[str, np.ndarray] = field(repr=False)
    limpo: float = 1.0                          # fração do talhão sem nuvem

    @property
    def ano_agricola(self) -> str:
        a = self.data.year if self.data.month >= 7 else self.data.year - 1
        return f"{a}/{(a + 1) % 100:02d}"


@dataclass
class Lavoura:
    grade: Grade
    cenas: list[Cena] = field(repr=False)
    serie: pd.DataFrame = field(default_factory=pd.DataFrame)          # data, média, p10, p90 (NDVI)
    picos: dict[str, np.ndarray] = field(default_factory=dict, repr=False)   # ano agrícola → NDVI pico
    vigor_rel: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    estab_media: np.ndarray | None = field(default=None, repr=False)
    estab_dp: np.ndarray | None = field(default=None, repr=False)
    estab_classe: np.ndarray | None = field(default=None, repr=False)     # índice em CLASSES_ESTAB
    estab_area: dict[str, float] = field(default_factory=dict)
    anomalia: np.ndarray | None = field(default=None, repr=False)          # relativa (desconta o talhão)
    anomalia_info: dict = field(default_factory=dict)
    uniformidade: dict = field(default_factory=dict)
    baixa_cobertura: np.ndarray | None = field(default=None, repr=False)   # fração 0–1
    indices_ano: pd.DataFrame = field(default_factory=pd.DataFrame)
    hist: dict[str, np.ndarray] = field(default_factory=dict, repr=False)   # índice → média histórica do pico
    hist_anos: list[str] = field(default_factory=list)                      # anos agrícolas que entraram na média
    pontos: pd.DataFrame = field(default_factory=pd.DataFrame)
    avisos: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ leitura
def escolher_datas(itens: list[dict], extras_recentes: int = 2) -> list[list[dict]]:
    """Uma data por mês (a menos nublada) + as datas mais recentes. Devolve grupos de itens por data."""
    por_data: dict[date, list[dict]] = {}
    for it in itens:
        por_data.setdefault(fontes.data_item(it), []).append(it)
    nuv = {d: float(np.mean([i["properties"].get("eo:cloud_cover", 100) for i in its])) for d, its in por_data.items()}
    por_mes: dict[tuple[int, int], date] = {}
    for d in por_data:
        k = (d.year, d.month)
        if k not in por_mes or nuv[d] < nuv[por_mes[k]]:
            por_mes[k] = d
    escolhidas = set(por_mes.values())
    recentes = sorted((d for d in por_data if nuv[d] < 30), reverse=True)[:extras_recentes]
    escolhidas |= set(recentes)
    return [por_data[d] for d in sorted(escolhidas)]


def _escala(item, chave):
    a = item["assets"][chave]
    rb = (a.get("raster:bands") or [{}])[0]
    if "scale" in rb or "offset" in rb:
        return float(rb.get("scale", 1e-4)), float(rb.get("offset", 0.0))
    p = item["properties"]
    try:
        base = float(p.get("s2:processing_baseline", "0"))
    except ValueError:
        base = 0.0
    return 1e-4, (-0.1 if base >= 4 and not p.get("earthsearch:boa_offset_applied", False) else 0.0)


def indices_de(bandas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    red, nir, re1, sw = (bandas[k] for k in ("red", "nir", "rededge1", "swir16"))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndre = (nir - re1) / (nir + re1)
        ndmi = (nir - sw) / (nir + sw)
        msavi2 = (2 * nir + 1 - np.sqrt(np.clip((2 * nir + 1) ** 2 - 8 * (nir - red), 0, None))) / 2
    return {"ndvi": np.clip(ndvi, -1, 1), "ndre": np.clip(ndre, -1, 1), "ndmi": np.clip(ndmi, -1, 1),
            "msavi2": np.clip(msavi2, -1, 1)}


def ler_cena(itens: list[dict], grade: Grade, assinar=lambda h: h) -> Cena | None:
    """Mosaico dos tiles de uma data, com máscara de nuvem, e índices na grade."""
    acum = None
    for it in itens:
        a = it["assets"]
        b = {}
        for k in ("red", "nir", "rededge1", "swir16"):
            dn = ler_na_grade(assinar(a[k]["href"]), grade, "bilinear", nodata_origem=0)
            s, o = _escala(it, k)
            b[k] = dn * s + o
        scl = ler_na_grade(assinar(a["scl"]["href"]), grade, "vizinho", nodata_origem=0)
        ok = np.isin(scl, SCL_VALIDAS) & np.all([np.isfinite(v) & (v > 0) for v in b.values()], axis=0)
        for k in b:
            b[k] = np.where(ok, b[k], np.nan)
        if acum is None:
            acum = b
        else:
            for k in acum:
                acum[k] = np.where(np.isnan(acum[k]), b[k], acum[k])
    if acum is None:
        return None
    ind = {k: v.astype(np.float32) for k, v in indices_de(acum).items()}
    limpo = float(np.isfinite(ind["ndvi"][grade.mascara]).mean())
    return Cena(fontes.data_item(itens[0]), ind, limpo)


def obter(area: Area, anos: int = 4, fim: date | None = None, trabalhadores: int = 8, buscar=None,
          min_limpo: float = 0.6) -> Lavoura | None:
    fim = fim or fontes.hoje()
    inicio = fontes.anos_atras(anos, fim)
    grade = grade_da_area(area, 10.0)
    itens = (buscar or fontes.sentinel2)(area.bbox_ll(50), inicio, fim)
    if not itens:
        return None
    grupos = escolher_datas(itens)

    def ler(g):
        try:
            return ler_cena(g, grade)
        except Exception:  # noqa: BLE001
            return None

    from .base import mapear_com_prazo
    cenas = [c for c in mapear_com_prazo(ler, grupos, trabalhadores, prazo_s=420)
             if c is not None and c.limpo >= min_limpo]
    if len(cenas) < 4:
        return None
    return analisar(grade, cenas)


# ------------------------------------------------------------------ análises
def z_robusto(v: np.ndarray, mascara: np.ndarray, piso: float = 0.03) -> np.ndarray:
    x = v[mascara & np.isfinite(v)]
    if len(x) < 10:
        return np.full(v.shape, np.nan)
    med = np.median(x)
    esc = 1.4826 * np.median(np.abs(x - med))
    # piso na escala: em talhões muito uniformes, diferenças < ~0,03 de NDVI são ruído, não padrão
    esc = max(esc, piso, 0.04 * abs(med))
    z = (v - med) / esc
    z[~mascara] = np.nan
    return z


def _suavizar(v: np.ndarray, m: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    ok = np.isfinite(v)
    num = ndimage.gaussian_filter(np.where(ok, v, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    with np.errstate(invalid="ignore"):
        return np.where(ok & (den > 0.2), num / np.maximum(den, 1e-9), np.nan)


def analisar(grade: Grade, cenas: list[Cena]) -> Lavoura:
    cenas = sorted(cenas, key=lambda c: c.data)
    L = Lavoura(grade, cenas)
    m = grade.mascara
    # série temporal do NDVI
    linhas = []
    for c in cenas:
        v = c.indices["ndvi"][m]
        v = v[np.isfinite(v)]
        linhas.append({"data": c.data, "ano_agricola": c.ano_agricola, "media": float(np.mean(v)),
                       "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                       "cv": float(100 * np.std(v) / np.mean(v)) if np.mean(v) > 0 else np.nan,
                       **{f"{k}_media": float(np.nanmean(c.indices[k][m])) for k in INDICES}})
    L.serie = pd.DataFrame(linhas)
    # picos por ano agrícola (≥ 3 imagens no ano)
    for ano, grupo in L.serie.groupby("ano_agricola"):
        cs = [c for c in cenas if c.ano_agricola == ano]
        if len(cs) < 3:
            continue
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pico = np.nanmax(np.stack([c.indices["ndvi"] for c in cs]), axis=0)
        pico = _suavizar(pico, m)                       # atenua ruído de pixel (janela ~30 m)
        pico[~m] = np.nan
        if np.nanmedian(pico[m]) < 0.35:              # ano sem lavoura/cobertura relevante
            continue
        L.picos[ano] = pico
        L.vigor_rel[ano] = z_robusto(pico, m)
    _estabilidade(L)
    _anomalia(L)
    _uniformidade(L)
    _baixa_cobertura(L)
    _indices_por_ano(L)
    _historico_indices(L)
    L.pontos = pontos_prioritarios(L)
    return L


def _historico_indices(L: Lavoura):
    """Mapa histórico de cada índice: em cada ano agrícola com lavoura, o maior valor de cada pixel (pico do
    ano, como no NDVI); depois, a média entre os anos. Um ano isolado mostra a safra; a média de vários anos
    mostra o padrão que se repete no talhão."""
    import warnings
    m = L.grade.mascara
    anos = list(L.picos)
    if not anos:
        return
    for k in INDICES:
        por_ano = []
        for ano in anos:
            cs = [c for c in L.cenas if c.ano_agricola == ano]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pico = np.nanmax(np.stack([c.indices[k] for c in cs]), axis=0)
            por_ano.append(_suavizar(pico, m))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            media = np.nanmean(np.stack(por_ano), axis=0)
        media[~m] = np.nan
        L.hist[k] = media
    L.hist_anos = anos


def _estabilidade(L: Lavoura, lim_media: float = 0.5, lim_dp: float = 0.8):
    if len(L.vigor_rel) < 2:
        L.avisos.append("Estabilidade: é preciso ao menos 2 anos agrícolas com lavoura")
        return
    Z = np.stack(list(L.vigor_rel.values()))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            media, dp = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0)
    cls = np.full(media.shape, np.nan)
    cls[np.isfinite(media)] = 1                                    # médio
    cls[(dp > lim_dp)] = 2                                         # instável
    cls[(media >= lim_media) & (dp <= lim_dp)] = 0                 # alto e estável
    cls[(media <= -lim_media) & (dp <= lim_dp)] = 3                # baixo e estável
    cls[~L.grade.mascara] = np.nan
    L.estab_media, L.estab_dp, L.estab_classe = media, dp, cls
    ap = L.grade.area_pixel_ha
    L.estab_area = {n: float((cls == i).sum() * ap) for i, n in enumerate(CLASSES_ESTAB)}


def _anomalia(L: Lavoura, janela_dias: int = 35):
    m = L.grade.mascara
    ult = L.cenas[-1]
    doy = ult.data.timetuple().tm_yday
    def dist_doy(c):
        d = abs(c.data.timetuple().tm_yday - doy)
        return min(d, 365 - d)
    ref = [c for c in L.cenas[:-1] if (ult.data - c.data).days > 200 and dist_doy(c) <= janela_dias]
    if len(ref) >= 2:
        esperado = np.nanmedian(np.stack([c.indices["ndvi"] for c in ref]), axis=0)
        base = f"mediana de {len(ref)} imagens da mesma época em anos anteriores"
    else:
        ant = [c for c in L.cenas[:-1] if 5 <= (ult.data - c.data).days <= 60]
        if not ant:
            L.avisos.append("Anomalia recente: sem imagens de referência suficientes")
            return
        esperado = ant[-1].indices["ndvi"]
        base = f"imagem anterior ({ant[-1].data:%d/%m/%Y})"
    dif = ult.indices["ndvi"] - esperado
    geral = float(np.nanmedian(dif[m]))
    rel = dif - geral
    rel[~m] = np.nan
    ap = L.grade.area_pixel_ha
    L.anomalia = rel
    L.anomalia_info = {"data": ult.data, "base": base, "desvio_geral": geral,
                       "queda_forte_ha": float(np.nansum(rel < -0.15) * ap),
                       "queda_ha": float(np.nansum((rel >= -0.15) & (rel < -0.07)) * ap),
                       "alta_ha": float(np.nansum(rel > 0.07) * ap)}


def _uniformidade(L: Lavoura):
    if not L.picos:
        return
    ano = list(L.picos)[-1]
    p = L.picos[ano][L.grade.mascara]
    p = p[np.isfinite(p)]
    med = np.median(p)
    rel = 100 * p / med
    lims = [0, 80, 90, 110, 120, 1e9]
    rot = ["< 80%", "80–90%", "90–110%", "110–120%", "> 120%"]
    cont = np.histogram(rel, bins=lims)[0] * L.grade.area_pixel_ha
    L.uniformidade = {"ano": ano, "cv": float(100 * p.std() / p.mean()), "mediana": float(med),
                      "classes": dict(zip(rot, map(float, cont))),
                      "uniforme_pct": float(100 * ((rel >= 90) & (rel <= 110)).mean())}


def _baixa_cobertura(L: Lavoura, lim_lavoura: float = 0.55, lim_pixel: float = 0.7):
    """Fração das imagens com lavoura estabelecida em que o pixel ficou abaixo de 70% da mediana."""
    m = L.grade.mascara
    cs = [c for c in L.cenas if np.nanmedian(c.indices["ndvi"][m]) >= lim_lavoura]
    if len(cs) < 3:
        L.avisos.append("Baixa cobertura recorrente: poucas imagens com lavoura estabelecida")
        return
    cont = np.zeros(m.shape)
    n = np.zeros(m.shape)
    for c in cs:
        v = c.indices["ndvi"]
        med = np.nanmedian(v[m])
        ok = np.isfinite(v)
        cont += (ok & (v < lim_pixel * med))
        n += ok
    with np.errstate(invalid="ignore"):
        f = np.where(n > 0, cont / n, np.nan)
    f[~m] = np.nan
    L.baixa_cobertura = f


def _indices_por_ano(L: Lavoura):
    """Média de cada índice na data de pico (maior NDVI médio) de cada ano agrícola."""
    linhas = []
    for ano, g in L.serie.groupby("ano_agricola"):
        if ano not in L.picos:
            continue
        i = g["media"].idxmax()
        linhas.append({"Ano agrícola": ano, "Data do pico": g.loc[i, "data"],
                       **{INDICES[k]: g.loc[i, f"{k}_media"] for k in INDICES},
                       "CV do NDVI (%)": g.loc[i, "cv"], "Imagens": len(g)})
    L.indices_ano = pd.DataFrame(linhas)


def pontos_prioritarios(L: Lavoura, area_min_ha: float = 0.3, maximo: int = 8) -> pd.DataFrame:
    """Coordenadas para visitar: centro das manchas mais relevantes de cada tipo."""
    g = L.grade
    ap = g.area_pixel_ha
    candidatos = []
    camadas = []
    if L.estab_classe is not None:
        camadas.append(("Baixo vigor persistente", L.estab_classe == 3, 1.0,
                        "vigor abaixo do talhão em vários anos"))
    if L.anomalia is not None:
        camadas.append(("Queda recente de vigor", np.nan_to_num(L.anomalia, nan=0) < -0.12, 0.9,
                        f"queda em relação à referência ({L.anomalia_info.get('data', ''):%d/%m/%Y})"
                        if L.anomalia_info.get("data") else "queda recente"))
    if L.baixa_cobertura is not None:
        camadas.append(("Baixa cobertura recorrente", np.nan_to_num(L.baixa_cobertura, nan=0) >= 0.5, 0.8,
                        "pouca vegetação em metade ou mais das imagens com lavoura"))
    for tipo, masc, peso, desc in camadas:
        masc = ndimage.binary_opening(masc & g.mascara, iterations=1)
        lab, n = ndimage.label(masc)
        if n == 0:
            continue
        tam = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * ap
        dist = ndimage.distance_transform_edt(masc)
        por_tipo = []
        for k in np.argsort(-tam)[:4]:
            if tam[k] < area_min_ha:
                continue
            reg = lab == k + 1
            d = np.where(reg, dist, -1)
            i, j = np.unravel_index(np.argmax(d), d.shape)
            lon, lat = g.lonlat(i, j)
            por_tipo.append({"Tipo": tipo, "Área (ha)": float(tam[k]), "Latitude": float(lat),
                             "Longitude": float(lon), "Motivo": desc, "_escore": float(tam[k]) * peso,
                             "_lin": int(i), "_col": int(j)})
        candidatos += por_tipo
    if not candidatos:
        return pd.DataFrame(columns=["Ponto", "Tipo", "Área (ha)", "Latitude", "Longitude", "Motivo"])
    df = pd.DataFrame(candidatos).sort_values("_escore", ascending=False).head(maximo).reset_index(drop=True)
    df.insert(0, "Ponto", [f"P{i + 1}" for i in range(len(df))])
    return df


def kml_pontos(df: pd.DataFrame, nome: str = "Pontos prioritários") -> bytes:
    pm = "".join(
        f"<Placemark><name>{r['Ponto']}</name><description>{r['Tipo']} · {r['Área (ha)']:.1f} ha · "
        f"{r['Motivo']}</description><Point><coordinates>{r['Longitude']:.6f},{r['Latitude']:.6f},0"
        f"</coordinates></Point></Placemark>" for _, r in df.iterrows())
    return (f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>{nome}</name>{pm}</Document></kml>").encode("utf-8")
