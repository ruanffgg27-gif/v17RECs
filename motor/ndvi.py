"""NDVI (Sentinel-2 L2A) dos últimos 12 meses — pico de vigor e série temporal por talhão.

Fonte: Copernicus Sentinel-2 (ESA), processamento L2A, em Cloud-Optimized GeoTIFF no AWS Open Data,
catalogado pelo Earth Search (Element 84). Sem chave de acesso.

Procedimento padronizado (igual para todas as fazendas):
1. Busca as cenas com até 60% de nuvens (no tile inteiro) dos últimos 12 meses.
2. Escolhe uma data por mês (a de menor cobertura de nuvens) — no máximo 12 datas.
3. Lê só a janela da fazenda (vermelho B04, infravermelho próximo B08 e classificação SCL),
   reamostrada numa grade de 20 m no sistema UTM do projeto.
4. Remove nuvens, sombras e pixels inválidos pela SCL (mantém vegetação, solo exposto, água e
   não classificado).
5. Uma data entra na análise se ≥ 60% da área da fazenda estiver limpa.
6. Pico de vigor = NDVI máximo por pixel entre as datas aceitas; série temporal = NDVI médio de cada
   talhão em cada data (só quando ≥ 60% do talhão estiver limpo).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import requests

STAC = "https://earth-search.aws.element84.com/v1/search"
COLECOES = ["sentinel-2-c1-l2a", "sentinel-2-l2a"]
SCL_VALIDAS = (4, 5, 6, 7)       # vegetação, solo exposto, água, não classificado
CREDITO = ("Contém dados modificados Copernicus Sentinel-2 (ESA), processamento L2A, "
           "via Earth Search (Element 84) / AWS Open Data")

# classes padronizadas do NDVI (limites e cores) — as mesmas em todos os books
FAIXAS_NDVI = [(-1.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
CORES_NDVI = ["#A50026", "#F46D43", "#FDD76B", "#D9EF8B", "#91CF60", "#1A9850", "#005A32"]
ROTULOS_NDVI = ["< 0,20", "0,20–0,35", "0,35–0,50", "0,50–0,60", "0,60–0,70", "0,70–0,80", "> 0,80"]


@dataclass
class ResultadoNDVI:
    grade_x0: float                 # canto superior esquerdo (UTM)
    grade_y1: float
    res: float
    pico: np.ndarray                # NDVI máximo por pixel (nan fora dos talhões)
    mascara: np.ndarray             # pixels dentro dos talhões
    datas: list[date]               # datas aceitas
    serie: dict[str, list[float]]   # talhão → NDVI médio em cada data (nan se nublado)
    serie_fazenda: list[float]
    pico_talhao: dict[str, float] = field(default_factory=dict)
    cv_talhao: dict[str, float] = field(default_factory=dict)
    periodo: tuple[date, date] | None = None
    n_buscadas: int = 0

    @property
    def extent(self):
        h, w = self.pico.shape
        return (self.grade_x0, self.grade_x0 + w * self.res, self.grade_y1 - h * self.res, self.grade_y1)


# ---------------------------------------------------------------- catálogo
def buscar_cenas(bbox_ll, inicio: date, fim: date, nuvem_max: float = 60, timeout: float = 30) -> list[dict]:
    """Itens STAC que cobrem a caixa (lon0, lat0, lon1, lat1) no período."""
    for colecao in COLECOES:
        corpo = {"collections": [colecao], "bbox": list(bbox_ll), "limit": 200,
                 "datetime": f"{inicio.isoformat()}T00:00:00Z/{fim.isoformat()}T23:59:59Z",
                 "query": {"eo:cloud_cover": {"lt": nuvem_max}}}
        itens = None
        for tentativa in (corpo, {k: v for k, v in corpo.items() if k != "query"}):
            try:
                r = requests.post(STAC, json=tentativa, timeout=timeout)
                r.raise_for_status()
                itens = r.json().get("features", [])
                break
            except Exception:  # noqa: BLE001
                continue
        if not itens:
            continue
        itens = [i for i in itens if i["properties"].get("eo:cloud_cover", 0) < nuvem_max]
        itens = [i for i in itens if all(k in i.get("assets", {}) for k in ("red", "nir", "scl"))]
        if itens:
            return itens
    return []


def _data(item) -> date:
    return datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date()


def escolher_datas(itens: list[dict], max_datas: int = 12) -> list[list[dict]]:
    """Agrupa os itens por data e escolhe uma data por mês (a menos nublada). Mais recente primeiro."""
    por_data: dict[date, list[dict]] = {}
    for it in itens:
        por_data.setdefault(_data(it), []).append(it)
    por_mes: dict[tuple[int, int], tuple[float, date]] = {}
    for d, its in por_data.items():
        nuv = float(np.mean([i["properties"].get("eo:cloud_cover", 100) for i in its]))
        k = (d.year, d.month)
        if k not in por_mes or nuv < por_mes[k][0]:
            por_mes[k] = (nuv, d)
    datas = sorted((d for _, d in por_mes.values()), reverse=True)[:max_datas]
    return [por_data[d] for d in sorted(datas)]


def _escala(item, chave: str) -> tuple[float, float]:
    """(escala, deslocamento) para converter o número digital em reflectância."""
    a = item["assets"][chave]
    rb = (a.get("raster:bands") or [{}])[0]
    if "scale" in rb or "offset" in rb:
        return float(rb.get("scale", 1e-4)), float(rb.get("offset", 0.0))
    p = item["properties"]
    try:
        base = float(p.get("s2:processing_baseline", "0"))
    except ValueError:
        base = 0.0
    desl = -0.1 if base >= 4 and not p.get("earthsearch:boa_offset_applied", False) else 0.0
    return 1e-4, desl


# ------------------------------------------------------------------ leitura
def _ler(href: str, epsg: int, transform, w: int, h: int, categorico: bool) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=f"EPSG:{epsg}", transform=transform, width=w, height=h, nodata=0,
                       resampling=Resampling.nearest if categorico else Resampling.average) as vrt:
            return vrt.read(1).astype(float)


def _ambiente():
    import rasterio
    return rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                        AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_TIMEOUT="25",
                        GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                        VSI_CACHE="TRUE", GDAL_CACHEMAX=256)


def ndvi_data(itens: list[dict], epsg: int, transform, w: int, h: int) -> np.ndarray:
    """NDVI de uma data (mosaico dos tiles MGRS daquela passagem), nan onde inválido/nublado."""
    ndvi = np.full((h, w), np.nan)
    for it in itens:
        a = it["assets"]
        red = _ler(a["red"]["href"], epsg, transform, w, h, False)
        nir = _ler(a["nir"]["href"], epsg, transform, w, h, False)
        scl = _ler(a["scl"]["href"], epsg, transform, w, h, True)
        sr, orr = _escala(it, "red")
        sn, on = _escala(it, "nir")
        ok = (red > 0) & (nir > 0) & np.isin(scl, SCL_VALIDAS)
        r, n = red * sr + orr, nir * sn + on
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (n - r) / (n + r)
        ok &= np.isfinite(v) & (n + r > 0.02)
        novo = ok & np.isnan(ndvi)
        ndvi[novo] = np.clip(v[novo], -1, 1)
    return ndvi


def calcular(talhoes: dict[str, object], epsg: int, bbox_ll, hoje: date | None = None,
             res: float = 20.0, max_px: int = 2_500_000, min_limpo: float = 0.6,
             trabalhadores: int = 6, buscar=None) -> ResultadoNDVI | None:
    """NDVI da fazenda. `talhoes`: nome → polígono UTM (shapely). Devolve None se não houver dados."""
    try:
        import rasterio  # noqa: F401
        from rasterio.transform import from_origin
    except ImportError:
        return None
    from shapely import contains_xy
    from shapely.ops import unary_union

    buscar = buscar or buscar_cenas
    hoje = hoje or date.today()
    inicio = hoje - timedelta(days=365)
    itens = buscar(bbox_ll, inicio, hoje)
    if not itens:
        return None
    grupos = escolher_datas(itens)

    todos = unary_union(list(talhoes.values()))
    x0, y0, x1, y1 = todos.bounds
    x0, y0, x1, y1 = x0 - 2 * res, y0 - 2 * res, x1 + 2 * res, y1 + 2 * res
    while (x1 - x0) * (y1 - y0) / res ** 2 > max_px:
        res *= 1.5
    w, h = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    T = from_origin(x0, y1, res, res)
    xs = x0 + (np.arange(w) + 0.5) * res
    ys = y1 - (np.arange(h) + 0.5) * res
    XX, YY = np.meshgrid(xs, ys)
    mask_t = {n: contains_xy(p, XX, YY) for n, p in talhoes.items()}
    mascara = np.zeros((h, w), bool)
    for m in mask_t.values():
        mascara |= m
    if mascara.sum() == 0:
        return None

    def uma(its):
        try:
            with _ambiente():
                return ndvi_data(its, epsg, T, w, h)
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(trabalhadores) as ex:
        camadas = list(ex.map(uma, grupos))

    datas, aceitas = [], []
    for its, cam in zip(grupos, camadas):
        if cam is None:
            continue
        if np.isfinite(cam[mascara]).mean() >= min_limpo:
            datas.append(_data(its[0]))
            aceitas.append(cam)
    if len(aceitas) < 2:
        return None
    pilha = np.stack(aceitas)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pico = np.nanmax(pilha, axis=0)
    pico[~mascara] = np.nan

    serie, pico_t, cv_t = {}, {}, {}
    for n, m in mask_t.items():
        vals = []
        for cam in aceitas:
            v = cam[m]
            vals.append(float(np.nanmean(v)) if m.sum() and np.isfinite(v).mean() >= min_limpo else np.nan)
        serie[n] = vals
        p = pico[m]
        p = p[np.isfinite(p)]
        pico_t[n] = float(p.mean()) if len(p) else np.nan
        cv_t[n] = float(100 * p.std() / p.mean()) if len(p) and p.mean() > 0 else np.nan
    serie_faz = [float(np.nanmean(c[mascara])) for c in aceitas]
    return ResultadoNDVI(x0, y1, res, pico, mascara, datas, serie, serie_faz, pico_t, cv_t,
                         (inicio, hoje), len(grupos))


def classe_ndvi(v: float) -> int:
    for i, (lo, hi) in enumerate(FAIXAS_NDVI):
        if v < hi:
            return i
    return len(FAIXAS_NDVI) - 1
