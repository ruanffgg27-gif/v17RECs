"""Área de estudo, grade de análise e leitura de rasters remotos (COG) por janela."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..geo import epsg_utm_sirgas, ler_kml, projetar


@dataclass
class Area:
    nome: str
    poligono_ll: Polygon | MultiPolygon       # lon/lat (WGS84)
    poligono: Polygon | MultiPolygon          # UTM (m)
    epsg: int

    @property
    def area_ha(self) -> float:
        return self.poligono.area / 1e4

    @property
    def centro(self) -> tuple[float, float]:
        c = self.poligono_ll.representative_point()
        return c.x, c.y

    def bbox_ll(self, buffer_m: float = 0.0) -> tuple[float, float, float, float]:
        g = projetar(self.poligono.buffer(buffer_m), self.epsg, inverso=True) if buffer_m else self.poligono_ll
        return g.bounds


def area_de_kml(conteudo: bytes, nome_arquivo: str, nome: str = "", indices: list[int] | None = None) -> Area:
    """Área a partir de um KML/KMZ de perímetro. Vários polígonos: usa os escolhidos (ou todos, unidos)."""
    k = ler_kml(conteudo, nome_arquivo)
    if not k.poligonos:
        raise ValueError("O arquivo não tem polígono de perímetro.")
    polys = [k.poligonos[i] for i in indices] if indices else k.poligonos
    ll = unary_union(polys)
    lon, lat = ll.representative_point().x, ll.representative_point().y
    epsg = epsg_utm_sirgas(lon, lat)
    return Area(nome or nome_arquivo.rsplit(".", 1)[0], ll, projetar(ll, epsg), epsg)


@dataclass
class Grade:
    """Grade regular em UTM (centro dos pixels), com máscara do talhão."""
    x0: float                  # borda esquerda
    y1: float                  # borda superior
    res: float
    largura: int
    altura: int
    epsg: int
    mascara: np.ndarray = field(repr=False)

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y1, self.res, self.res)

    @property
    def extent(self):
        return (self.x0, self.x0 + self.largura * self.res, self.y1 - self.altura * self.res, self.y1)

    @property
    def xs(self):
        return self.x0 + (np.arange(self.largura) + 0.5) * self.res

    @property
    def ys(self):
        return self.y1 - (np.arange(self.altura) + 0.5) * self.res

    def malha(self):
        return np.meshgrid(self.xs, self.ys)

    @property
    def area_pixel_ha(self) -> float:
        return self.res ** 2 / 1e4

    def lonlat(self, linhas, colunas):
        from pyproj import Transformer
        x = self.x0 + (np.asarray(colunas) + 0.5) * self.res
        y = self.y1 - (np.asarray(linhas) + 0.5) * self.res
        return Transformer.from_crs(self.epsg, 4326, always_xy=True).transform(x, y)


def grade_da_area(area: Area, res: float, buffer_m: float = 0.0, max_px: int = 2_000_000) -> Grade:
    geom = area.poligono
    x0, y0, x1, y1 = geom.buffer(buffer_m).bounds if buffer_m else geom.bounds
    x0, y0, x1, y1 = x0 - res, y0 - res, x1 + res, y1 + res
    while (x1 - x0) * (y1 - y0) / res ** 2 > max_px:
        res *= 1.25
    w, h = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    g = Grade(x0, y1, res, w, h, area.epsg, np.zeros((h, w), bool))
    X, Y = g.malha()
    g.mascara = contains_xy(geom, X, Y)
    return g


def ambiente_gdal():
    import rasterio
    return rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
                        AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1",
                        GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                        VSI_CACHE="TRUE", GDAL_CACHEMAX=256)


def ler_na_grade(href: str, grade: Grade, metodo: str = "bilinear", banda: int = 1,
                 nodata_origem=None) -> np.ndarray:
    """Lê um raster (local ou COG remoto) reprojetado na grade. Sem dado ou fora do raster → nan."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    res = {"bilinear": Resampling.bilinear, "media": Resampling.average, "vizinho": Resampling.nearest,
           "moda": Resampling.mode}[metodo]
    fora = -99999.0
    with ambiente_gdal():
        with rasterio.open(href) as src:
            nd = src.nodata if nodata_origem is None else nodata_origem
            with WarpedVRT(src, crs=f"EPSG:{grade.epsg}", transform=grade.transform, width=grade.largura,
                           height=grade.altura, resampling=res, src_nodata=nd, nodata=fora,
                           dtype="float32") as vrt:
                a = vrt.read(banda).astype(float)
    a[a == fora] = np.nan
    return a


def classes_area(valores: np.ndarray, mascara: np.ndarray, limites, area_px: float) -> list[float]:
    """Área (ha) em cada intervalo [limites[i], limites[i+1])."""
    v = valores[mascara & np.isfinite(valores)]
    cls = np.clip(np.searchsorted(limites, v, side="right") - 1, 0, len(limites) - 2)
    return list(np.bincount(cls, minlength=len(limites) - 1) * area_px)


def mapear_com_prazo(funcao, itens, trabalhadores: int = 8, prazo_s: float = 300.0) -> list:
    """Aplica `funcao` aos itens em paralelo e devolve o que terminou dentro do prazo (na ordem dos itens).

    Evita que uma fonte lenta trave o relatório inteiro: o que não chegou a tempo fica como None."""
    from concurrent.futures import ThreadPoolExecutor, wait
    itens = list(itens)
    ex = ThreadPoolExecutor(trabalhadores)
    futs = [ex.submit(funcao, it) for it in itens]
    feitos, _ = wait(futs, timeout=prazo_s)
    ex.shutdown(wait=False, cancel_futures=True)
    out = []
    for f in futs:
        try:
            out.append(f.result() if f in feitos else None)
        except Exception:  # noqa: BLE001
            out.append(None)
    return out
