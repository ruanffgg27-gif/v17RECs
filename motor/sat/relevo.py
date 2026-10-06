"""Relevo e comportamento provável da água (Copernicus DEM GLO-30).

O Copernicus DEM é um modelo de SUPERFÍCIE (inclui vegetação e estruturas) com 30 m de resolução.
Serve para reconhecimento e priorização; não substitui levantamento topográfico para terraços ou
drenagem, e tem limitações maiores em áreas muito planas. O escoamento é modelado com uma faixa de
entorno do talhão, porque a água que chega ao talhão vem também de fora dele.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from .. import externos
from . import fontes
from .base import Area, Grade, grade_da_area, ler_na_grade

# classes de declividade (Embrapa, 1979)
DECLIV_LIM = [0, 3, 8, 20, 45, 75, 1e9]
DECLIV_ROT = ["Plano (0–3%)", "Suave ondulado (3–8%)", "Ondulado (8–20%)", "Forte ondulado (20–45%)",
              "Montanhoso (45–75%)", "Escarpado (> 75%)"]
DECLIV_COR = ["#1A9850", "#91CF60", "#FEE08B", "#FC8D59", "#D73027", "#7F0000"]
POSICOES = ["Topo", "Encosta superior", "Encosta média", "Plano", "Encosta inferior", "Baixada"]
COR_POS = {"Topo": "#8C510A", "Encosta superior": "#D8B365", "Encosta média": "#F6E8C3", "Plano": "#E6E6E6",
           "Encosta inferior": "#80CDC1", "Baixada": "#01665E"}
EROSAO_ROT = ["Baixa", "Moderada", "Alta", "Muito alta"]
EROSAO_COR = ["#1A9850", "#FEE08B", "#FC8D59", "#B2182B"]
EROSAO_LIM = [0.5, 1.5, 3.0]
D8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


@dataclass
class Relevo:
    grade: Grade
    fonte: str
    dem: np.ndarray = field(repr=False)
    declividade: np.ndarray = field(repr=False)          # %
    posicao: np.ndarray = field(repr=False)              # índice em POSICOES
    acumulacao: np.ndarray = field(repr=False)           # área de contribuição (ha)
    direcao: np.ndarray = field(repr=False)              # índice D8 (−1 = sem saída)
    erosao: np.ndarray = field(repr=False)               # índice em EROSAO_ROT
    ls: np.ndarray = field(repr=False)
    estat: dict = field(default_factory=dict)
    decliv_ha: dict = field(default_factory=dict)
    posicao_ha: dict = field(default_factory=dict)
    erosao_ha: dict = field(default_factory=dict)
    perfis: list[dict] = field(default_factory=list)
    saidas: pd.DataFrame = field(default_factory=pd.DataFrame)     # pontos onde o escoamento deixa o talhão
    avisos: list[str] = field(default_factory=list)
    cobertura_usada: bool = False


def obter(area: Area, res: float = 30.0, buffer_m: float = 600.0, cobertura=None) -> Relevo | None:
    grade = grade_da_area(area, res, buffer_m=buffer_m, max_px=600_000)
    dem, fonte = None, ""
    try:
        partes = []
        for href in fontes.tiles_cop_dem(area.bbox_ll(buffer_m + 200)):
            try:
                partes.append(ler_na_grade(href, grade, "bilinear"))
            except Exception:  # noqa: BLE001
                continue
        if partes:
            dem = partes[0]
            for p in partes[1:]:
                dem = np.where(np.isfinite(dem), dem, p)
            if np.isfinite(dem).mean() < 0.95:
                dem = None
            else:
                fonte = "Copernicus DEM GLO-30 (ESA/Airbus)"
    except Exception:  # noqa: BLE001
        dem = None
    if dem is None:                                   # reserva: tiles de terreno (SRTM)
        from pyproj import Transformer
        X, Y = grade.malha()
        lon, lat = Transformer.from_crs(grade.epsg, 4326, always_xy=True).transform(X.ravel(), Y.ravel())
        try:
            v = externos.elevacao_tiles(lon, lat)
        except Exception:  # noqa: BLE001
            v = None
        if v is None:
            return None
        dem = np.asarray(v, float).reshape(X.shape)
        fonte = "SRTM (Terrain Tiles, AWS Open Data)"
    if np.isnan(dem).any():
        idx = ndimage.distance_transform_edt(np.isnan(dem), return_distances=False, return_indices=True)
        dem = dem[tuple(idx)]
    return analisar(area, grade, dem, fonte, cobertura)


# ------------------------------------------------------------------ cálculos
def declividade_pct(dem: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(dem, res)
    return 100 * np.hypot(gx, gy)


def preencher_depressoes(dem: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Priority-flood (Barnes et al., 2014): remove depressões para o escoamento seguir até a borda."""
    h, w = dem.shape
    z = dem.copy()
    visto = np.zeros(dem.shape, bool)
    fila: list = []
    for i in range(h):
        for j in (0, w - 1):
            heapq.heappush(fila, (z[i, j], i, j))
            visto[i, j] = True
    for j in range(1, w - 1):
        for i in (0, h - 1):
            heapq.heappush(fila, (z[i, j], i, j))
            visto[i, j] = True
    while fila:
        zc, i, j = heapq.heappop(fila)
        for di, dj in D8:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and not visto[a, b]:
                visto[a, b] = True
                if z[a, b] <= zc:
                    z[a, b] = zc + eps
                heapq.heappush(fila, (z[a, b], a, b))
    return z


def direcao_d8(z: np.ndarray, res: float) -> np.ndarray:
    h, w = z.shape
    melhor = np.full((h, w), -1, int)
    maior = np.zeros((h, w))
    zp = np.pad(z, 1, constant_values=np.inf)
    for k, (di, dj) in enumerate(D8):
        viz = zp[1 + di:1 + di + h, 1 + dj:1 + dj + w]
        dist = res * (np.sqrt(2) if di and dj else 1)
        queda = (z - viz) / dist
        m = queda > maior
        maior[m] = queda[m]
        melhor[m] = k
    return melhor


def acumulacao_d8(z: np.ndarray, direcao: np.ndarray) -> np.ndarray:
    """Número de células que drenam para cada célula (incluindo ela mesma)."""
    h, w = z.shape
    acc = np.ones(h * w)
    ordem = np.argsort(-z, axis=None)
    dirv = direcao.ravel()
    di = np.array([d[0] for d in D8])
    dj = np.array([d[1] for d in D8])
    for idx in ordem:
        k = dirv[idx]
        if k < 0:
            continue
        i, j = divmod(idx, w)
        a, b = i + di[k], j + dj[k]
        if 0 <= a < h and 0 <= b < w:
            acc[a * w + b] += acc[idx]
    return acc.reshape(h, w)


def posicao_paisagem(dem: np.ndarray, decliv: np.ndarray, res: float, raio_m: float = 150.0) -> np.ndarray:
    """Classes de posição pelo TPI (Weiss, 2001): elevação menos a média da vizinhança."""
    r = max(2, int(round(raio_m / res)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kern = ((xx ** 2 + yy ** 2) <= r * r).astype(float)
    kern[r, r] = 0
    kern /= kern.sum()
    media = ndimage.convolve(dem, kern, mode="nearest")
    tpi = dem - media
    dp = np.nanstd(tpi) or 1.0
    t = tpi / dp
    cls = np.full(dem.shape, 2)                         # encosta média
    cls[t > 1.0] = 0
    cls[(t > 0.5) & (t <= 1.0)] = 1
    cls[(np.abs(t) <= 0.5) & (decliv <= 3)] = 3
    cls[(t < -0.5) & (t >= -1.0)] = 4
    cls[t < -1.0] = 5
    return cls


def fator_ls(acc_celulas: np.ndarray, decliv_pct: np.ndarray, res: float) -> np.ndarray:
    """LS de Moore & Burch (1986): (As/22,13)^0,4 · (sen β / 0,0896)^1,3, com As = área específica."""
    As = acc_celulas * res
    beta = np.arctan(decliv_pct / 100)
    ls = (np.clip(As, 0, 3000) / 22.13) ** 0.4 * (np.sin(beta) / 0.0896) ** 1.3
    return ls


def _perfis(area: Area, grade: Grade, dem: np.ndarray, n: int = 120) -> list[dict]:
    from shapely.geometry import LineString
    ret = area.poligono.minimum_rotated_rectangle
    xs, ys = ret.exterior.coords.xy
    lados = [((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(4)]
    comp = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in lados]
    c = area.poligono.centroid
    perfis = []
    for nome, k in (("Maior eixo", int(np.argmax(comp))), ("Eixo transversal", int(np.argmin(comp)))):
        (xa, ya), (xb, yb) = lados[k]
        dx, dy = (xb - xa), (yb - ya)
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        linha = LineString([(c.x - ux * L, c.y - uy * L), (c.x + ux * L, c.y + uy * L)]).intersection(area.poligono)
        if linha.is_empty:
            continue
        segs = list(linha.geoms) if hasattr(linha, "geoms") else [linha]
        seg = max(segs, key=lambda s: s.length)
        d = np.linspace(0, seg.length, n)
        pts = [seg.interpolate(v) for v in d]
        cols = ((np.array([p.x for p in pts]) - grade.x0) / grade.res - 0.5)
        lins = ((grade.y1 - np.array([p.y for p in pts])) / grade.res - 0.5)
        z = ndimage.map_coordinates(dem, [lins, cols], order=1, mode="nearest")
        a0, a1 = seg.coords[0], seg.coords[-1]
        perfis.append({"nome": nome, "dist": d, "z": z, "inicio": a0, "fim": a1,
                       "decliv_media": float(100 * np.mean(np.abs(np.diff(z)) / np.diff(d)))})
    return perfis


def analisar(area: Area, grade: Grade, dem: np.ndarray, fonte: str, cobertura=None) -> Relevo:
    res = grade.res
    m = grade.mascara
    dem_s = ndimage.gaussian_filter(dem, 0.8)                    # atenua ruído do modelo de superfície
    decl = declividade_pct(dem_s, res)
    pos = posicao_paisagem(dem_s, decl, res)
    zf = preencher_depressoes(dem_s)
    dire = direcao_d8(zf, res)
    acc = acumulacao_d8(zf, dire)
    ls = fator_ls(acc, decl, res)
    ap = grade.area_pixel_ha
    R = Relevo(grade, fonte, dem_s, decl, pos, acc * ap, dire, np.digitize(ls, EROSAO_LIM), ls)
    aplicar_cobertura(R, cobertura)
    z = dem_s[m]
    R.estat = {"min": float(z.min()), "max": float(z.max()), "media": float(z.mean()),
               "amplitude": float(z.max() - z.min()), "decliv_media": float(decl[m].mean()),
               "decliv_p90": float(np.percentile(decl[m], 90))}
    cls = np.clip(np.searchsorted(DECLIV_LIM, decl[m], side="right") - 1, 0, len(DECLIV_ROT) - 1)
    R.decliv_ha = {r: float((cls == i).sum() * ap) for i, r in enumerate(DECLIV_ROT)}
    R.posicao_ha = {p: float((pos[m] == i).sum() * ap) for i, p in enumerate(POSICOES)}
    R.perfis = _perfis(area, grade, dem_s)
    R.saidas = saidas_do_escoamento(R)
    if R.estat["amplitude"] < 5:
        R.avisos.append("Relevo muito plano (amplitude < 5 m): direções de escoamento são pouco confiáveis "
                        "com um modelo de 30 m")
    return R


def aplicar_cobertura(R: Relevo, cobertura=None) -> None:
    """Suscetibilidade relativa à erosão = LS × fator de cobertura (NDVI médio observado, se houver).

    Mais cobertura vegetal ao longo do ano → menor suscetibilidade (fator de 1,3 a 0,7)."""
    fator_c = np.ones(R.ls.shape)
    R.cobertura_usada = False
    if cobertura is not None and np.isfinite(cobertura[R.grade.mascara]).mean() > 0.5:
        n = np.clip((np.nan_to_num(cobertura, nan=float(np.nanmedian(cobertura))) - 0.2) / 0.6, 0, 1)
        fator_c = 1.3 - 0.6 * n
        R.cobertura_usada = True
    R.erosao = np.digitize(R.ls * fator_c, EROSAO_LIM)
    m = R.grade.mascara
    R.erosao_ha = {e: float((R.erosao[m] == i).sum() * R.grade.area_pixel_ha) for i, e in enumerate(EROSAO_ROT)}


def saidas_do_escoamento(R: Relevo, min_ha: float = 2.0, maximo: int = 6) -> pd.DataFrame:
    """Pontos onde fluxos concentrados (área de contribuição ≥ min_ha) deixam o talhão."""
    g = R.grade
    m = g.mascara
    borda = m & ~ndimage.binary_erosion(m)
    linhas = []
    h, w = m.shape
    for i, j in zip(*np.where(borda & (R.acumulacao >= min_ha))):
        k = R.direcao[i, j]
        if k < 0:
            continue
        a, b = i + D8[k][0], j + D8[k][1]
        if 0 <= a < h and 0 <= b < w and not m[a, b]:
            lon, lat = g.lonlat(i, j)
            linhas.append({"Contribuição (ha)": float(R.acumulacao[i, j]), "Latitude": float(lat),
                           "Longitude": float(lon), "_lin": int(i), "_col": int(j)})
    if not linhas:
        return pd.DataFrame(columns=["Contribuição (ha)", "Latitude", "Longitude"])
    df = pd.DataFrame(linhas).sort_values("Contribuição (ha)", ascending=False)
    # uma saída por trecho de borda (evita pontos vizinhos repetidos)
    escolhidos = []
    for _, r in df.iterrows():
        if all(abs(r["_lin"] - e["_lin"]) + abs(r["_col"] - e["_col"]) > 5 for e in escolhidos):
            escolhidos.append(r)
        if len(escolhidos) >= maximo:
            break
    return pd.DataFrame(escolhidos).reset_index(drop=True)
