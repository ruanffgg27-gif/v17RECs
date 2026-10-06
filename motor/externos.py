"""Dados externos consultados na hora de montar o book (precisam de internet).

Tudo aqui é opcional: se a consulta falhar, a função devolve None e o book segue
sem aquela informação (com aviso).

- Imagem de satélite: mosaico de tiles XYZ (Esri World Imagery ou Sentinel-2 cloudless/EOX).
- Elevação: tiles de terreno Terrarium (SRTM, AWS Open Data) — rápido; reserva: OpenTopoData (SRTM 30 m)
  e Open-Meteo (Copernicus 90 m), por pontos.
- Clima: NASA POWER (normais mensais; dados públicos da NASA).
"""
from __future__ import annotations

import io
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import requests.adapters
from PIL import Image

UA = {"User-Agent": "recomenda-solo/1.0 (relatorio de fertilidade)"}

FONTES_SATELITE = {
    "Google (Map Tiles API)": {
        "url": "https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?session={sessao}&key={chave}",
        "credito": "Imagem © Google", "zmax": 18, "google": True},
    "Esri World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "credito": "Imagem: Esri, Maxar, Earthstar Geographics", "zmax": 18},
    "Sentinel-2 cloudless (EOX)": {
        "url": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2016_3857/default/g/{z}/{y}/{x}.jpg",
        "credito": "Sentinel-2 cloudless 2016 by EOX IT Services GmbH (CC BY 4.0)", "zmax": 15},
}


# ------------------------------------------------------------------ satélite
def _merc(lon, lat):
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) * 20037508.34 / math.pi
    return x, y


def _tile(lon, lat, z):
    n = 2 ** z
    xt = (lon + 180) / 360 * n
    yt = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return xt, yt


def _sessao_google(chave: str, timeout: float) -> str:
    r = requests.post(f"https://tile.googleapis.com/v1/createSession?key={chave}",
                      json={"mapType": "satellite", "language": "pt-BR", "region": "BR"}, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()["session"]


def _credito_google(chave, sessao, z, lon0, lat0, lon1, lat1, timeout):
    try:
        r = requests.get("https://tile.googleapis.com/tile/v1/viewport", headers=UA, timeout=timeout,
                         params={"session": sessao, "key": chave, "zoom": z, "north": lat1, "south": lat0,
                                 "east": lon1, "west": lon0})
        r.raise_for_status()
        return r.json().get("copyright") or "Imagem © Google"
    except Exception:  # noqa: BLE001
        return "Imagem © Google"


def _mosaico(url: str, z: int, tx0: int, ty0: int, tx1: int, ty1: int, extra: dict | None = None,
             timeout: float = 10, modo: str = "RGB", fundo=(230, 230, 230), trabalhadores: int = 8):
    """Baixa os tiles XYZ em paralelo e monta o mosaico. Devolve (array, nº de tiles obtidos)."""
    extra = extra or {}
    W, H = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    mosaico = Image.new(modo, (W, H), fundo)
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=trabalhadores, pool_maxsize=trabalhadores))

    def baixar(txy):
        tx, ty = txy
        try:
            r = s.get(url.format(z=z, x=tx, y=ty, **extra), headers=UA, timeout=timeout)
            r.raise_for_status()
            return tx, ty, Image.open(io.BytesIO(r.content)).convert(modo)
        except Exception:  # noqa: BLE001
            return tx, ty, None

    tiles = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
    ok = 0
    with ThreadPoolExecutor(trabalhadores) as ex:
        for tx, ty, img in ex.map(baixar, tiles):
            if img is not None:
                mosaico.paste(img, ((tx - tx0) * 256, (ty - ty0) * 256))
                ok += 1
    return np.asarray(mosaico), ok


def imagem_satelite(lon0, lat0, lon1, lat1, fonte: str = "Esri World Imagery", max_px: int = 2400,
                    timeout: float = 10, chave_google: str | None = None):
    """Mosaico para a caixa (lon0,lat0)-(lon1,lat1). Devolve (imagem RGB, extent em EPSG:3857, crédito).

    Google: usa a Map Tiles API oficial (precisa de chave em `google_maps_key` nos Secrets).
    """
    cfg = FONTES_SATELITE[fonte]
    extra = {}
    if cfg.get("google"):
        if not chave_google:
            return None
        extra = {"sessao": _sessao_google(chave_google, timeout), "chave": chave_google}
    z = cfg["zmax"]
    while z > 3:
        x0, y0 = _tile(lon0, lat1, z)
        x1, y1 = _tile(lon1, lat0, z)
        if (x1 - x0) * 256 <= max_px and (y1 - y0) * 256 <= max_px:
            break
        z -= 1
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    mosaico, ok = _mosaico(cfg["url"], z, tx0, ty0, tx1, ty1, extra, timeout)
    if ok == 0:
        return None
    mosaico = Image.fromarray(mosaico)
    n = 2 ** z
    lon_a, lon_b = tx0 / n * 360 - 180, (tx1 + 1) / n * 360 - 180
    lat_a = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty0 / n))))
    lat_b = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty1 + 1) / n))))
    xa, ya = _merc(lon_a, lat_a)
    xb, yb = _merc(lon_b, lat_b)
    credito = cfg["credito"]
    if cfg.get("google"):
        credito = _credito_google(extra["chave"], extra["sessao"], z, lon0, lat0, lon1, lat1, timeout)
    return np.asarray(mosaico), (xa, xb, yb, ya), credito


# ------------------------------------------------------------------ elevação
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CREDITO_ALTITUDE = "Altitude: SRTM/NASA via Terrain Tiles (Mapzen, AWS Open Data)"


def elevacao_tiles(lons, lats, zmax: int = 13, max_tiles: int = 49, timeout: float = 15) -> np.ndarray | None:
    """Altitude (m) em cada ponto, lida de tiles de terreno (formato Terrarium; SRTM ~30 m no Brasil).

    Poucas requisições (uma por tile de 256×256 px, baixadas em paralelo) em vez de milhares de pontos:
    é a forma rápida. Zoom 13 ≈ 19 m por pixel; para áreas grandes o zoom é reduzido.
    """
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    if not len(lons):
        return None
    lon0, lon1, lat0, lat1 = lons.min(), lons.max(), lats.min(), lats.max()
    z = zmax
    while z > 8:
        x0, y0 = _tile(lon0, lat1, z)
        x1, y1 = _tile(lon1, lat0, z)
        if (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1) <= max_tiles:
            break
        z -= 1
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    rgb, ok = _mosaico(TERRAIN_URL, z, tx0, ty0, tx1, ty1, timeout=timeout, fundo=(0, 0, 0))
    if ok < (tx1 - tx0 + 1) * (ty1 - ty0 + 1):
        return None
    rgb = rgb.astype(float)
    dem = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
    return amostrar_terrarium(dem, z, tx0, ty0, lons, lats)


def amostrar_terrarium(dem: np.ndarray, z: int, tx0: int, ty0: int, lons, lats) -> np.ndarray:
    """Interpolação bilinear do mosaico (pixels em coordenadas de tile XYZ) nos pontos lon/lat."""
    from scipy.ndimage import map_coordinates
    n = 2 ** z
    lats = np.asarray(lats, float)
    col = ((np.asarray(lons, float) + 180) / 360 * n - tx0) * 256 - 0.5
    lin = ((1 - np.arcsinh(np.tan(np.radians(lats))) / np.pi) / 2 * n - ty0) * 256 - 0.5
    out = map_coordinates(dem, [lin, col], order=1, mode="nearest")
    out[(out < -500) | (out > 9000)] = np.nan
    return out


def elevacao(lons, lats, timeout: float = 20) -> np.ndarray | None:
    """Altitude (m) para cada ponto. Tenta OpenTopoData (SRTM 30 m) e depois Open-Meteo."""
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    out = np.full(len(lons), np.nan)
    try:
        for i in range(0, len(lons), 100):
            loc = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in zip(lats[i:i + 100], lons[i:i + 100]))
            r = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={loc}", headers=UA,
                             timeout=timeout)
            r.raise_for_status()
            res = r.json().get("results", [])
            out[i:i + len(res)] = [np.nan if x.get("elevation") is None else x["elevation"] for x in res]
            time.sleep(1.05)                       # limite do serviço público: 1 consulta/s
        if np.isfinite(out).mean() > 0.8:
            return out
    except Exception:  # noqa: BLE001
        pass
    try:
        for i in range(0, len(lons), 100):
            la = ",".join(f"{v:.6f}" for v in lats[i:i + 100])
            lo = ",".join(f"{v:.6f}" for v in lons[i:i + 100])
            r = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={la}&longitude={lo}",
                             headers=UA, timeout=timeout)
            r.raise_for_status()
            out[i:i + 100] = r.json()["elevation"]
        return out if np.isfinite(out).mean() > 0.8 else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------- clima
MESES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DIAS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def clima(lon, lat, timeout: float = 30) -> dict | None:
    """Normais mensais da NASA POWER: chuva (mm/mês), temperatura média, máxima e mínima (°C)."""
    url = ("https://power.larc.nasa.gov/api/temporal/climatology/point?parameters="
           "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR&community=AG"
           f"&longitude={lon:.4f}&latitude={lat:.4f}&format=JSON")
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        p = r.json()["properties"]["parameter"]
        chuva = [p["PRECTOTCORR"][m] * d for m, d in zip(MESES, DIAS)]
        res = {"chuva": chuva, "tmed": [p["T2M"][m] for m in MESES],
               "tmax": [p["T2M_MAX"][m] for m in MESES], "tmin": [p["T2M_MIN"][m] for m in MESES],
               "fonte": "NASA POWER (climatologia 2001–2020)"}
        if any(v is None or v < -900 for k in ("chuva", "tmed") for v in res[k]):
            return None
        return res
    except Exception:  # noqa: BLE001
        return None
