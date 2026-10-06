"""Acesso às fontes públicas (todas gratuitas e sem cadastro).

Endereços conferidos em set/2026:
- Sentinel-2 L2A (Copernicus/ESA): catálogo Earth Search (Element 84), COGs no AWS Open Data.
- Landsat 8/9 Coleção 2 Nível 2 (USGS/NASA): Microsoft Planetary Computer (token SAS anônimo).
- Copernicus DEM GLO-30: bucket público `copernicus-dem-30m` (AWS Open Data).
- MapBiomas Coleção 11 (1985–2025) e 10 (1985–2024): bucket público `mapbiomas-public` (Google Cloud).
- CHIRPS 2.0 diário (UCSB/CHC): API do ClimateSERV (NASA/SERVIR). Reserva: NASA POWER.
- NASA POWER diário: chuva e temperaturas desde 1981.
- SoilGrids 2.0 (ISRIC): API REST por ponto (limite de uso: ~5 consultas/min).
- PRODES (INPE/TerraBrasilis): WFS de desmatamento anual.

Cada função devolve None (ou lista vazia) quando a fonte não responde; quem chama registra o aviso.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "atria-satelites/1.0"}

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
MAPBIOMAS = [
    ("MapBiomas Coleção 11", "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection11/lulc/"
                             "coverage/brazil_coverage/brazil_coverage-col11_{ano}.tif", 1985, 2025),
    ("MapBiomas Coleção 10", "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/"
                             "coverage/brazil_coverage_{ano}.tif", 1985, 2024),
]
COP_DEM = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
CLIMATESERV = "https://climateserv.servirglobal.net/api/"
POWER_DIARIO = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_HORARIO = "https://power.larc.nasa.gov/api/temporal/hourly/point"
SOILGRIDS = "https://rest.isric.org/soilgrids/v2.0/properties/query"
TERRABRASILIS = "https://terrabrasilis.dpi.inpe.br/geoserver/{ws}/ows"


# ------------------------------------------------------------------ STAC
def buscar_stac(url: str, colecoes: list[str], bbox, inicio: date, fim: date, query: dict | None = None,
                paginas: int = 6, limite: int = 200, timeout: float = 40) -> list[dict]:
    corpo = {"collections": colecoes, "bbox": list(bbox), "limit": limite,
             "datetime": f"{inicio.isoformat()}T00:00:00Z/{fim.isoformat()}T23:59:59Z"}
    if query:
        corpo["query"] = query
    itens: list[dict] = []
    get_url = None                              # paginação por GET (token na URL do link "next")
    for _ in range(paginas):
        if get_url:
            r = requests.get(get_url, headers=UA, timeout=timeout)
        else:
            r = requests.post(url, json=corpo, headers=UA, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        itens += js.get("features", [])
        prox = next((l for l in js.get("links", []) if l.get("rel") == "next"), None)
        if not prox:
            break
        if prox.get("body") or str(prox.get("method", "")).upper() == "POST":
            corpo = {**corpo, **(prox.get("body") or {})}
            url, get_url = prox.get("href", url), None
        else:
            get_url = prox["href"]
    return itens


def data_item(item) -> date:
    return datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date()


def sentinel2(bbox, inicio: date, fim: date, nuvem_max: float = 60) -> list[dict]:
    """Itens Sentinel-2 L2A com as bandas usadas nos índices (B04, B05, B08, B11 e SCL)."""
    precisa = ("red", "nir", "rededge1", "swir16", "scl")
    for col in ("sentinel-2-c1-l2a", "sentinel-2-l2a"):
        for q in ({"eo:cloud_cover": {"lt": nuvem_max}}, None):
            try:
                itens = buscar_stac(EARTH_SEARCH, [col], bbox, inicio, fim, q)
            except Exception:  # noqa: BLE001
                continue
            itens = [i for i in itens if all(k in i.get("assets", {}) for k in precisa)
                     and i["properties"].get("eo:cloud_cover", 0) < nuvem_max]
            if itens:
                return itens
    return []


_tokens: dict[str, tuple[str, float]] = {}


def assinar_pc(href: str) -> str:
    """Assina um endereço do Planetary Computer com token SAS anônimo (válido por ~1 h)."""
    if ".blob.core.windows.net" not in href:
        return href
    conta = href.split("//", 1)[1].split(".", 1)[0]
    container = href.split(".blob.core.windows.net/", 1)[1].split("/", 1)[0]
    chave = f"{conta}/{container}"
    tok, validade = _tokens.get(chave, ("", 0.0))
    if not tok or time.time() > validade:
        r = requests.get(f"{PC_SAS}/{conta}/{container}", headers=UA, timeout=30)
        r.raise_for_status()
        tok = r.json()["token"]
        _tokens[chave] = (tok, time.time() + 40 * 60)
    return f"{href}?{tok}"


def landsat(bbox, inicio: date, fim: date, nuvem_max: float = 40) -> list[dict]:
    """Itens Landsat 8/9 Coleção 2 Nível 2 (temperatura de superfície: banda lwir11)."""
    q = {"eo:cloud_cover": {"lt": nuvem_max}, "platform": {"in": ["landsat-8", "landsat-9"]}}
    try:
        itens = buscar_stac(PC_STAC, ["landsat-c2-l2"], bbox, inicio, fim, q)
    except Exception:  # noqa: BLE001
        return []
    return [i for i in itens if "lwir11" in i.get("assets", {}) and "qa_pixel" in i.get("assets", {})]


# ------------------------------------------------------------------ DEM
def tiles_cop_dem(bbox) -> list[str]:
    lon0, lat0, lon1, lat1 = bbox
    hrefs = []
    for la in range(math.floor(lat0), math.floor(lat1) + 1):
        for lo in range(math.floor(lon0), math.floor(lon1) + 1):
            n = (f"Copernicus_DSM_COG_10_{'S' if la < 0 else 'N'}{abs(la):02d}_00_"
                 f"{'W' if lo < 0 else 'E'}{abs(lo):03d}_00_DEM")
            hrefs.append(COP_DEM.format(n=n))
    return hrefs


# ------------------------------------------------------------------ MapBiomas
def mapbiomas_hrefs() -> tuple[str, dict[int, str]]:
    """(nome da coleção, {ano: href}) da coleção mais recente que responder."""
    for nome, modelo, a0, a1 in MAPBIOMAS:
        teste = modelo.format(ano=a1)
        try:
            r = requests.head(teste, headers=UA, timeout=20, allow_redirects=True)
            if r.status_code == 200:
                return nome, {a: modelo.format(ano=a) for a in range(a0, a1 + 1)}
        except Exception:  # noqa: BLE001
            continue
    nome, modelo, a0, a1 = MAPBIOMAS[0]
    return nome, {a: modelo.format(ano=a) for a in range(a0, a1 + 1)}


# ------------------------------------------------------------------ clima
CHIRPS_ANOS_POR_PEDIDO = 10          # a API aceita no máximo 20 anos por pedido; blocos menores respondem mais rápido


def _chirps_pedido(geom: str, inicio: date, fim: date, timeout: float) -> dict | None:
    """Um pedido ao ClimateSERV (≤ 20 anos): {data: mm} ou None."""
    params = {"datatype": 0, "begintime": inicio.strftime("%m/%d/%Y"), "endtime": fim.strftime("%m/%d/%Y"),
              "intervaltype": 0, "operationtype": 5, "dateType_Category": "default",
              "isZip_CurrentDataType": "false", "geometry": geom}
    try:
        r = requests.post(CLIMATESERV + "submitDataRequest/", params=params, headers=UA, timeout=60)
        r.raise_for_status()
        pedido = json.loads(r.text)[0]
        t0 = time.time()
        while time.time() - t0 < timeout:
            p = requests.get(CLIMATESERV + "getDataRequestProgress/", params={"id": pedido}, headers=UA, timeout=30)
            prog = float(json.loads(p.text)[0])
            if prog >= 100:
                break
            if prog < 0:
                return None
            time.sleep(2)
        else:
            return None
        d = requests.get(CLIMATESERV + "getDataFromRequest/", params={"id": pedido}, headers=UA, timeout=120)
        dados = d.json().get("data", [])
    except Exception:  # noqa: BLE001
        return None
    serie = {}
    for g in dados:
        try:
            v = float(g["value"]["avg"])
            dia = datetime.strptime(g["date"], "%m/%d/%Y").date()
        except Exception:  # noqa: BLE001
            continue
        if v > -900:
            serie[pd.Timestamp(dia)] = v
    return serie or None


def blocos_de_anos(inicio: date, fim: date, anos: int = CHIRPS_ANOS_POR_PEDIDO) -> list[tuple[date, date]]:
    out, a = [], inicio
    while a <= fim:
        b = min(date(a.year + anos, a.month, a.day) - timedelta(days=1), fim)
        out.append((a, b))
        a = b + timedelta(days=1)
    return out


def chirps_climateserv(lon: float, lat: float, inicio: date, fim: date, timeout: float = 240,
                       meia_janela: float = 0.04, pedir=None) -> pd.Series | None:
    """Chuva diária CHIRPS (mm) — média na janela de ~9 × 9 km em torno do ponto (API ClimateSERV).

    O período é dividido em blocos de 10 anos pedidos em paralelo (a API recusa mais de 20 anos por pedido).
    Se algum bloco falhar mesmo após nova tentativa, devolve None (a série não fica com buracos)."""
    anel = [[lon - meia_janela, lat - meia_janela], [lon + meia_janela, lat - meia_janela],
            [lon + meia_janela, lat + meia_janela], [lon - meia_janela, lat + meia_janela],
            [lon - meia_janela, lat - meia_janela]]
    geom = json.dumps({"type": "Polygon", "coordinates": [anel]}).replace(" ", "")
    pedir = pedir or _chirps_pedido
    blocos = blocos_de_anos(inicio, fim)

    def um(b):
        for _ in range(2):
            r = pedir(geom, b[0], b[1], timeout)
            if r:
                return r
        return None

    with ThreadPoolExecutor(min(5, len(blocos))) as ex:
        partes = list(ex.map(um, blocos))
    if any(p is None for p in partes):
        return None
    serie = {k: v for p in partes for k, v in p.items()}
    if len(serie) < 365 * 5:
        return None
    return pd.Series(serie).sort_index()


def power_diario(lon: float, lat: float, inicio: date, fim: date,
                 parametros=("PRECTOTCORR", "T2M_MAX", "T2M_MIN")) -> pd.DataFrame | None:
    """Série diária da NASA POWER (chuva em mm, temperaturas em °C)."""
    try:
        r = requests.get(POWER_DIARIO, headers=UA, timeout=120, params={
            "parameters": ",".join(parametros), "community": "AG", "longitude": f"{lon:.4f}",
            "latitude": f"{lat:.4f}", "start": inicio.strftime("%Y%m%d"), "end": fim.strftime("%Y%m%d"),
            "format": "JSON"})
        r.raise_for_status()
        par = r.json()["properties"]["parameter"]
    except Exception:  # noqa: BLE001
        return None
    df = pd.DataFrame({k: pd.Series(v) for k, v in par.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.where(df > -900)
    return df.sort_index()


def power_horario(lon: float, lat: float, anos: list[int],
                  parametros=("WS2M", "WD10M", "T2M", "RH2M"), trabalhadores: int = 5) -> pd.DataFrame | None:
    """Série horária da NASA POWER (hora local, LST), um pedido por ano em paralelo.

    Vento a 2 m em m/s, direção de onde o vento sopra (graus, a 10 m), temperatura (°C) e umidade relativa (%)."""
    def um(ano):
        try:
            r = requests.get(POWER_HORARIO, headers=UA, timeout=180, params={
                "parameters": ",".join(parametros), "community": "AG", "longitude": f"{lon:.4f}",
                "latitude": f"{lat:.4f}", "start": f"{ano}0101", "end": f"{ano}1231", "format": "JSON",
                "time-standard": "LST"})
            r.raise_for_status()
            par = r.json()["properties"]["parameter"]
            df = pd.DataFrame({k: pd.Series(v) for k, v in par.items()})
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d%H")
            return df.where(df > -900)
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(min(trabalhadores, max(1, len(anos)))) as ex:
        partes = [p for p in ex.map(um, anos) if p is not None and len(p)]
    if not partes:
        return None
    return pd.concat(partes).sort_index()


# ------------------------------------------------------------------ solo
SOILGRIDS_PROPS = ["clay", "sand", "silt"]          # só granulometria: os demais atributos preditos
                                                  # ficavam longe da realidade de áreas manejadas
SOILGRIDS_PROF = ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]


def soilgrids_ponto(lon: float, lat: float, props=SOILGRIDS_PROPS, profundidades=SOILGRIDS_PROF,
                    tentativas: int = 3) -> list[dict]:
    """Propriedades preditas no ponto: [{propriedade, profundidade, media, q05, q95, unidade}]."""
    params = [("lon", f"{lon:.5f}"), ("lat", f"{lat:.5f}")] + [("property", p) for p in props] + \
             [("depth", d) for d in profundidades] + [("value", v) for v in ("mean", "Q0.05", "Q0.95")]
    for k in range(tentativas):
        try:
            r = requests.get(SOILGRIDS, params=params, headers=UA, timeout=60)
            if r.status_code == 429:                   # limite de uso: espera e tenta de novo
                time.sleep(15 * (k + 1))
                continue
            r.raise_for_status()
            camadas = r.json()["properties"]["layers"]
            break
        except Exception:  # noqa: BLE001
            time.sleep(3)
    else:
        return []
    out = []
    for c in camadas:
        um = c.get("unit_measure", {})
        fator = float(um.get("d_factor", 1) or 1)
        for d in c.get("depths", []):
            v = d.get("values", {})
            if v.get("mean") is None:
                continue
            out.append({"propriedade": c["name"], "profundidade": d.get("label", ""),
                        "media": v["mean"] / fator,
                        "q05": None if v.get("Q0.05") is None else v["Q0.05"] / fator,
                        "q95": None if v.get("Q0.95") is None else v["Q0.95"] / fator,
                        "unidade": um.get("target_units", "")})
    return out


# ------------------------------------------------------------------ PRODES
PRODES_BIOMAS = [("prodes-cerrado-nb", "Cerrado"), ("prodes-legal-amz", "Amazônia Legal"),
                 ("prodes-mata-atlantica-nb", "Mata Atlântica"), ("prodes-pantanal-nb", "Pantanal"),
                 ("prodes-caatinga-nb", "Caatinga"), ("prodes-pampa-nb", "Pampa")]


def prodes(bbox, timeout: float = 40) -> tuple[str, list[dict]] | None:
    """Polígonos de desmatamento anual (PRODES) na caixa: (bioma, [{'ano', 'geom' (GeoJSON)}])."""
    lon0, lat0, lon1, lat1 = bbox

    def consulta(ws_bioma):
        ws, bioma = ws_bioma
        try:
            r = requests.get(TERRABRASILIS.format(ws=ws), headers=UA, timeout=timeout, params={
                "service": "WFS", "version": "1.0.0", "request": "GetFeature",
                "typeName": f"{ws}:yearly_deforestation", "outputFormat": "application/json",
                "bbox": f"{lon0},{lat0},{lon1},{lat1},EPSG:4326", "srsName": "EPSG:4326", "maxFeatures": 500})
            r.raise_for_status()
            feats = r.json().get("features", [])
        except Exception:  # noqa: BLE001
            return None
        out = []
        for f in feats:
            p = f.get("properties", {})
            ano = p.get("year") or p.get("ano")
            try:
                ano = int(float(ano))
            except (TypeError, ValueError):
                continue
            out.append({"ano": ano, "geom": f.get("geometry"), "classe": p.get("main_class", "")})
        return bioma, out

    with ThreadPoolExecutor(3) as ex:
        res = [r for r in ex.map(consulta, PRODES_BIOMAS) if r is not None]
    if not res:
        return None
    com_dados = [r for r in res if r[1]]
    return com_dados[0] if com_dados else ("consultado", [])


def hoje() -> date:
    return date.today()


def anos_atras(anos: int, ref: date | None = None) -> date:
    ref = ref or date.today()
    return ref - timedelta(days=int(365.25 * anos))
