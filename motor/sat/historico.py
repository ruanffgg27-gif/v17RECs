"""Histórico de uso e cobertura (MapBiomas) e desmatamento (PRODES): a "biografia do talhão"."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fontes
from .base import Area, Grade, grade_da_area, ler_na_grade

# legenda MapBiomas (coleções 9–11): código → (nome, grupo)
LEGENDA = {
    3: ("Formação florestal", "Vegetação nativa"), 4: ("Formação savânica", "Vegetação nativa"),
    5: ("Mangue", "Vegetação nativa"), 6: ("Floresta alagável", "Vegetação nativa"),
    49: ("Restinga arbórea", "Vegetação nativa"), 11: ("Campo alagado e área pantanosa", "Vegetação nativa"),
    12: ("Formação campestre", "Vegetação nativa"), 32: ("Apicum", "Vegetação nativa"),
    29: ("Afloramento rochoso", "Vegetação nativa"), 50: ("Restinga herbácea", "Vegetação nativa"),
    13: ("Outra formação não florestal", "Vegetação nativa"),
    15: ("Pastagem", "Pastagem"),
    18: ("Agricultura", "Agricultura"), 19: ("Lavoura temporária", "Agricultura"), 39: ("Soja", "Agricultura"),
    20: ("Cana", "Agricultura"), 40: ("Arroz", "Agricultura"), 62: ("Algodão", "Agricultura"),
    41: ("Outras lavouras temporárias", "Agricultura"), 36: ("Lavoura perene", "Agricultura"),
    46: ("Café", "Agricultura"), 47: ("Citrus", "Agricultura"), 35: ("Dendê", "Agricultura"),
    48: ("Outras lavouras perenes", "Agricultura"),
    9: ("Silvicultura", "Silvicultura"), 21: ("Mosaico de usos", "Mosaico de usos"),
    23: ("Praia, duna e areal", "Área não vegetada"), 24: ("Área urbanizada", "Área não vegetada"),
    30: ("Mineração", "Área não vegetada"), 25: ("Outra área não vegetada", "Área não vegetada"),
    75: ("Usina fotovoltaica", "Área não vegetada"),
    33: ("Rio, lago e oceano", "Água"), 31: ("Aquicultura", "Água"),
    27: ("Não observado", "Não observado"),
}
GRUPOS = ["Vegetação nativa", "Pastagem", "Agricultura", "Silvicultura", "Mosaico de usos", "Água",
          "Área não vegetada", "Não observado", "Outros"]
COR_GRUPO = {"Vegetação nativa": "#1F8D49", "Pastagem": "#EDDE8E", "Agricultura": "#E974ED",
             "Silvicultura": "#7A5900", "Mosaico de usos": "#FFEFC3", "Água": "#2532E4",
             "Área não vegetada": "#DB4D4F", "Não observado": "#CCCCCC", "Outros": "#999999"}


def grupo_de(codigo) -> str:
    return LEGENDA.get(int(codigo), ("", "Outros"))[1]


def nome_de(codigo) -> str:
    return LEGENDA.get(int(codigo), (f"Classe {int(codigo)}", ""))[0]


@dataclass
class Historico:
    colecao: str
    grade: Grade
    mapas: dict[int, np.ndarray] = field(repr=False)          # ano → classes (nan fora)
    grupos_ha: pd.DataFrame = field(default_factory=pd.DataFrame)   # ano × grupo (ha)
    classes_ha: pd.DataFrame = field(default_factory=pd.DataFrame)  # ano × classe (ha)
    eventos: list[tuple[str, str]] = field(default_factory=list)    # (título, descrição)
    prodes: pd.DataFrame | None = None                               # ano, área (ha)
    prodes_bioma: str = ""
    avisos: list[str] = field(default_factory=list)


def composicao(mapas: dict[int, np.ndarray], mascara: np.ndarray, area_px: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    g_linhas, c_linhas = {}, {}
    for ano, m in sorted(mapas.items()):
        v = m[mascara & np.isfinite(m)].astype(int)
        cod, n = np.unique(v, return_counts=True)
        c_linhas[ano] = {nome_de(c): k * area_px for c, k in zip(cod, n)}
        g = {}
        for c, k in zip(cod, n):
            g[grupo_de(c)] = g.get(grupo_de(c), 0) + k * area_px
        g_linhas[ano] = g
    # sort_index: no pandas 2.x o from_dict com dicionários de colunas diferentes pode desordenar os anos
    # (ex.: um ano sem a classe da 1ª coluna vai para o fim) — gráfico e "uso atual" sairiam errados
    grupos = pd.DataFrame.from_dict(g_linhas, orient="index").reindex(columns=GRUPOS).fillna(0).sort_index()
    grupos = grupos.loc[:, (grupos > 0).any()]
    classes = pd.DataFrame.from_dict(c_linhas, orient="index").fillna(0).sort_index()
    return grupos, classes


def eventos(grupos_ha: pd.DataFrame, classes_ha: pd.DataFrame, area_ha: float) -> list[tuple[str, str]]:
    """Linha do tempo interpretada a partir das proporções anuais."""
    if grupos_ha.empty:
        return []
    pct = grupos_ha.div(grupos_ha.sum(axis=1), axis=0) * 100
    anos = list(pct.index)
    a0, a1 = anos[0], anos[-1]
    ev = []
    nat = pct.get("Vegetação nativa", pd.Series(0, index=anos))
    antr = pct.get("Pastagem", 0) + pct.get("Agricultura", 0) + pct.get("Silvicultura", 0) + \
        pct.get("Mosaico de usos", 0)
    if nat.iloc[0] >= 50:
        abaixo = [a for a in anos if nat[a] < 50]
        if abaixo:
            ano_ab = abaixo[0]
            inicio = next((a for a in anos if nat[a] <= nat.iloc[0] - 10), ano_ab)
            ev.append(("Abertura provável", f"a vegetação nativa deixou de ser majoritária em {ano_ab} "
                                            f"(início da conversão por volta de {inicio}; em {a0} ocupava "
                                            f"{nat.iloc[0]:.0f}% da área)"))
        else:
            ev.append(("Vegetação nativa", f"majoritária em toda a série ({a0}–{a1}); hoje {nat.iloc[-1]:.0f}% "
                                           "da área"))
    else:
        ev.append(("Área já aberta em " + str(a0), f"em {a0} a vegetação nativa ocupava {nat.iloc[0]:.0f}% e os "
                                                   f"usos agropecuários {float(antr.iloc[0]):.0f}% da área"))
    pas = pct.get("Pastagem", pd.Series(0, index=anos))
    agr = pct.get("Agricultura", pd.Series(0, index=anos))
    if (pas >= 50).any():
        anos_pas = [a for a in anos if pas[a] >= 50]
        ev.append(("Fase de pastagem", f"pastagem predominante em {len(anos_pas)} anos "
                                       f"({anos_pas[0]}–{anos_pas[-1]})"))
        depois = [a for a in anos if a > anos_pas[0] and agr[a] >= 50]
        if depois:
            ev.append(("Conversão para lavoura", f"agricultura passou a predominar em {depois[0]}"))
    elif (agr >= 50).any():
        ev.append(("Uso agrícola", f"agricultura predominante desde {next(a for a in anos if agr[a] >= 50)}"))
    agua = pct.get("Água", pd.Series(0, index=anos))
    lim = max(0.5, 100 * 0.3 / max(area_ha, 1e-6))
    anos_agua = [a for a in anos if agua[a] >= lim]
    if anos_agua:
        ev.append(("Água superficial detectada", f"em {len(anos_agua)} ano(s): " + _faixas(anos_agua)))
    # culturas identificadas nos últimos anos
    ult = classes_ha.tail(5)
    culturas = [c for c in ("Soja", "Cana", "Arroz", "Algodão", "Café", "Citrus", "Outras lavouras temporárias")
                if c in ult and ult[c].sum() > 0]
    if culturas:
        ev.append(("Culturas mapeadas (últimos 5 anos)", ", ".join(culturas)))
    dom = pct.iloc[-1].idxmax()
    ev.append((f"Uso em {a1}", f"{dom.lower()} ({pct.iloc[-1].max():.0f}%)"
                               + (f"; vegetação nativa remanescente {nat.iloc[-1]:.0f}% "
                                  f"({grupos_ha['Vegetação nativa'].iloc[-1]:.1f} ha)"
                                  if "Vegetação nativa" in grupos_ha and nat.iloc[-1] > 0.5 else "")))
    return ev


def _faixas(anos: list[int]) -> str:
    """[2001, 2002, 2003, 2010] → '2001–2003, 2010'."""
    out, ini, ant = [], anos[0], anos[0]
    for a in anos[1:] + [None]:
        if a is not None and a == ant + 1:
            ant = a
            continue
        out.append(f"{ini}–{ant}" if ant != ini else f"{ini}")
        if a is not None:
            ini = ant = a
    return ", ".join(out)


def obter(area: Area, trabalhadores: int = 8, progresso=None) -> Historico | None:
    """Lê todos os anos do MapBiomas na área (grade de 30 m) e consulta o PRODES."""
    grade = grade_da_area(area, 30.0, buffer_m=60)
    colecao, hrefs = fontes.mapbiomas_hrefs()

    def ler(item):
        ano, href = item
        try:
            return ano, ler_na_grade(href, grade, "vizinho", nodata_origem=0)
        except Exception:  # noqa: BLE001
            return ano, None

    from .base import mapear_com_prazo
    lidos = dict(r for r in mapear_com_prazo(ler, hrefs.items(), trabalhadores, prazo_s=240) if r is not None)
    mapas = {a: lidos[a] for a in sorted(lidos)
             if lidos[a] is not None and np.isfinite(lidos[a][grade.mascara]).mean() > 0.5}
    if len(mapas) < 5:
        return None
    return montar(area, grade, colecao, mapas)


def montar(area: Area, grade: Grade, colecao: str, mapas: dict[int, np.ndarray],
           prodes_ok: bool = True) -> Historico:
    mapas = dict(sorted(mapas.items()))
    g, c = composicao(mapas, grade.mascara, grade.area_pixel_ha)
    h = Historico(colecao, grade, mapas, g, c, eventos(g, c, area.area_ha))
    faltam = sorted(set(range(min(mapas), max(mapas) + 1)) - set(mapas))
    if faltam:
        h.avisos.append(f"MapBiomas: {len(faltam)} ano(s) não lidos ({_faixas(faltam)})")
    if prodes_ok:
        try:
            r = fontes.prodes(area.bbox_ll(100))
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            h.avisos.append("PRODES (INPE): serviço indisponível no momento")
        else:
            h.prodes_bioma, feats = r
            h.prodes = prodes_na_area(area, feats)
    return h


def prodes_na_area(area: Area, feats: list[dict]) -> pd.DataFrame:
    from shapely.geometry import shape
    from ..geo import projetar
    linhas = []
    for f in feats:
        try:
            g = projetar(shape(f["geom"]), area.epsg).intersection(area.poligono)
        except Exception:  # noqa: BLE001
            continue
        if not g.is_empty and g.area > 100:
            linhas.append({"Ano": f["ano"], "Área (ha)": g.area / 1e4})
    if not linhas:
        return pd.DataFrame(columns=["Ano", "Área (ha)"])
    return pd.DataFrame(linhas).groupby("Ano", as_index=False).sum().sort_values("Ano")
