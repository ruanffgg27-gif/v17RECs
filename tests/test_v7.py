"""Testes da versão 7: junção de laudos, atípicos locais, LOO rápido, altitude por tiles, NDVI e exportação."""
import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from motor import ler_laudo, recomendar
from motor.externos import amostrar_terrarium
from motor.juntar import juntar, normalizar_nome, sugerir_grupos
from motor.krigagem import MODELOS, _atipicos_locais, _loo_ok, _loo_ok_lento
from motor.ndvi import classe_ndvi, escolher_datas


def _laudo(prop, faz, talhoes):
    cab = ["N Lab", "Proprietário", "Propriedade", "Talhão", "Profundidade", "Ca", "Mg", "H+Al", "Al", "K",
           "P", "S", "Argila"]
    linhas = [[i, prop, faz, t, "0-20", 30, 10, 25, 0, 2, 20, 8, 400] for i, t in enumerate(talhoes)]
    buf = io.BytesIO()
    pd.DataFrame(linhas, columns=cab).to_excel(buf, index=False)
    return ler_laudo(buf.getvalue(), "x.xlsx")


def test_normalizar_nome():
    assert normalizar_nome("Faz. Santa Fé II") == "SANTA FE II"
    assert normalizar_nome("FAZENDA  DOZE") == normalizar_nome("Faz Doze") == "DOZE"


def test_sugerir_grupos_e_juntar():
    a = _laudo("JOSÉ TESTE", "FAZENDA DOZE", [1, 1, 2])
    b = _laudo("Jose Teste", "Faz. Doze", [3, 4])
    c = _laudo("MARIA", "SITIO BOA VISTA", [1])
    assert sugerir_grupos([a, b, c]) == [1, 1, 2]
    j, av = juntar([a, b], ["a.xlsx", "b.xlsx"])
    assert len(j.dados) == 5 and not av
    res = recomendar(j)
    assert "Arquivo" in res and set(res["Arquivo"]) == {"a.xlsx", "b.xlsx"}
    j2, av2 = juntar([a, c], ["a.xlsx", "c.xlsx"])           # talhão 1 repetido → renomeado
    assert "1 (B)" in set(j2.dados["talhao"]) and av2


def test_atipico_local_limitado_aos_vizinhos():
    x, y = np.meshgrid(np.arange(8) * 100.0, np.arange(8) * 100.0)
    xy = np.column_stack([x.ravel(), y.ravel()])
    z = 10 + 0.01 * xy[:, 0] + np.random.default_rng(0).normal(0, 0.3, len(xy))
    z[27] = 60                                                 # ponto isolado muito alto
    z2, n = _atipicos_locais(xy, z)
    assert n >= 1 and z2[27] < 20
    assert np.allclose(np.delete(z2, 27), np.delete(z, 27))    # os demais não mudam


def test_loo_fechado_igual_ao_lento():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 800, (40, 2))
    z = np.cos(xy[:, 1] / 150) + rng.normal(0, 0.1, 40)
    for f in MODELOS.values():
        par = np.array([0.02, 0.4, 350.0])
        assert np.allclose(_loo_ok(xy, z, f, par), _loo_ok_lento(xy, z, f, par), atol=1e-8)


def test_terrarium():
    z = 13
    dem = np.full((256, 256), 812.0)
    dem[:, 128:] = 830.0
    n = 2 ** z
    tx0, ty0 = 2900, 4700
    lon = lambda c: (tx0 + c / 256) / n * 360 - 180                                     # noqa: E731
    lat = lambda r: np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (ty0 + r / 256) / n))))  # noqa: E731
    v = amostrar_terrarium(dem, z, tx0, ty0, [lon(10.5), lon(250.5)], [lat(100.5), lat(100.5)])
    assert v[0] == pytest.approx(812) and v[1] == pytest.approx(830)


def test_ndvi_datas_uma_por_mes():
    def item(d, nuv):
        return {"properties": {"datetime": f"{d}T13:00:00Z", "eo:cloud_cover": nuv}, "assets": {}}
    itens = [item("2026-01-03", 40), item("2026-01-18", 5), item("2026-01-18", 15), item("2026-02-10", 30)]
    g = escolher_datas(itens)
    assert len(g) == 2 and len(g[0]) == 2                                 # 18/jan (2 tiles) e 10/fev
    assert classe_ndvi(0.1) == 0 and classe_ndvi(0.85) == 6 and classe_ndvi(0.65) == 4
