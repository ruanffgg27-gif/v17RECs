"""Testes do módulo Satélites (análises com dados sintéticos; sem internet)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from motor.sat import clima, fontes, historico, lavoura, relevo, termico
from motor.sat.base import Area, Grade, grade_da_area
from motor.sat.diagnostico import reamostrar


def _area(lado=1000.0):
    p = box(500000, 7500000, 500000 + lado, 7500000 + lado)
    from motor.geo import projetar
    return Area("teste", projetar(p, 31982, inverso=True), p, 31982)


def test_nomes_dos_tiles_copernicus():
    h = fontes.tiles_cop_dem((-54.6, -22.4, -54.2, -22.1))
    assert h == ["https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S23_00_W055_00_DEM/"
                 "Copernicus_DSM_COG_10_S23_00_W055_00_DEM.tif"]
    assert len(fontes.tiles_cop_dem((-54.6, -22.4, -53.8, -21.9))) == 4


def test_lavoura_estabilidade_e_pontos():
    A = _area()
    g = grade_da_area(A, 10.0)
    X, Y = g.malha()
    mancha = ((X - 500250) ** 2 + (Y - 7500700) ** 2) < 150 ** 2
    cenas = []
    rng = np.random.default_rng(0)
    for k in range(36):
        d = date(2023, 7, 15) + timedelta(days=30 * k)
        fase = 0.3 + 0.5 * max(0.0, np.sin(2 * np.pi * (d.timetuple().tm_yday - 280) / 365))
        ndvi = np.full(X.shape, fase) + rng.normal(0, 0.01, X.shape)
        ndvi[mancha] -= 0.2 * fase
        cenas.append(lavoura.Cena(d, {"ndvi": ndvi, "ndre": ndvi * 0.6, "ndmi": ndvi * 0.5, "msavi2": ndvi}))
    L = lavoura.analisar(g, cenas)
    assert len(L.picos) >= 2
    assert L.estab_area["Baixo e estável"] == pytest.approx(np.pi * 0.15 ** 2 * 100, rel=0.35)
    assert len(L.pontos) >= 1 and L.pontos.iloc[0]["Tipo"] == "Baixo vigor persistente"
    lon, lat = L.pontos.iloc[0][["Longitude", "Latitude"]]
    x, y = __import__("pyproj").Transformer.from_crs(4326, 31982, always_xy=True).transform(lon, lat)
    assert abs(x - 500250) < 60 and abs(y - 7500700) < 60
    assert b"<Placemark>" in lavoura.kml_pontos(L.pontos)


def test_relevo_escoa_para_o_lado_baixo():
    A = _area(900)
    g = grade_da_area(A, 30.0, buffer_m=300)
    X, Y = g.malha()
    z = 500 + 0.05 * (X - X.min())                      # sobe para leste → água vai para oeste
    R = relevo.analisar(A, g, z, "teste")
    assert R.estat["amplitude"] > 20
    assert R.decliv_ha["Suave ondulado (3–8%)"] == pytest.approx(A.area_ha, rel=0.05)
    j_max = np.unravel_index(np.argmax(R.acumulacao), R.acumulacao.shape)[1]
    assert j_max == 0                                   # acumulação máxima na borda oeste
    assert len(R.saidas) >= 1


def test_clima_veranicos_e_estacao():
    idx = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    chuva = pd.Series(0.0, index=idx)
    chuva[(idx.month >= 11) | (idx.month <= 2)] = 8.0          # chove todo dia de nov a fev
    for a in range(1990, 2021):                               # veranico de 12 dias em janeiro
        chuva[pd.Timestamp(a, 1, 10):pd.Timestamp(a, 1, 21)] = 0
    df = pd.DataFrame({"chuva": chuva, "tmax": 30.0, "tmin": 18.0})
    C = clima.analisar(df)
    v = C.veranicos.set_index("Mês")
    assert v.loc["Jan", "≥ 10 dias secos (%)"] == pytest.approx(100)
    assert v.loc["Dez", "≥ 10 dias secos (%)"] == pytest.approx(0)
    est = clima.texto_estacao(C)
    assert est["Início"]["mediana"] in ("30/10", "31/10", "01/11")
    assert C.anual["Chuva (mm)"].iloc[0] == pytest.approx((120 - 12) * 8, rel=0.02)


def test_historico_eventos():
    anos = list(range(1985, 2025))
    g = pd.DataFrame({"Vegetação nativa": [100 if a < 2000 else 10 for a in anos],
                      "Pastagem": [0 if a < 2000 else (90 if a < 2010 else 0) for a in anos],
                      "Agricultura": [0 if a < 2010 else 90 for a in anos]}, index=anos, dtype=float)
    ev = dict(historico.eventos(g, pd.DataFrame(index=anos), 100))
    assert "2000" in ev["Abertura provável"]
    assert "2010" in ev["Conversão para lavoura"]


def test_qa_landsat_e_reamostragem():
    qa = np.array([21824, 22280, 23888, 1], float)            # claro, nuvem, sombra/nuvem, sem dado
    assert termico.mascara_qa(qa).tolist() == [True, False, False, False]
    A = _area()
    g1, g2 = grade_da_area(A, 10.0), grade_da_area(A, 30.0)
    X, Y = g1.malha()
    v = reamostrar((X - X.min()) / 10, g1, g2)
    X2, _ = g2.malha()
    ok = np.isfinite(v)
    assert np.allclose(v[ok], ((X2 - X.min()) / 10)[ok], atol=1.0)


def test_historico_anos_em_ordem_mesmo_com_classes_diferentes():
    """No pandas 2.x o from_dict mandava para o fim o ano sem a classe da 1ª coluna (bug visto em campo)."""
    g = grade_da_area(_area(300), 30.0)
    mapas = {}
    for ano in range(1985, 2026):
        m = np.full(g.mascara.shape, 39.0)
        if ano != 2017:
            m[0, :] = 3.0
        mapas[ano] = m
    grupos, classes = historico.composicao(dict(sorted(mapas.items(), key=lambda kv: -kv[0])), g.mascara, 0.09)
    assert list(grupos.index) == list(range(1985, 2026))
    assert list(classes.index) == list(range(1985, 2026))


def test_chirps_em_blocos_de_ate_10_anos():
    blocos = fontes.blocos_de_anos(date(1981, 1, 1), date(2026, 9, 26))
    assert all((b - a).days < 366 * 10 for a, b in blocos)
    assert blocos[0][0] == date(1981, 1, 1) and blocos[-1][1] == date(2026, 9, 26)
    pedidos = []

    def falso(geom, a, b, t):
        pedidos.append((a, b))
        return {pd.Timestamp(d): 1.0 for d in pd.date_range(a, b)}
    s = fontes.chirps_climateserv(-51, -22, date(1981, 1, 1), date(2026, 9, 26), pedir=falso)
    assert len(pedidos) == len(blocos) and s.index.is_unique and len(s) == (date(2026, 9, 26) - date(1981, 1, 1)).days + 1
    assert fontes.chirps_climateserv(-51, -22, date(1981, 1, 1), date(2026, 9, 26), pedir=lambda *a: None) is None


def test_vento_rosa_e_janelas():
    from motor.sat import vento
    idx = pd.date_range("2022-01-01", "2023-12-31 23:00", freq="h")
    h = idx.hour.values
    dia = (h >= 7) & (h <= 17)
    df = pd.DataFrame({"vel": np.where(dia, 14.0, 6.0), "dir": np.full(len(idx), 135.0),
                       "t": np.where(dia, 32.0, 22.0), "ur": np.where(dia, 40.0, 80.0)}, index=idx)
    v = vento.analisar(df)
    assert v.predominante == "SE" and v.predominante_pct == pytest.approx(100)
    assert v.rosa.values.sum() == pytest.approx(100)
    assert v.janela.loc[1, 6] == pytest.approx(100) and v.janela.loc[1, 12] == pytest.approx(0)
    assert "18h" in vento.melhores_horas(v)
    assert set(vento.tabelas(v)) >= {"Vento mensal", "Rosa dos ventos (%)", "Janelas pulverização (%)"}


def test_solo_so_granulometria_e_camada_ponderada():
    from motor.sat import solo
    linhas = []
    for prop, v in (("clay", 40), ("sand", 40), ("silt", 20), ("soc", 99)):
        for k, p in enumerate(["0-5cm", "5-15cm", "15-30cm", "30-60cm"]):
            val = v + (10 * k if prop == "clay" else 0)
            linhas.append({"propriedade": prop, "profundidade": p, "media": val, "q05": val - 5, "q95": val + 5,
                           "ponto": 0})
    S = solo.resumir(pd.DataFrame(linhas), [(0, 0)])
    assert set(S.tabela["propriedade"]) == {"clay", "sand", "silt"}
    # argila 0–30 ponderada: (40·5 + 50·10 + 60·15) / 30
    assert S.camada["clay"][0] == pytest.approx((40 * 5 + 50 * 10 + 60 * 15) / 30)
    assert S.textura.startswith("argilosa")


def test_termico_superficie_suave():
    A = _area(600)
    g = grade_da_area(A, 30.0, buffer_m=90)
    X, Y = g.malha()
    quente = np.exp(-(((X - 500200) ** 2 + (Y - 7500300) ** 2) / (2 * 100 ** 2)))
    cenas = [termico.CenaTermica(date(2024, 1, 1) + timedelta(days=16 * k), 30 + 2 * quente + k * 0.1, 1.0)
             for k in range(5)]
    T = termico.analisar(g, cenas)
    fino, ext, res = termico.superficie(T)
    assert fino.shape == (g.altura * 3, g.largura * 3) and np.isfinite(fino).all()
    rs = termico.resumo_superficie(T, fino, ext, res, A)
    assert 0.5 < rs["amplitude"] < 2.5 and rs["quente_ha"] > 0
