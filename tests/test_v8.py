"""Testes da versão 8: argila estimada pela CTC, dados ausentes e blocos de aplicação."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import shapefile

from motor import Ajustes, Parametros, ler_laudo, recomendar
from motor.geo import ler_kml, montar_projeto
from motor.prescricao import calcular_doses, gerar_todas_zonas, superficies, zip_prescricoes
from motor.textura import COEF_REGIONAL, completar_argila
from tests.test_mapas import _kml_perimetro, _kml_pontos, _laudo


def _xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def test_laudo_sem_granulometria_e_micros():
    df = pd.DataFrame({"Talhão": [1, 1, 2], "Profundidade": ["0-20"] * 3, "Ca": [30, 20, 40], "Mg": [10, 8, 12],
                       "H+Al": [30, 25, 40], "Al": [0, 1, 0], "K": [2, 1.5, 3], "P": [20, 10, 15], "S": [8, 6, 9],
                       "CTC": [72, 55, 95]})
    L = ler_laudo(_xlsx(df), "x.xlsx")
    assert {"argila", "areia", "silte", "b", "zn"} <= set(L.ausentes)
    a, b = COEF_REGIONAL
    assert L.dados["argila"].tolist() == pytest.approx([round(a + b * c) for c in (72, 55, 95)])
    assert L.dados["argila_estimada"].all()
    assert any("estimada pela CTC" in m for m in L.avisos_dados)
    assert any("Micronutrientes" in m for m in L.avisos_dados)
    r = recomendar(L)
    assert r["Gesso (kg/ha)"].notna().all()
    assert (r["Argila - origem"] == "estimada (CTC)").all()


def test_argila_parcial_usa_amostras_medidas():
    rng = np.random.default_rng(1)
    ctc = rng.uniform(40, 120, 30)
    arg = 50 + 5 * ctc + rng.normal(0, 15, 30)
    d = pd.DataFrame({"ctc": ctc, "argila": arg, "ca": 20.0, "mg": 8.0, "hal": 30.0, "k": 2.0})
    d.loc[::3, "argila"] = np.nan
    d, metodo = completar_argila(d)
    assert "regressão local" in metodo
    est = d["argila_estimada"]
    assert np.abs(d.loc[est, "argila"] - (50 + 5 * ctc[est.values])).mean() < 25


def test_blocos_de_aplicacao():
    L = _laudo(20, com_2040=False)
    kml = [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"), ler_kml(_kml_pontos(20), "Pontos_TH_1.kml")]
    P = montar_projeto(L.dados, kml)
    M = superficies(P, 20)
    calcular_doses(M, Parametros(), {}, Ajustes())
    blocos = gerar_todas_zonas(M, Parametros(), 6, {M[0].talhao.nome: "Bloco A"})
    assert len(blocos) == 1 and blocos[0].nome == M[0].talhao.nome     # bloco de um talhão só = separado
    # dois talhões (o mesmo perímetro duplicado com outro nome) num bloco
    import copy
    m2 = copy.deepcopy(M[0])
    m2.talhao.nome = "Talhão 2"
    from shapely.affinity import translate
    m2.talhao.perimetro = translate(m2.talhao.perimetro, 2000, 0)
    blocos = gerar_todas_zonas([M[0], m2], Parametros(), 6, {M[0].talhao.nome: "Bloco A", "Talhão 2": "Bloco A"})
    assert len(blocos) == 1 and not blocos[0].unico
    b = blocos[0]
    z = b.zonas["Calcário"]
    assert len(z.poligonos) == len({d for d, _ in z.poligonos}) <= 6        # um registro por dose
    assert z.area_ha == pytest.approx(M[0].zonas["Calcário"].area_ha + m2.zonas["Calcário"].area_ha, rel=1e-6)
    zb = zip_prescricoes(blocos, P.epsg, produtos=["Calcário"])
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        shps = [n for n in zf.namelist() if n.endswith(".shp")]
        assert len(shps) == 1 and shps[0].startswith("Bloco_A/")
        r = shapefile.Reader(shp=io.BytesIO(zf.read(shps[0])), dbf=io.BytesIO(zf.read(shps[0][:-4] + ".dbf")))
        assert len(r.records()) == len(z.poligonos)


def test_kcl_descontado_pelo_k2o_da_formulacao():
    from motor.regras import kcl_equivalente
    assert kcl_equivalente(200, 10) == pytest.approx(33.333, rel=1e-3)          # 20 kg K2O ÷ 0,60
    df = pd.DataFrame({"Talhão": [1, 1], "Profundidade": ["0-20"] * 2, "Ca": [30, 30], "Mg": [10, 10],
                       "H+Al": [30, 30], "Al": [0, 0], "K": [1.0, 4.0], "P": [8, 30], "S": [8, 8],
                       "Argila": [400, 400]})
    L = ler_laudo(_xlsx(df), "x.xlsx")
    sem = recomendar(L, Parametros(), {}, Ajustes(formula_p="00-30-00"))
    com = recomendar(L, Parametros(), {}, Ajustes(formula_p="00-30-10"))
    assert "KCl equivalente descontado (kg/ha)" in com and "KCl equivalente descontado (kg/ha)" not in sem
    for i in range(2):
        prod = com.loc[i, "Produto 00-30-10 (kg/ha)"]
        esperado = max(com.loc[i, "KCl pela regra (kg/ha)"] - prod * 0.10 / 0.60, 0)
        assert abs(com.loc[i, "KCl (kg/ha)"] - esperado) <= 5                      # arredondamento de 5 kg
        assert com.loc[i, "KCl (kg/ha)"] < sem.loc[i, "KCl (kg/ha)"]
    muito_k = recomendar(L, Parametros(), {}, Ajustes(formula_p="00-20-40"))
    assert (muito_k["KCl (kg/ha)"] >= 0).all()
    assert (muito_k["KCl (kg/ha)"] == 0).any() and "K suprido" in " ".join(muito_k["Observações"])
