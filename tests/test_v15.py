"""Testes da versão 15: interpolação fiel às amostras (mapas) separada da das doses, interpretação invertida de
H+Al, Al e m%, contornos vetoriais e o módulo de amostragem (grade e zonas de manejo)."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from motor import Ajustes, Parametros
from motor import amostragem as am
from motor import interpretacao as it
from motor.geo import ler_kml, montar_projeto
from motor.krigagem import interpolar
from motor.prescricao import BASE_DOSE, calcular_doses, superficies
from tests.test_mapas import _kml_perimetro, _kml_pontos, _laudo


def _campo(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 2000, n), rng.uniform(0, 2000, n)
    return x, y, rng


# ------------------------------------------------------------------ interpolação
def test_modo_fiel_segue_as_amostras_e_o_estatistico_alisa():
    """Sem estrutura na escala da área (ruído ponto a ponto), o modo estatístico tende à média; o fiel mostra o
    que cada ponto mediu."""
    x, y, rng = _campo()
    z = rng.normal(30, 8, len(x))
    gx, gy = np.meshgrid(np.linspace(0, 2000, 60), np.linspace(0, 2000, 60))
    ve, _ = interpolar(x, y, z, gx, gy)
    vf, aj = interpolar(x, y, z, gx, gy, modo="fiel")
    assert aj.modo == "fiel" and "fiel às amostras" in aj.descricao()
    assert np.ptp(vf) > 1.5 * np.ptp(ve)
    perto, _ = interpolar(x, y, z, x + 1.0, y + 1.0, modo="fiel", limitar_extremos=False)
    assert np.sqrt(np.mean((perto - z) ** 2)) < 0.35 * z.std()        # passa perto do valor medido
    assert vf.min() >= z.min() - 1e-9 and vf.max() <= z.max() + 1e-9   # não extrapola


def test_modo_fiel_nao_cria_degrau_em_cima_do_ponto():
    x, y, rng = _campo(40, 1)
    z = rng.normal(10, 2, len(x))
    d = np.array([0.0, 0.5, 1.0, 2.0])
    v, _ = interpolar(x, y, z, x[0] + d, np.full(4, y[0]), modo="fiel")
    assert np.abs(np.diff(v)).max() < 0.05 * z.std()                   # superfície contínua (pepita filtrada)


def test_mapas_fieis_e_doses_estatisticas():
    L = _laudo(25, com_2040=False)
    P = montar_projeto(L.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                 ler_kml(_kml_pontos(25), "Pontos_TH_1.kml")])
    antigo = superficies(P, 20)
    novo = superficies(P, 20, modo="fiel", modo_doses="estatistico")
    m = novo[0]
    assert set(m.base) == {k for k in BASE_DOSE if k in m.atributos} and m.ajustes["p"].modo == "fiel"
    calcular_doses(antigo, Parametros(), {}, Ajustes())
    calcular_doses(novo, Parametros(), {}, Ajustes())
    for prod in antigo[0].doses:                                       # prescrições idênticas às da v14
        assert np.allclose(antigo[0].doses[prod], m.doses[prod], equal_nan=True)
    assert not np.allclose(antigo[0].atributos["p"], m.atributos["p"], equal_nan=True)
    iguais = superficies(P, 20, modo="fiel")
    assert not iguais[0].base


# ------------------------------------------------------------------ interpretação
def test_acidez_baixa_e_excelente():
    assert it.classificar("al", 0.0) == "Excelente" and it.classificar("m", 0.3) == "Excelente"
    assert it.classificar("hal", 15) == "Excelente" and it.classificar("hal", 25) == "Excelente"
    assert it.classificar("al", 3.2) == "Crítico"
    assert it.classificar("v", 78) == "Muito alto"                     # os demais atributos não mudam
    assert 6.0 <= it.pos_condicao("al", 0.0) <= 7.0 and it.pos_condicao("al", 0.0) > it.pos_condicao("al", 0.5)


# ------------------------------------------------------------------ amostragem
def _talhao(pol, nome="TH 01"):
    return am.TalhaoAmostragem(nome, pol, 31982)


@pytest.mark.parametrize("formato", list(am.FORMATOS))
@pytest.mark.parametrize("posicao", list(am.POSICOES))
def test_grade_densidade_e_pontos_dentro(formato, posicao):
    pol = Polygon([(0, 0), (1500, 100), (1600, 900), (700, 1100), (-100, 800)]).difference(box(600, 400, 800, 600))
    t = _talhao(pol)
    p = am.plano_grade([t], 4.0, formato, posicao=posicao)
    n = len(p.pontos)
    assert abs(t.area_ha / n - 4.0) < 1.0
    assert all(pol.contains(Point(x, y)) for x, y in zip(p.pontos["x"], p.pontos["y"]))
    assert p.pontos["Ponto"].tolist() == list(range(1, n + 1))
    xy = p.pontos[["x", "y"]].to_numpy()
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) + np.eye(n) * 1e9
    assert d.min() > (60 if posicao == "centro" else 5)                # sem pontos colados
    passo = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(1))
    assert np.median(passo) < 1.6 * 200                                # caminhamento: vizinhos em sequência


def test_grade_alinha_ao_talhao_e_numera_por_talhao():
    ret = Polygon([(0, 0), (2000, 0), (2000, 400), (0, 400)])
    from shapely import affinity
    inclinado = affinity.rotate(ret, 30, origin=(0, 0))
    assert abs(am.orientacao(inclinado) - 30) < 0.5
    p = am.plano_grade([_talhao(inclinado, "TH 01"), _talhao(affinity.translate(ret, 0, 3000), "TH 02")], 4.0)
    assert len(p.pontos) == 40                                         # 2 × (80 ha ÷ 4 ha): células inteiras
    assert p.pontos.groupby("Talhão")["Ponto"].min().tolist() == [1, 1]
    corrido = am.plano_grade(p.talhoes, 4.0, numerar_por_talhao=False)
    assert corrido.pontos["Ponto"].tolist() == list(range(1, 41))


def test_zonas_de_manejo_e_exportacao():
    pol = box(0, 0, 1000, 800)
    xs, ys = np.arange(5, 1000, 10.0), np.arange(795, 0, -10.0)
    X, Y = np.meshgrid(xs, ys)
    vigor = (X - 500) / 250 + 0.02 * np.sin(Y / 40)                   # gradiente leste-oeste
    vigor[10:14, 10:14] = 5.0                                          # mancha de 0,16 ha: tem de sumir
    M = np.ones(X.shape, bool)
    t = _talhao(pol)
    p = am.plano_zonas([t], [(vigor, M, xs, ys)], n_zonas=3, ha_por_ponto=5.0, min_por_zona=2, area_min_ha=1.0)
    areas = {z: g.area / 1e4 for _, z, g in p.zonas}
    assert list(areas) == am.NOMES_ZONA[3] and all(abs(a - 80 / 3) < 4 for a in areas.values())
    assert abs(sum(areas.values()) - 80) < 0.05                        # partição exata do talhão
    cont = p.pontos.groupby("Zona").size()
    assert (cont >= 2).all() and len(p.pontos) == 15
    for _, r in p.pontos.iterrows():
        g = next(g for _, z, g in p.zonas if z == r["Zona"])
        assert g.buffer(15).contains(Point(r["x"], r["y"]))
    assert p.pontos.loc[p.pontos["Zona"] == "Baixo vigor", "x"].mean() < 400
    with zipfile.ZipFile(io.BytesIO(am.zip_amostragem(p))) as z:
        nomes = z.namelist()
        assert {"KML por talhao/Pontos_TH_1.kml", "Pontos_todos.gpx", "Zonas_de_manejo.kml",
                "Coordenadas.xlsx"} <= set(nomes)
        k = ler_kml(z.read("KML por talhao/Pontos_TH_1.kml"), "Pontos_TH_1.kml")
        assert k.pontos["nome"].tolist() == [str(i) for i in range(1, 16)]   # nome do ponto = número
        assert len(pd.read_excel(io.BytesIO(z.read("Coordenadas.xlsx")))) == 15


def test_pdf_de_amostragem():
    from motor import book_amostragem
    p = am.plano_grade([_talhao(box(0, 0, 1200, 900))], 1.0)
    pdf = book_amostragem.montar(p, "Atria", "Teste")
    assert pdf[:4] == b"%PDF" and len(p.pontos) == 108
