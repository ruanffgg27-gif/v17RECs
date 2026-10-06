"""Testes da versão 12: semeadura, book comparativo, cores das prescrições e nomes curtos de shapefile."""
import copy
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from motor import Ajustes, Parametros
from motor.book import COR_PRESC, cores_doses
from motor.geo import ler_kml, montar_projeto
from motor.prescricao import (calcular_doses, codigo_talhao, gerar_todas_zonas, nome_shapefile, nomes_curtos,
                              superficies, zip_prescricoes)
from tests.test_mapas import _kml_perimetro, _kml_pontos, _laudo


def test_cores_sempre_do_verde_escuro():
    assert list(cores_doses([500, 300]).values()) == COR_PRESC[:2]                  # verde escuro, verde claro
    assert list(cores_doses([900, 300, 600]).values()) == COR_PRESC[:3]             # + amarelo
    c = cores_doses([100, 200, 300])
    assert c[100.0] == "#008000" and c[300.0] == "#FFFF00"
    assert len(set(cores_doses(range(8)).values())) == 8


def test_nomes_curtos_de_shapefile():
    assert codigo_talhao("Talhão 4") == "T4"
    assert codigo_talhao("TH 1") == "TH1"
    assert codigo_talhao("Perimetro_TH_12") == "TH12"
    assert codigo_talhao("Bloco A") == "BA"
    assert codigo_talhao("Gleba Norte 3") == "GN3"
    cods = nomes_curtos(["TH 1", "TH-1", "Talhão 1"])
    assert len(set(cods.values())) == 3
    for prod in ("Calcário", "Gesso", "S elementar", "P2O5", "Produto 11-52-00", "KCl 1ª aplicação", "Sementes"):
        n = nome_shapefile(prod, "TH123")
        assert len(n) <= 9 and n.replace("_", "").isalnum(), n


def _projeto(n=20):
    L = _laudo(n, com_2040=False)
    P = montar_projeto(L.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                 ler_kml(_kml_pontos(n), "Pontos_TH_1.kml")])
    return L, P


def test_zip_com_nomes_curtos_e_leia_me():
    L, P = _projeto()
    M = superficies(P, 20)
    calcular_doses(M, Parametros(), {}, Ajustes())
    gerar_todas_zonas(M, Parametros(), 6)
    with zipfile.ZipFile(io.BytesIO(zip_prescricoes(M, P.epsg))) as zf:
        nomes = zf.namelist()
        shps = [n for n in nomes if n.endswith(".shp")]
        assert "LEIA-ME.txt" in nomes and "Rx/" in zf.read("LEIA-ME.txt").decode()
        assert all(len(n.split("/")[-1][:-4]) <= 9 and " " not in n.split("/")[-1] for n in shps)


def test_indice_e_taxa_de_sementes():
    from motor import sementes as sm
    mx = {"p": 40.0, "mo": 30.0, "argila": 600.0, "sb": 80.0}
    # amostra com tudo no máximo → 100%; sem argila → SB com peso 75
    assert sm.indice(40, 30, 600, 80, mx) == pytest.approx(100)
    assert sm.indice(20, 15, np.nan, 40, mx) == pytest.approx(50)
    assert sm.indice(20, 30, 300, 80, mx) == pytest.approx((50 * 5 + 100 * 20 + 50 * 40 + 100 * 35) / 100)
    ifs = np.array([60.0, 70, 80, 90])
    (t,), k = sm.taxa_inversa([ifs], [np.ones(4, bool)], 13.0, 20.0)
    assert t.mean() == pytest.approx(13.0, abs=1e-6)
    assert np.all(np.diff(t) < 0)                                  # mais fértil → menos sementes
    assert t.max() <= 13 * 1.2 + 1e-9 and t.min() >= 13 * 0.8 - 1e-9


def test_mapa_de_sementes_mantem_a_media():
    from motor import sementes as sm
    L, P = _projeto(25)
    M = superficies(P, 20, sm.ATRIB_SEMENTES)
    cfg = sm.ConfigSementes(media=14.0, classes=8)
    res = sm.gerar(M, L.dados, cfg)
    z = M[0].zonas["Sementes"]
    assert z.dose_media == pytest.approx(14.0, abs=0.06)
    assert len({d for d, _ in z.poligonos}) <= 8
    assert z.area_ha == pytest.approx(M[0].talhao.area_ha, rel=1e-3)
    assert "Índice de fertilidade (%)" in res.amostras
    zb = sm.zip_sementes(M, P.epsg, sm.ConfigSementes(media=14.0, unidade_shape="sementes/ha"))
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        assert any(n.endswith("SEM_T1.shp") for n in zf.namelist())
    assert cfg.por_ha(13) == pytest.approx(260000)
    assert sm.ConfigSementes(medias_talhao={"Talhão 1": 11.0}).media_de("1") == 11.0


def test_comparacao_na_mesma_grade_e_volumes():
    from motor import comparacao as cp
    L, P = _projeto(25)
    LA = copy.deepcopy(L)
    LA.dados["p"] = LA.dados["p"] * 0.5
    LA.dados["ca"] = LA.dados["ca"] * 0.7
    PA = montar_projeto(LA.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                   ler_kml(_kml_pontos(25), "Pontos_TH_1.kml")])
    M = superficies(P, 20)
    aj = Ajustes()
    calcular_doses(M, Parametros(), {}, aj)
    gerar_todas_zonas(M, Parametros(), 6)
    c = cp.superficies_na_grade(M, PA)
    c.rotulos = ("2024", "2026")
    assert len(c.pares) == 1
    a, b = c.pares[0]
    assert a.grade is b.grade and a.atributos["p"].shape == b.atributos["p"].shape
    assert np.nanmean(a.atributos["p"]) == pytest.approx(0.5 * np.nanmean(b.atributos["p"]), rel=0.05)
    cp.volumes_anterior(c, Parametros(), {}, aj)
    v = c.volumes.set_index("Produto")
    assert v.loc["Calcário", "Total 2024 (t)"] > v.loc["Calcário", "Total 2026 (t)"]   # solo mais pobre antes
    ev = cp.tabela_evolucao(c, ["p", "ca"]).set_index("chave")
    assert ev.loc["p", "Δ"] > 0 and ev.loc["p", "melhora"] > 0
    from motor.book_comparacao import arranjo, transicoes
    assert arranjo((0, 0, 1000, 3000))[0] == "lado" and arranjo((0, 0, 3000, 600))[0] == "empilhado"
    from motor.book import escala_classes
    lim, cores, _ = escala_classes("p")
    T = transicoes(c.pares, "p", lim, len(cores))
    assert T.sum() == pytest.approx(b.grade.mascara.sum() * b.grade.area_pixel_ha)
    assert cp.ano_do_nome("Laudo 8438-2026 - ATRIA") == "2026"


def test_book_comparativo_e_de_sementes_montam():
    from motor import book_comparacao, book_sementes, comparacao as cp, sementes as sm
    from motor.book import DadosBook
    L, P = _projeto(25)
    LA = copy.deepcopy(L)
    LA.dados["ph"] = LA.dados["ph"] - 0.3
    PA = montar_projeto(LA.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                   ler_kml(_kml_pontos(25), "Pontos_TH_1.kml")])
    M = superficies(P, 20)
    aj = Ajustes()
    calcular_doses(M, Parametros(), {}, aj)
    bl = gerar_todas_zonas(M, Parametros(), 6)
    c = cp.superficies_na_grade(M, PA)
    c.rotulos = ("2024", "2026")
    cp.volumes_anterior(c, Parametros(), {}, aj)
    d = DadosBook(marca="Atria", usar_internet=False, incluir_ndvi=False,
                  titulo_capa=("COMPARATIVO DA FERTILIDADE DO SOLO", "2024 × 2026"))
    pdf, _ = book_comparacao.montar(P, M, d, Parametros(), aj, c, L.dados, blocos=bl)
    assert pdf[:4] == b"%PDF" and len(pdf) > 50_000
    M2 = superficies(P, 20, sm.ATRIB_SEMENTES)
    cfg = sm.ConfigSementes()
    res = sm.gerar(M2, L.dados, cfg)
    pdf2 = book_sementes.montar(P, M2, DadosBook(marca="Atria", usar_internet=False), cfg, res)
    assert pdf2[:4] == b"%PDF"
