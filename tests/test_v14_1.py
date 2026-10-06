"""Testes da versão 14.1: paleta própria por índice, rede de segurança contra texto fora da moldura e PDF simples
do mapa de semeadura (capa + um mapa por talhão + resumo)."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def test_cada_indice_tem_paleta_propria():
    from motor.sat import lavoura
    from motor.sat.relatorio import CMAP_INDICE, LEGENDA_INDICE
    assert set(CMAP_INDICE) == set(lavoura.INDICES) == set(LEGENDA_INDICE)
    pontas = {k: tuple(np.round(c(1.0)[:3], 2)) + tuple(np.round(c(0.0)[:3], 2)) for k, c in CMAP_INDICE.items()}
    assert len(set(pontas.values())) == len(pontas)                     # nenhuma paleta repetida


def test_texto_nao_passa_da_linha_direita():
    from motor.sat.relatorio import TEXTO_DIR, _encaixar
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.5, "frase bem comprida que não cabe de jeito nenhum " * 4, fontsize=9)
    fig.text(0.8, 0.4, "1.234,5 mm acumulados", fontsize=8)
    curto = fig.text(0.1, 0.3, "cabe", fontsize=8)
    fig.canvas.draw()
    _encaixar(fig)
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    for t in fig.texts:
        assert inv.transform(t.get_window_extent(r))[1][0] <= TEXTO_DIR + 0.003
    assert curto.get_text() == "cabe" and curto.get_fontsize() == 8
    plt.close(fig)


def test_pdf_simples_de_semeadura():
    from motor import book_sementes, sementes as sm
    from motor.book import DadosBook
    from motor.geo import ler_kml, montar_projeto
    from motor.prescricao import superficies
    from tests.test_mapas import _kml_perimetro, _kml_pontos, _laudo
    L = _laudo(20, com_2040=False)
    P = montar_projeto(L.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                 ler_kml(_kml_pontos(20), "Pontos_TH_1.kml")])
    M = superficies(P, 20, sm.ATRIB_SEMENTES)
    cfg = sm.ConfigSementes()
    res = sm.gerar(M, L.dados, cfg)
    dados = DadosBook(marca="Atria", usar_internet=False)
    simples = book_sementes.montar(P, M, dados, cfg, res, cultura="Soja")
    completo = book_sementes.montar(P, M, dados, cfg, res, cultura="Soja", com_indice=True)
    paginas = lambda b: len(re.findall(rb"/Type\s*/Page\b", b))
    assert paginas(simples) == 3 and paginas(completo) == 4            # capa + talhão + resumo (+ índice)
