"""Testes da versão 14: prescrições agrupadas por produto, nomes de talhões que cabem na página e mapas
históricos dos índices com legenda ajustada ao talhão."""
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np

from motor import Ajustes
from motor.book import lista_talhoes, ordem_prescricoes


def _bloco(nome, prods):
    return SimpleNamespace(nome=nome, zonas={p: object() for p in prods})


def test_prescricoes_agrupadas_por_produto():
    prods = ["KCl", "Calcário", "P2O5", "Produto 11-52-00", "Gesso", "KCl 1ª aplicação", "KCl 2ª aplicação"]
    blocos = [_bloco("Talhão 1", prods), _bloco("Talhão 2", prods), _bloco("Talhão 3", ["Calcário", "KCl"])]
    saida = ordem_prescricoes(blocos, Ajustes(kcl_parcelado=True, formula_p="11-52-00"), "sub p", {})
    ordem = [(b.nome, p) for _, b, p, _ in saida]
    produtos = [p for _, p in ordem]
    assert produtos[:3] == ["Calcário"] * 3                              # calagem de todos primeiro
    assert produtos[3:5] == ["Gesso"] * 2 and produtos[5:7] == ["Produto 11-52-00"] * 2
    assert "P2O5" not in produtos                                        # com formulação, só o produto
    assert produtos.count("KCl") == 1 and ordem[-1][1] == "KCl 2ª aplicação"   # talhão 3 não tem parcelas
    rotulos = [r for r, *_ in saida if r]
    assert rotulos[:3] == ["Calagem", "Gessagem", "Fósforo – 11-52-00"]


def test_lista_de_talhoes_cabe_no_titulo():
    def mapas(nomes):
        return [SimpleNamespace(talhao=SimpleNamespace(nome=n)) for n in nomes]
    assert lista_talhoes(mapas(["Talhão 1", "Talhão 2"])) == "TH 01, TH 02"
    longo = lista_talhoes(mapas([f"Gleba Santo Antônio da Serra {i}" for i in range(1, 5)]))
    assert longo.endswith("(4 talhões)") and len(longo) < 80


def test_indices_historicos_e_legenda_ajustada():
    from motor.sat import lavoura
    from motor.sat.relatorio import faixa_do_talhao
    from tests.test_sat import _area
    from motor.sat.base import grade_da_area
    A = _area()
    g = grade_da_area(A, 10.0)
    X, _ = g.malha()
    grad = (X - X.min()) / (X.max() - X.min())                       # gradiente leste-oeste persistente
    cenas = []
    for k in range(36):
        d = date(2023, 7, 15) + timedelta(days=30 * k)
        fase = 0.3 + 0.55 * max(0.0, np.sin(2 * np.pi * (d.timetuple().tm_yday - 280) / 365))
        ndvi = fase * (0.95 + 0.05 * grad)
        cenas.append(lavoura.Cena(d, {"ndvi": ndvi, "ndre": ndvi * 0.6, "ndmi": ndvi * 0.5, "msavi2": ndvi * 0.9}))
    L = lavoura.analisar(g, cenas)
    assert set(L.hist) == set(lavoura.INDICES) and len(L.hist_anos) >= 2
    v = L.hist["ndvi"][g.mascara]
    lo, hi = faixa_do_talhao(v)
    assert 0.7 < lo < hi < 0.9 and hi - lo < 0.1                      # legenda dentro da faixa real do talhão
    assert np.corrcoef(L.hist["ndvi"][g.mascara], grad[g.mascara])[0, 1] > 0.95
    assert faixa_do_talhao(np.full(10, 0.8))[1] - faixa_do_talhao(np.full(10, 0.8))[0] >= 0.02
