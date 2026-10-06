"""Testes da versão 16: deslocamento/giro da malha, edição manual de pontos e dados do editor de mapa."""
import numpy as np
from shapely.geometry import Point, box

from motor import amostragem as am


def _t():
    return am.TalhaoAmostragem("TH 01", box(500000, 7500000, 501200, 7500900), 31982)


def test_malha_deslocada_e_girada_continua_dentro_e_na_densidade():
    t = _t()
    base = am.plano_grade([t], 4.0)
    for kw in ({"deslocamento": (60.0, -40.0)}, {"giro": 25.0}, {"deslocamento": (-90.0, 30.0), "giro": -15.0}):
        p = am.plano_grade([t], 4.0, **kw)
        assert all(t.poligono.contains(Point(x, y)) for x, y in zip(p.pontos["x"], p.pontos["y"]))
        assert abs(t.area_ha / len(p.pontos) - 4.0) < 1.2 and "malha deslocada" in p.descricao
        assert not np.allclose(p.pontos[["x", "y"]].to_numpy()[:5], base.pontos[["x", "y"]].to_numpy()[:5])
    # células inteiras andam exatamente o deslocamento pedido
    q = am.plano_grade([t], 4.0, deslocamento=(60.0, -40.0))
    d = q.pontos[["x", "y"]].to_numpy()[:, None, :] - base.pontos[["x", "y"]].to_numpy()[None, :, :]
    assert (np.abs(d - np.array([60.0, -40.0])).sum(-1) < 1e-6).any()
    assert am.plano_grade([t], 4.0, deslocamento=(0.0, 0.0)).pontos.equals(base.pontos)


def test_edicao_manual_valida_e_renumera():
    p = am.plano_grade([_t()], 4.0)
    n = len(p.pontos)
    r = p.pontos.iloc[3]
    recusas = am.editar_pontos(p, mover={3: (r["Longitude"] + 0.0003, r["Latitude"]),
                                         0: (r["Longitude"] + 1.0, r["Latitude"])},
                               remover=[5], adicionar=[(r["Longitude"], r["Latitude"] + 0.0005), (0.0, 0.0)])
    assert len(recusas) == 2 and "fora do talhão" in recusas[0]
    assert len(p.pontos) == n and p.pontos["Ponto"].tolist() == list(range(1, n + 1))
    assert abs(p.pontos.iloc[3]["Longitude"] - (r["Longitude"] + 0.0003)) < 1e-9
    assert _t().poligono.contains(Point(p.pontos.iloc[3]["x"], p.pontos.iloc[3]["y"]))   # x/y acompanham lon/lat
    assert len(p.celulas) == n and "à mão" in p.descricao
    # o KML sai com a edição
    import io
    import zipfile
    from motor.geo import ler_kml
    with zipfile.ZipFile(io.BytesIO(am.zip_amostragem(p))) as z:
        k = ler_kml(z.read("KML por talhao/Pontos_TH_1.kml"), "Pontos_TH_1.kml")
    assert len(k.pontos) == n and abs(k.pontos.iloc[3]["lon"] - (r["Longitude"] + 0.0003)) < 1e-6


def test_dados_do_editor_e_pdf_apos_edicao():
    from motor import book_amostragem
    p = am.plano_grade([_t()], 4.0)
    d = am.dados_do_mapa(p)
    assert len(d["pontos"]) == len(p.pontos) and len(d["talhoes"]) == 1 and d["celulas"]
    lat, lon = d["talhoes"][0]["poligonos"][0][0][0]
    assert -90 < lat < 0 and -180 < lon < 0                              # [lat, lon], como o Leaflet espera
    am.editar_pontos(p, remover=[0, 1])
    assert book_amostragem.montar(p, "Atria", "Teste")[:4] == b"%PDF"


def test_componente_do_editor_tem_o_leaflet_embutido():
    from motor import editor_mapa
    assert (editor_mapa.ASSETS / "leaflet.js").stat().st_size > 100_000
    assert "setTriggerValue" in editor_mapa.JS and 'data-acao="aplicar"' in editor_mapa.HTML
