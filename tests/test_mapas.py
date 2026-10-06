"""Testes do módulo de mapas: KML, ligação com o laudo, krigagem, zonas e shapefile."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import shapefile
from shapely.geometry import Polygon

from motor import Ajustes, Parametros, ler_laudo
from motor.geo import chave_talhao, epsg_utm_sirgas, ler_kml, montar_projeto, numero_talhao
from motor.krigagem import grade_do_poligono, interpolar
from motor.prescricao import calcular_doses, gerar_todas_zonas, superficies, zip_prescricoes

LON0, LAT0 = -53.155, -19.280


def _kml_perimetro(tal, dx=0.012, dy=0.010):
    c = f"{LON0},{LAT0},0 {LON0+dx},{LAT0},0 {LON0+dx},{LAT0+dy},0 {LON0},{LAT0+dy},0 {LON0},{LAT0},0"
    return (f'<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
            f'<ExtendedData><SchemaData><SimpleData name="Field">Th_0{tal}</SimpleData></SchemaData></ExtendedData>'
            f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{c}</coordinates></LinearRing></outerBoundaryIs>'
            f'</Polygon></Placemark></Document></kml>').encode()


def _kml_pontos(n):
    rng = np.random.default_rng(1)
    pm = "".join(f"<Placemark><name>{i+1}</name><Point><coordinates>{LON0+0.001+rng.uniform(0,0.010)},"
                 f"{LAT0+0.001+rng.uniform(0,0.008)},0</coordinates></Point></Placemark>" for i in range(n))
    return f"<kml><Document><Folder>{pm}</Folder></Document></kml>".encode()


def _laudo(n=20, com_2040=True):
    rng = np.random.default_rng(2)
    cab = ["N Lab", "Proprietário", "Propriedade", "Identificação", "Talhão", "Profundidade", "pH CaCl2",
           "P resina", "M.O.", "K", "Ca", "Mg", "H+Al", "Al", "S-SO4", "Argila", "Areia", "Silte"]
    linhas = []
    for i in range(n):
        linhas.append([100 + i, "P", "F", f"AMOSTRA {i+1:02d}", 1, "0-20", round(rng.uniform(4.8, 5.8), 1),
                       rng.uniform(8, 40), 20, rng.uniform(1, 4), rng.uniform(15, 50), rng.uniform(5, 20),
                       rng.uniform(15, 40), 0, rng.uniform(4, 15), 200 + 15 * i, 700 - 15 * i, 100])
        if com_2040 and i == n - 1:
            linhas.append([999, "P", "F", f"AMOSTRA {i+1:02d}", 1, "20-40", 4.9, 5, 12, 1, 12, 5, 30, 2, 20,
                           400, 500, 100])
    buf = io.BytesIO()
    pd.DataFrame([cab] + linhas).to_excel(buf, header=False, index=False)
    buf.seek(0)
    return ler_laudo(buf, "l.xlsx")


def test_numero_talhao():
    assert numero_talhao("Pontos_TH_1_Campo_Verde.kml") == "1"
    assert numero_talhao("Th_02 Campo verde.kml") == "2"
    assert numero_talhao("Talhão 3.kmz") == "3"
    assert chave_talhao(1.0) == "1" and chave_talhao("TH 01") == "1"


def test_epsg():
    assert epsg_utm_sirgas(-53.15, -19.28) == 31982
    assert epsg_utm_sirgas(-55.0, -22.0) == 31981


def test_kml_e_kmz():
    k = ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml")
    assert k.tipo == "perimetro" and k.atributos["Field"] == "Th_01"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("doc.kml", _kml_pontos(5))
    kz = ler_kml(buf.getvalue(), "Pontos_TH_1.kmz")
    assert kz.tipo == "pontos" and len(kz.pontos) == 5


def test_ligacao_ordem_e_2040():
    L = _laudo(20)
    P = montar_projeto(L.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                 ler_kml(_kml_pontos(20), "Pontos_TH_1.kml")])
    t = P.talhoes[0]
    assert len(t.amostras) == 20 and list(t.amostras["ponto"][:3]) == [1, 2, 3]
    assert len(t.sub) == 1 and t.sub["ponto"].iloc[0] == 20          # 20-40 herda o ponto 20
    assert 100 < t.area_ha < 160          # ~1,26 km × 1,1 km


def test_krigagem_recupera_tendencia():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 1000, 40), rng.uniform(0, 1000, 40)
    z = 0.05 * x + rng.normal(0, 2, 40)                              # tendência forte → dependência espacial
    v, aj = interpolar(x, y, z, np.array([100, 900]), np.array([500, 500]))
    assert v[1] > v[0] + 20
    assert aj.metodo in ("krigagem", "IDW")


def test_zonas_particionam_o_talhao_e_shapefile():
    L = _laudo(20, com_2040=False)
    P = montar_projeto(L.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                 ler_kml(_kml_pontos(20), "Pontos_TH_1.kml")])
    M = superficies(P, 20)
    calcular_doses(M, Parametros(), {}, Ajustes(kcl_parcelado=True, kcl_pct_1=60))
    gerar_todas_zonas(M, Parametros(), 6)
    m = M[0]
    for prod, z in m.zonas.items():
        assert z.area_ha == pytest.approx(m.talhao.area_ha, rel=1e-3), prod
        assert len(z.niveis) <= 6
    k, k1, k2 = (m.doses[c] for c in ("KCl", "KCl 1ª aplicação", "KCl 2ª aplicação"))
    assert np.allclose(np.nan_to_num(k1 + k2), np.nan_to_num(k))
    zbytes = zip_prescricoes(M, P.epsg)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        shp = [n for n in zf.namelist() if n.endswith(".shp") and "/CAL_" in n][0]
        base = shp[:-4]
        r = shapefile.Reader(shp=io.BytesIO(zf.read(shp)), shx=io.BytesIO(zf.read(base + ".shx")),
                             dbf=io.BytesIO(zf.read(base + ".dbf")))
        assert r.fields[1][0] == "Taxa_Dest_"
        assert -54 < r.bbox[0] < -53 and -20 < r.bbox[1] < -19                  # WGS84 lon/lat
        assert (base + ".prj") in zf.namelist()
