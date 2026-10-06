"""Testes da versão 13: casos levantados na revisão técnica (ligação amostra-ponto, volumes, dados ausentes,
profundidade, fusão de polígonos, dose zero, parcelas de KCl, conferência, método analítico, krigagem sem
estrutura, paginação STAC e consistência planilha × mapa)."""
import copy
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from motor import AjusteProduto, Ajustes, Parametros, ler_laudo, recomendar
from motor.geo import ler_kml, montar_projeto
from motor.krigagem import Grade, interpolar
from motor.prescricao import (MapaTalhao, _fundir_pequenos, _vetor, calcular_doses, conferir, gerar_todas_zonas,
                              superficies, zip_prescricoes)
from tests.test_mapas import _kml_perimetro, _kml_pontos, _laudo


def _xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _kmls(n=20):
    return [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"), ler_kml(_kml_pontos(n), "Pontos_TH_1.kml")]


# ------------------------------------------------------------------ ligação amostra × ponto
def test_ligacao_pelo_numero_independe_da_ordem_do_laudo():
    L = _laudo(20, com_2040=False)
    inv = L.dados.iloc[::-1].reset_index(drop=True)
    P = montar_projeto(inv, _kmls())
    a = P.talhoes[0].amostras
    assert (a["identificacao"].str.extract(r"(\d+)")[0].astype(int) == a["ponto"]).all()
    assert not P.pendencias
    assert set(P.conferencia["Ligação"]) == {"número"}


def test_divergencias_viram_pendencia():
    L = _laudo(20, com_2040=False)
    menos = L.dados[L.dados["identificacao"] != "AMOSTRA 05"]
    P = montar_projeto(menos, _kmls())
    assert any("sem amostra" in p for p in P.pendencias)
    assert "ponto sem amostra" in set(P.conferencia["Situação"])
    # numeração corrida do laudo (21–40) com pontos 1–20: mesmo deslocamento → aceito sem pendência
    d = L.dados.copy()
    d["identificacao"] = [f"AMOSTRA {i + 21}" for i in range(len(d))]
    P2 = montar_projeto(d, _kmls())
    assert not P2.pendencias and P2.conferencia["Ligação"].iloc[0].startswith("número (")


# ------------------------------------------------------------------ dados ausentes e dose zero
def test_sem_enxofre_so_o_gesso_fica_de_fora():
    L = _laudo(20, com_2040=False)
    P = montar_projeto(L.dados, _kmls())
    M = superficies(P, 20)
    for m in M:
        m.atributos.pop("s", None)
    avisos = []
    calcular_doses(M, Parametros(), {}, Ajustes(), avisos)
    assert "Gesso" not in M[0].doses and {"Calcário", "P2O5", "KCl"} <= set(M[0].doses)
    assert any("Gesso" in a for a in avisos)
    gerar_todas_zonas(M, Parametros(), 6)


def test_dose_zero_nao_vira_ausencia():
    from motor.regras import Resultado
    out = _vetor(lambda v: Resultado(0.0 if v < 1 else None), np.array([0.5, 2.0, np.nan]))
    assert out[0] == 0.0 and np.isnan(out[1]) and np.isnan(out[2])


# ------------------------------------------------------------------ profundidade e método
def test_profundidade_validada():
    base = {"Talhão": [1] * 6, "Ca": [30] * 6, "Mg": [10] * 6, "H+Al": [30] * 6, "Al": [0] * 6, "K": [2] * 6,
            "P": [10] * 6, "S": [8] * 6, "Argila": [400] * 6,
            "Identificação": [f"A{i}" for i in range(6)],
            "Profundidade": ["0-20", "0,20-0,40", "0-40", "10-20", "0,00-0,20 m", "20-40"]}
    L = ler_laudo(_xlsx(pd.DataFrame(base)), "x.xlsx")
    assert len(L.dados) == 4 and len(L.fora_camada) == 2                  # 0-40 e 10-20 ficam de fora
    assert L.dados["subsuperficial"].tolist() == [False, True, False, True]
    assert any("FORA" in a for a in L.avisos)


def test_p_mehlich_reconhecido_e_bloqueado():
    df = pd.DataFrame({"Talhão": [1, 1], "Ca": [30, 30], "Mg": [10, 10], "H+Al": [30, 30], "Al": [0, 0], "K": [2, 2],
                       "P Mehlich-1": [10, 20], "S": [8, 8], "Argila": [400, 400], "pH H2O": [5.5, 5.8]})
    L = ler_laudo(_xlsx(df), "x.xlsx")
    assert L.metodos == {"p": "Mehlich-1", "ph": "água"}
    assert L.dados["p"].tolist() == [10, 20]
    r = recomendar(L, ajustes=Ajustes(bloquear_p=True))
    assert r["P2O5 (kg/ha)"].isna().all() and r["Calcário (kg/ha)"].notna().all()
    assert recomendar(L)["P2O5 (kg/ha)"].notna().all()


# ------------------------------------------------------------------ zonas: fusão, volume, parcelas, conferência
def test_fusao_termina_com_ilhas_isoladas():
    grande = box(0, 0, 1000, 1000)
    ilhas = [box(2000, 0, 2010, 10), box(3000, 0, 3010, 10)]
    out = _fundir_pequenos([(100.0, grande), (200.0, ilhas[0]), (300.0, ilhas[1])], area_min=5000)
    assert len(out) == 3


def test_volume_comprado_fecha_depois_das_zonas_e_parcelas_fecham():
    L = _laudo(25, com_2040=False)
    P = montar_projeto(L.dados, _kmls(25))
    M = superficies(P, 20)
    aj = Ajustes(calcario=AjusteProduto(media_alvo=1000), kcl=AjusteProduto(media_alvo=150), kcl_parcelado=True,
                 kcl_pct_1=60)
    calcular_doses(M, Parametros(), {}, aj)
    bl = gerar_todas_zonas(M, Parametros(), 3, aj=aj)
    z = bl[0].zonas
    passo = Parametros().arredondamento.calcario
    assert abs(z["Calcário"].dose_media - 1000) <= passo / 2 + 1e-6
    assert abs(z["KCl"].dose_media - 150) <= Parametros().arredondamento.kcl / 2 + 1e-6
    for (d1, g1), (d2, g2), (d, g) in zip(z["KCl 1ª aplicação"].poligonos, z["KCl 2ª aplicação"].poligonos,
                                          z["KCl"].poligonos):
        assert d1 + d2 == pytest.approx(d) and g1.equals(g) and g2.equals(g)
    conf = conferir(bl, P.epsg)
    assert (conf["Situação"] == "ok").all()
    with zipfile.ZipFile(io.BytesIO(zip_prescricoes(bl, P.epsg))) as zf:
        assert "CONFERENCIA.txt" in zf.namelist()


def test_planilha_e_mapa_usam_as_mesmas_regras():
    """No local de cada amostra (1 pixel por amostra, sem interpolação) o mapa dá a mesma dose da planilha, com os
    mesmos ajustes — diferenças só pelo arredondamento da planilha."""
    L = _laudo(20, com_2040=False)
    aj = Ajustes(calcario=AjusteProduto(pct=10), kcl=AjusteProduto(pct=-5), formula_p="11-52-00")
    res = recomendar(L, ajustes=aj)
    d = L.dados
    n = len(d)
    g = Grade(np.arange(n, dtype=float), np.zeros(1), np.ones((1, n), bool), 1.0)
    P = montar_projeto(d, _kmls())
    m = MapaTalhao(P.talhoes[0], g, {k: d[k].to_numpy(float)[None, :] for k in ("ca", "mg", "hal", "al", "k", "p",
                                                                               "s", "argila")})
    calcular_doses([m], Parametros(), {}, aj)
    arr = Parametros().arredondamento
    for col, prod, passo in (("Calcário (kg/ha)", "Calcário", arr.calcario), ("Gesso (kg/ha)", "Gesso", arr.gesso),
                             ("KCl (kg/ha)", "KCl", arr.kcl)):
        assert np.allclose(res[col].to_numpy(float), m.doses[prod][0], atol=passo / 2 + 1e-6), col


# ------------------------------------------------------------------ krigagem sem estrutura espacial
def test_krigagem_sem_estrutura_e_valor_uniforme():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 1000, 30), rng.uniform(0, 1000, 30)
    z = rng.normal(20, 5, 30)                                 # ruído puro, sem estrutura espacial
    gx, gy = np.meshgrid(np.linspace(0, 1000, 10), np.linspace(0, 1000, 10))
    v, aj = interpolar(x, y, z, gx, gy)
    assert aj.sem_estrutura and "NÃO COMPROVADA" in aj.descricao()
    v2, aj2 = interpolar(x, y, z, gx, gy, sem_estrutura="uniforme")
    assert np.allclose(v2, z.mean()) and aj2.metodo == "constante"
    zz = z.copy()
    zz[0] = 500
    _, aj3 = interpolar(x, y, zz, gx, gy, limitar_extremos=False)
    assert aj3.outliers == 0


# ------------------------------------------------------------------ excesso não é melhora
def test_muito_alto_nao_conta_como_melhora():
    from motor.comparacao import posicao
    assert posicao("v", 90) == posicao("v", 78) == 7.0
    assert posicao("k", 5.0) > posicao("k", 4.5)


# ------------------------------------------------------------------ paginação STAC por GET
def test_paginacao_stac_por_get(monkeypatch):
    from motor.sat import fontes

    class R:
        def __init__(self, js):
            self.js = js

        def raise_for_status(self):
            pass

        def json(self):
            return self.js
    paginas = {"p2": {"features": [{"id": 2}], "links": [{"rel": "next", "href": "p3"}]},
               "p3": {"features": [{"id": 3}], "links": []}}
    monkeypatch.setattr(fontes.requests, "post",
                        lambda *a, **k: R({"features": [{"id": 1}], "links": [{"rel": "next", "href": "p2"}]}))
    monkeypatch.setattr(fontes.requests, "get", lambda url, **k: R(paginas[url]))
    from datetime import date
    itens = fontes.buscar_stac("u", ["c"], (0, 0, 1, 1), date(2024, 1, 1), date(2024, 2, 1))
    assert [i["id"] for i in itens] == [1, 2, 3]
