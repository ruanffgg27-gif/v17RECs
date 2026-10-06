"""Testes da versão 17: correção de P, escolha das prescrições, sementes (degrau, amplitude, ajuste manual),
textos dentro da moldura, balanço hídrico (ET0, lacunas, armazenamento) e leitura do NISAR."""
import io
from datetime import date, datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from motor import Ajustes, Parametros, recomendar
from motor.geo import ler_kml, montar_projeto
from motor.prescricao import (calcular_doses, filtrar_prescricoes, gerar_todas_zonas, superficies, zip_prescricoes)
from motor.processar import COL_P, COL_PC, colunas_doses
from motor.sat import agua, nisar
from tests.test_mapas import _kml_perimetro, _kml_pontos, _laudo


def _proj(n=25):
    L = _laudo(n, com_2040=False)
    return L, montar_projeto(L.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                       ler_kml(_kml_pontos(n), "Pontos_TH_1.kml")])


# ------------------------------------------------------------------ correção de P
def test_correcao_de_p_e_coluna_a_parte_e_nao_muda_a_manutencao():
    L, _ = _proj()
    base = recomendar(L)
    assert COL_PC not in base                                         # padrão: só manutenção
    r = recomendar(L, ajustes=Ajustes(p_correcao=120))
    assert r[COL_P].equals(base[COL_P]) and abs(r[COL_PC].astype(float).mean() - 120) < 1.5
    assert np.corrcoef(r["P resina"].astype(float), r[COL_PC].astype(float))[0, 1] < -0.9   # mais onde há menos P
    assert colunas_doses(r).index(COL_PC) == colunas_doses(r).index(COL_P) + 1
    prod = recomendar(L, ajustes=Ajustes(p_correcao=300, p_correcao_formula="00-18-00", p_correcao_em_produto=True))
    assert abs(prod["Correção 00-18-00 (kg/ha)"].astype(float).mean() - 300) < 5
    assert abs(prod[COL_PC].astype(float).mean() - 54) < 1.5         # 300 kg × 18%
    unif = recomendar(L, ajustes=Ajustes(p_correcao=100, p_correcao_variavel=False))
    assert set(unif[COL_PC].dropna().astype(float)) == {100.0}
    assert Ajustes(p_correcao=200, p_correcao_em_produto=True).correcao_p2o5() == 0.0   # produto sem formulação


def test_correcao_e_filtro_de_prescricoes_no_mapa():
    L, P = _proj()
    aj = Ajustes(p_correcao=250, p_correcao_formula="00-18-00", p_correcao_em_produto=True)
    M = superficies(P, 20, modo="fiel")
    calcular_doses(M, Parametros(), {}, aj)
    assert "Correção 00-18-00" in M[0].doses and "P2O5 correção" not in M[0].doses
    fora = filtrar_prescricoes(M, ["Calagem", "Fósforo", "Correção de P", "Potássio"])
    assert fora == ["Gessagem"] and "Gesso" not in M[0].doses and "s" in M[0].atributos   # mapas de fertilidade ficam
    bl = gerar_todas_zonas(M, Parametros(), 6, aj=aj)
    z = bl[0].zonas
    assert "Gesso" not in z and abs(z["Correção 00-18-00"].dose_media - 250) <= 2.5
    import zipfile
    nomes = zipfile.ZipFile(io.BytesIO(zip_prescricoes(bl, P.epsg))).namelist()
    assert any(n.endswith("PCOR_T1.shp") for n in nomes) and not any("GES_" in n for n in nomes)


# ------------------------------------------------------------------ sementes
@pytest.mark.parametrize("media,esticar,classes", [(13.0, False, 8), (3.0, False, 8), (3.0, True, 10)])
def test_sementes_chegam_ao_numero_de_doses(media, esticar, classes):
    from motor import sementes as sm
    L, P = _proj()
    M = superficies(P, 20, sm.ATRIB_SEMENTES, modo="fiel")
    cfg = sm.ConfigSementes(media=media, classes=classes, esticar=esticar)
    sm.gerar(M, L.dados, cfg)
    z = M[0].zonas["Sementes"]
    passo = cfg.passo_de(media)
    assert len(z.niveis) == classes                                    # milho (3 sem/m) não fica mais com 3 doses
    assert abs(z.dose_media - media) <= passo
    assert all(abs(d / passo - round(d / passo)) < 1e-6 for d in z.niveis)
    assert min(z.niveis) >= media * 0.8 - passo and max(z.niveis) <= media * 1.2 + passo


def test_ajuste_manual_das_doses_de_sementes():
    from motor import book_sementes
    from motor import sementes as sm
    from motor.book import DadosBook
    L, P = _proj()
    M = superficies(P, 20, sm.ATRIB_SEMENTES, modo="fiel")
    cfg = sm.ConfigSementes(media=13.0, classes=6)
    res = sm.gerar(M, L.dados, cfg)
    rec = M[0].zonas["Sementes"]
    alterados = sm.ajustar_doses(M[0], rec, {rec.niveis[0]: 10.0, rec.niveis[-1]: 16.0})
    z = M[0].zonas["Sementes"]
    assert len(alterados) == 2 and min(z.niveis) == 10.0 and max(z.niveis) == 16.0
    assert [g for _, g in z.poligonos] == [g for _, g in rec.poligonos]            # só a dose muda
    t = sm.tabela_faixas(M, {M[0].talhao.nome: rec})
    assert (t["Recomendado (sem/m)"] != t["Ajustado (sem/m)"]).sum() == 2
    assert abs(t["Área (ha)"].sum() - z.area_ha) < 0.05
    sm.ajustar_doses(M[0], rec, {})                                                # volta ao recomendado
    assert M[0].zonas["Sementes"].niveis == rec.niveis
    pdf = book_sementes.montar(P, M, DadosBook(marca="Atria", usar_internet=False), cfg, res,
                               manuais={M[0].talhao.nome: alterados})
    assert pdf[:4] == b"%PDF"


# ------------------------------------------------------------------ textos dentro da moldura
def test_titulo_e_texto_corrido_cabem():
    from types import SimpleNamespace
    from motor import book as bk
    ctx = SimpleNamespace(marca=bk.MARCAS["Atria"])
    fig = plt.figure(figsize=bk.A4)
    bk.base(fig, ctx, 1)
    nomes = ", ".join(f"Gleba Santo Antônio da Serra {i}" for i in range(1, 12))
    bk.titulo(fig, ctx, "POTÁSSIO · 1ª APLICAÇÃO - " + nomes.upper(), "KCl (60% K₂O)  ·  " + nomes, tam=18)
    bk.texto_corrido(fig, 0.13, 0.12, "; ".join(f"TH {i:02d}: P, K, S, B, Zn, Mn, Cu" for i in range(1, 40)) + ".",
                     larg=0.75, y_min=bk.MOLD_Y0 + 0.018)
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    for t in fig.texts:
        (x0, y0), (x1, y1) = inv.transform(t.get_window_extent(r))
        assert 0.07 < x0 and x1 < 0.93, t.get_text()[:30]
        assert y0 > bk.MOLD_Y0 or t.get_text().isdigit(), t.get_text()[:30]         # nada abaixo da moldura
    assert fig._cor_marca == ctx.marca["cor"]                                       # cor da barra de escala
    plt.close(fig)


def test_msavi2_tem_a_legenda_invertida():
    from motor.sat.relatorio import CMAP_INDICE, LEGENDA_INDICE
    baixo, alto = CMAP_INDICE["msavi2"](0.0), CMAP_INDICE["msavi2"](1.0)
    assert baixo[0] > 0.9 and baixo[1] > 0.8 and alto[2] > alto[1]                  # amarelo → roxo
    assert LEGENDA_INDICE["msavi2"].startswith("amarelo")


# ------------------------------------------------------------------ balanço hídrico
def _power(fim, dias=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(fim - timedelta(days=dias - 1), fim, freq="D")
    tmax = 31 + rng.normal(0, 1.5, len(idx))
    return pd.DataFrame({"T2M_MAX": tmax, "T2M_MIN": tmax - 11, "RH2M": 65.0, "WS2M": 2.0,
                         "ALLSKY_SFC_SW_DWN": 21.0, "PS": 96.0,
                         "PRECTOTCORR": np.where(rng.random(len(idx)) < 0.25, rng.gamma(1.2, 14, len(idx)), 0.0)},
                        index=idx)


def test_et0_penman_monteith_exemplo_da_fao():
    """FAO-56, exemplo 18 (Bruxelas, 6 de julho): ET0 = 3,9 mm/dia. Aqui a umidade entra como média diária
    (o exemplo usa máxima e mínima), então aceita-se ±0,3."""
    df = pd.DataFrame({"T2M_MAX": [21.5], "T2M_MIN": [12.3], "RH2M": [73.5], "WS2M": [2.078],
                       "ALLSKY_SFC_SW_DWN": [22.07], "PS": [100.1]}, index=pd.to_datetime(["2015-07-06"]))
    e = agua.et0_fao56(df, 50.8, 100)
    assert e["metodo"].iloc[0] == "PM" and abs(e["et0"].iloc[0] - 3.9) < 0.3
    sem_rad = agua.et0_fao56(df.assign(ALLSKY_SFC_SW_DWN=np.nan), 50.8, 100)
    assert sem_rad["metodo"].iloc[0] == "HS" and 2.5 < sem_rad["et0"].iloc[0] < 5.5
    sem_temp = agua.et0_fao56(df.assign(T2M_MAX=np.nan), 50.8, 100)
    assert sem_temp["metodo"].iloc[0] == "" and np.isnan(sem_temp["et0"].iloc[0])


def test_balanco_nao_trata_dado_ausente_como_zero_e_separa_fontes():
    fim = date(2026, 10, 4)
    pw = _power(fim)
    chirps = pw["PRECTOTCORR"].iloc[:-40] * 1.1
    pw.loc[pw.index[-20:-15], ["PRECTOTCORR", "T2M_MAX"]] = np.nan                  # 5 dias sem nada
    a = agua.analisar(pw, chirps, -22.2, 400, agua.ParametrosAgua(), fim)
    r = a.resumo
    assert r["dias"] == 90 and r["dias_chuva"] == 85 and r["dias_et0"] == 85 and r["dias_saldo"] == 85
    assert r["dias_chirps"] == 50 and r["dias_power"] == 35
    assert abs(r["saldo"] - (r["chuva_dias_saldo"] - r["et0_dias_saldo"])) < 1e-6
    assert a.serie["chuva"].isna().sum() == 5 and any("sem dado de chuva" in x for x in a.avisos)
    # sem parâmetros do usuário: nada de armazenamento
    assert a.armazenamento is None and "cultura" in a.motivo_sem_armazenamento and not a.cenario


def test_armazenamento_so_com_parametros_e_cenarios_de_irrigacao():
    fim = date(2026, 10, 4)
    pw = _power(fim)
    incompleto = agua.ParametrosAgua("soja", date(2026, 9, 1), None, 60, None)
    assert agua.analisar(pw, None, -22.2, 400, incompleto, fim).armazenamento is None
    par = agua.ParametrosAgua("soja", date(2026, 9, 1), None, 60, 1.2)
    a = agua.analisar(pw, None, -22.2, 400, par, fim)
    arm = a.armazenamento
    assert a.cad_total == 72 and len(arm) == 90 and arm["arm"].between(0, 72).all()
    assert "SEM irrigação" in a.cenario and "não informada" in a.cenario             # desconhecida ≠ inexistente
    assert (arm["arm_min"] <= arm["arm"] + 1e-9).all() and (arm["arm"] <= arm["arm_max"] + 1e-9).all()
    assert arm["kc"].iloc[0] == agua.KC_SEM_CULTURA and arm["fase"].iloc[-1] == "desenvolvimento"
    seq = agua.analisar(pw, None, -22.2, 400, agua.ParametrosAgua("soja", date(2026, 9, 1), None, 60, 1.2,
                                                                  "sequeiro"), fim)
    assert seq.cenario.startswith("sequeiro") and seq.armazenamento["arm"].equals(arm["arm"])
    irr = agua.analisar(pw, None, -22.2, 400, agua.ParametrosAgua(
        "soja", date(2026, 9, 1), None, 60, 1.2, "irrigado", ((date(2026, 9, 28), 25.0),)), fim)
    assert "irrigações informadas" in irr.cenario
    assert irr.armazenamento["arm"].iloc[-1] >= arm["arm"].iloc[-1]
    assert irr.armazenamento.loc["2026-09-28", "irrigacao"] == 25.0
    # muitos dias sem dado: não apresenta
    pw2 = pw.copy()
    pw2.loc[pw2.index[-30:-15], "PRECTOTCORR"] = np.nan
    falho = agua.analisar(pw2, None, -22.2, 400, par, fim)
    assert falho.armazenamento is None and "sem chuva ou ET" in falho.motivo_sem_armazenamento


def test_balanco_fecha_e_estresse_reduz_consumo():
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    s = pd.DataFrame({"chuva": 0.0, "et0": 5.0}, index=idx)
    s.loc[idx[30], "chuva"] = 200.0
    r = agua.simular(s, pd.Series(1.0, index=idx), 100.0, 0.5, 0.0)
    assert r["ks"].iloc[5] == 1.0 and r["ks"].iloc[25] < 1.0 and r["arm"].iloc[29] > 0
    assert r["arm"].iloc[30] == 100.0 and r["excedente"].iloc[30] > 50
    # conservação: variação do armazenamento = entradas − consumo − excedente
    assert abs((r["arm"].iloc[-1] - 100.0) - (200.0 - r["eta"].sum() - r["excedente"].sum())) < 1e-6


# ------------------------------------------------------------------ NISAR
def _h5(pol, valor=0.27, invalido=0.0, res=200.0):
    import h5py
    x0, y0, x1, y1 = pol.bounds
    xs = np.arange(np.floor(x0 / res) * res - 5 * res + res / 2, x1 + 5 * res, res)
    ys = np.arange(np.ceil(y1 / res) * res + 5 * res - res / 2, y0 - 5 * res, -res)
    rng = np.random.default_rng(0)
    sm = np.full((len(ys), len(xs)), valor, "f4")
    flag = np.zeros(sm.shape, "u2")
    ruim = rng.random(sm.shape) < invalido
    flag[ruim], sm[ruim] = 6, -9999
    b = io.BytesIO()
    with h5py.File(b, "w") as f:
        g = f.create_group(nisar.GRUPO)
        g.create_dataset("soilMoisture", data=sm).attrs["_FillValue"] = np.float32(-9999)
        g.create_dataset("soilMoistureUncertainty", data=np.full(sm.shape, 0.04, "f4"))
        g.create_dataset("retrievalQualityFlag", data=flag)
        g.create_dataset("surfaceQualityFlag", data=np.zeros(sm.shape, "i2"))
        g.create_dataset("xCoordinates", data=xs)
        g.create_dataset("yCoordinates", data=ys)
    return b.getvalue()


def _area(lado_m=1500):
    from shapely.geometry import box
    from motor.geo import epsg_utm_sirgas, projetar
    from motor.sat.base import Area
    d = lado_m / 111000
    ll = box(-54.80, -22.20, -54.80 + d, -22.20 + d)
    ep = epsg_utm_sirgas(-54.8, -22.2)
    return Area("TH", ll, projetar(ll, ep), ep)


def _granulos(fim, n=4):
    out = []
    for k in range(n):
        d = datetime.combine(fim - timedelta(days=2 + 6 * k), datetime.min.time())
        nome = f"NISAR_L3_PR_SME2_{30 - k:03d}_082_D_100_2005_DHDH_A_{d:%Y%m%dT%H%M%S}_{d:%Y%m%dT%H%M%S}_P05023_N_F_J_001"
        out.append({"nome": nome, "data": d, "url": f"mem://{k}", "chave": f"{30 - k:03d}-082-D",
                    "passagem": nisar._passagem(nome)[1]})
    return out


def test_nisar_usa_mascara_de_qualidade_e_conta_pixels_de_borda():
    from motor.geo import projetar
    A = _area(1500)
    pol = projetar(A.poligono_ll, nisar.EPSG_GRADE)
    fim = date(2026, 10, 4)
    arqs = {"mem://0": _h5(pol, 0.21, 1.0), "mem://1": _h5(pol, 0.25, 0.3), "mem://2": _h5(pol, 0.29),
            "mem://3": _h5(pol, 0.31)}
    n = nisar.obter(A, 90, fim, buscar=lambda *a, **k: _granulos(fim), abrir=lambda g: io.BytesIO(arqs[g["url"]]))
    assert n.situacao == "ok" and len(n.observacoes) == 4 and n.catalogados == 4
    assert not n.observacoes[-1].valida                              # passagem mais recente: tudo mascarado
    u = n.ultima_valida                                              # a válida mais recente é a anterior
    assert abs(u.mediana - 0.25) < 1e-6 and 0 < u.n_validos < u.n_pixels and u.passagem.startswith("trilha 082")
    assert 0 < u.n_inteiros < u.n_pixels and abs(u.incerteza - 0.04) < 1e-6
    assert 40 <= u.n_pixels <= 70                                    # ~225 ha ÷ 4 ha por pixel
    assert np.isnan(u.grade["sm"][~u.grade["valido"]]).all()         # preenchimento nunca vira valor


def test_nisar_indisponivel_informa_sem_observacao_valida():
    A = _area(300)                                                   # talhão de 9 ha
    fim = date(2026, 10, 4)
    assert nisar.obter(A, 90, fim, buscar=lambda *a, **k: []).mensagem.startswith(nisar.SEM_OBS)
    sem_login = nisar.obter(A, 90, fim, buscar=lambda *a, **k: _granulos(fim))
    assert sem_login.situacao == "sem_credencial" and sem_login.catalogados == 4
    assert sem_login.mensagem.startswith(nisar.SEM_OBS) and "login Earthdata" in sem_login.mensagem
    assert sem_login.ultima_valida is None
    from motor.geo import projetar
    pol = projetar(A.poligono_ll, nisar.EPSG_GRADE)
    tudo_ruim = nisar.obter(A, 90, fim, buscar=lambda *a, **k: _granulos(fim, 2),
                            abrir=lambda g: io.BytesIO(_h5(pol, 0.2, 1.0)))
    assert tudo_ruim.situacao == "sem_valido" and tudo_ruim.mensagem.startswith(nisar.SEM_OBS)
    pequeno = nisar.obter(A, 90, fim, buscar=lambda *a, **k: _granulos(fim, 1),
                          abrir=lambda g: io.BytesIO(_h5(pol, 0.2)))
    assert pequeno.situacao == "ok" and any("pixel" in a for a in pequeno.avisos)   # talhão pequeno: aviso


def test_nisar_catalogo_e_leitura_por_faixas():
    js = {"feed": {"entry": [{
        "producer_granule_id": "NISAR_L3_PR_SME2_032_010_D_100_2005_DHDH_A_20260930T215706_20260930T215741_P05023_N_F_J_001",
        "time_start": "2026-09-30T21:57:06.000Z",
        "links": [{"href": "s3://x/a.h5"}, {"href": "https://nisar.asf.earthdatacloud.nasa.gov/a/a_QA_STATS.h5"},
                  {"href": "https://nisar.asf.earthdatacloud.nasa.gov/a/a.h5"}]}]}}
    visto = {}
    g = nisar.buscar_granulos((-55, -23, -54, -22), date(2026, 7, 1), date(2026, 10, 4),
                              pedir=lambda p: visto.update(p) or js)
    assert visto["collection_concept_id"] == nisar.COLECAO and visto["sort_key"] == "-start_date"
    assert g[0]["url"].endswith("/a.h5") and g[0]["chave"] == "032-010-D" and "descendente" in g[0]["passagem"]
    from motor.geo import projetar
    pol = projetar(_area(1500).poligono_ll, nisar.EPSG_GRADE)
    dados = _h5(pol, 0.3)

    class Resp:
        def __init__(self, a, b):
            self.content, self.status_code, self.url = dados[a:b + 1], 206, "https://nisar.asf.exemplo/a.h5"
            self.headers = {"Content-Range": f"bytes {a}-{b}/{len(dados)}"}

        def raise_for_status(self):
            pass

    class Sessao:
        pedidos = 0

        def get(self, url, headers=None, **k):
            a, b = map(int, headers["Range"].split("=")[1].split("-"))
            Sessao.pedidos += 1
            return Resp(a, b)
    f = nisar.ArquivoRemoto("https://nisar.asf.exemplo/a.h5", Sessao(), bloco=64 * 1024)
    o = nisar.extrair(f, pol, g[0])
    assert o.valida and abs(o.mediana - 0.3) < 1e-6 and Sessao.pedidos >= 2 and f.baixado <= len(dados) + 65536


def test_pagina_de_agua_entra_no_relatorio_e_na_planilha():
    from motor.sat import exportar, relatorio
    from motor.sat.diagnostico import MODULOS, Diagnostico
    assert "agua" in MODULOS
    A = _area(1500)
    fim = date(2026, 10, 4)
    D = Diagnostico(A)
    D.agua = agua.analisar(_power(fim), None, -22.2, 400, agua.ParametrosAgua(), fim,
                           nisar.Nisar("sem_granulo", nisar.SEM_OBS + " — teste."))
    pdf = relatorio.montar(D, relatorio.DadosRelatorio(talhao="TH"))
    assert pdf[:4] == b"%PDF"
    t = exportar.tabelas(D)
    assert len(t["Água – balanço diário"]) == 90 and "Água – armazenamento" not in t
    assert nisar.SEM_OBS in t["NISAR – umidade superficial"]["Situação"].iloc[0]
