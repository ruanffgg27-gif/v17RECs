"""Testes das regras com valores conferidos à mão."""
import io
import math

import pandas as pd
import pytest

from motor import Parametros, gerar_excel, ler_laudo, recomendar
from motor import regras
from motor.parametros import Calcario, Gesso


# ---------------------------------------------------------------- calcário
def test_calcario_modelo_base():
    # CTC* = 39+15+35+5,6 = 94,6 ; Ca: 85·(0,55·94,6−39)=1107,6 ; Mg: 345·(0,194·94,6−15)=1156,6
    r = regras.calcario(39, 15, 35, 0)
    assert r.criterio == "Mg"
    assert r.bruto == pytest.approx(1156.58, abs=0.01)


def test_calcario_minimo_e_maximo():
    assert regras.calcario(80, 30, 10, 0).dose == 400
    r = regras.calcario(5, 2, 80, 5)
    assert r.dose == 3800 and r.bruto > 3800


def test_calcario_abertura_depois_dos_limites():
    assert regras.calcario(80, 30, 10, 0, abertura=True).dose == pytest.approx(720)
    assert regras.calcario(5, 2, 80, 5, abertura=True).dose == pytest.approx(3800 * 1.8)
    base = regras.calcario(39, 15, 35, 0).dose
    assert regras.calcario(39, 15, 35, 0, abertura=True).dose == pytest.approx(base * 1.8)


def test_calcario_abertura_antes_dos_limites():
    p = Calcario(abertura_antes_dos_limites=True)
    assert regras.calcario(5, 2, 80, 5, abertura=True, p=p).dose == 3800


def test_calcario_referencia_fatores_unitarios():
    f = regras.fatores_calcario(Calcario())
    for k in ("ca", "mg", "prnt"):
        assert f[k] == pytest.approx(1.0)


def test_calcario_prnt_maior_reduz_dose_proporcionalmente():
    base = regras.calcario(39, 15, 35, 1).bruto
    novo = regras.calcario(39, 15, 35, 1, p=Calcario(prnt=90)).bruto
    assert novo == pytest.approx(base * 82 / 90)


def test_calcario_mais_mgo_reduz_criterio_mg():
    base = regras.calcario(39, 15, 35, 0)
    mais_mg = regras.calcario(39, 15, 35, 0, p=Calcario(mgo=20, prnt=82))
    assert mais_mg.bruto < base.bruto
    assert mais_mg.criterio == "Ca"   # com mais MgO o Ca passa a limitar


def test_calcario_sem_dados():
    assert regras.calcario(None, 15, 35, 0).dose is None


# ------------------------------------------------------------------- gesso
@pytest.mark.parametrize("argila,alvo", [(150, 35), (199.9, 35), (200, 25), (400, 25), (400.1, 20), (600, 20)])
def test_gesso_faixas(argila, alvo):
    assert regras.s_alvo(argila, Gesso()) == alvo


def test_gesso_equacao():
    # ((20−13)·1000/75)·439/100 = 409,73
    r = regras.gesso(13, 439)
    assert r.dose == pytest.approx(409.733, abs=0.01)


def test_gesso_minimo_e_maximo():
    assert regras.gesso(30, 500).dose == 300          # S acima do alvo
    r = regras.gesso(2, 180)                          # (35−2)·13,33·1,8 = 792
    assert r.dose == pytest.approx(792)
    assert regras.gesso(1, 700).dose == pytest.approx(1773.33, abs=0.01)  # abaixo do teto
    r = regras.gesso(0, 700)                                              # 1866,7 → 1800
    assert r.dose == 1800 and r.bruto == pytest.approx(1866.67, abs=0.01)


# ----------------------------------------------------------------- P e K
def test_p2o5():
    assert regras.p2o5(19.3).dose == pytest.approx(141.84 * 19.3 ** -0.218)
    assert regras.p2o5(1).dose == 105
    assert regras.p2o5(500).dose == 55


def test_kcl():
    assert regras.kcl(3.34).dose == pytest.approx(164.66 * 3.34 ** -0.269)
    assert regras.kcl(10).dose == 100
    assert regras.kcl(0).dose is None


# ------------------------------------------------------------------ leitor
def _laudo_xlsx(unid_ca="mmolc/dm³", unid_arg="g/kg", arg=450):
    cab = ["N Lab", "Proprietário", "Propriedade", "Identificação", "Talhão", "Profundidade",
           "pH CaCl2", "P resina", "M.O.", "K", "Ca", "Mg", "H+Al", "Al", "S-SO4", "Argila"]
    uni = ["", "", "", "", "", "", "", "mg/dm³", "g/dm³", unid_ca, unid_ca, unid_ca, unid_ca, unid_ca,
           "mg/dm³", unid_arg]
    lin = [1, "Fulano", "Faz. Teste", "A1", "T1", "0-20", 5.2, 20, 25, 3.3, 39, 15, 35, 0, 13, arg]
    buf = io.BytesIO()
    pd.DataFrame([cab, uni, lin, [None] * 16, ["Responsável técnico"] + [None] * 15]).to_excel(
        buf, header=False, index=False)
    buf.seek(0)
    return buf


def test_leitor_basico():
    L = ler_laudo(_laudo_xlsx(), "x.xlsx")
    assert len(L.dados) == 1
    assert L.dados.loc[0, "ca"] == 39 and L.dados.loc[0, "talhao"] == "T1"


def test_leitor_converte_cmolc_e_porcentagem():
    L = ler_laudo(_laudo_xlsx(unid_ca="cmolc/dm³", unid_arg="%", arg=45), "x.xlsx")
    d = L.dados.loc[0]
    assert d["ca"] == 390 and d["argila"] == 450
    assert any("cmolc" in a for a in L.avisos)


def test_ponta_a_ponta_gera_excel():
    L = ler_laudo(_laudo_xlsx(), "x.xlsx")
    res = recomendar(L, Parametros(), {"T1": True})
    assert res.loc[0, "Abertura"] == "Sim"
    xlsx = gerar_excel(res, L, Parametros(), "x.xlsx")
    abas = pd.read_excel(io.BytesIO(xlsx), sheet_name=None)
    assert {"Recomendações", "Resumo por talhão", "Memória de cálculo", "Parâmetros",
            "Laudo original"} <= set(abas)


# ------------------------------------------------------- versão 2: novidades
from motor import Ajustes, AjusteProduto, gerar_excel_interpretacao  # noqa: E402
from motor import interpretacao as it  # noqa: E402
from motor.leitor import e_subsuperficial  # noqa: E402


def _laudo_camadas():
    cab = ["N Lab", "Proprietário", "Propriedade", "Identificação", "Talhão", "Profundidade",
           "pH CaCl2", "P resina", "M.O.", "K", "Ca", "Mg", "H+Al", "Al", "S-SO4", "Argila", "Zn"]
    uni = ["", "", "", "", "", "", "", "mg/dm³", "g/dm³"] + ["mmolc/dm³"] * 5 + ["mg/dm³", "g/kg", "mg/dm³"]
    lin = [
        [1, "P", "F", "A1", "T1", "0-20", 5.2, 20, 25, 3.3, 39, 15, 35, 0, 13, 450, 1.2],
        [2, "P", "F", "A1", "T1", "20-40", 4.6, 5, 15, 1.5, 15, 6, 40, 4, 25, 480, 0.4],
        [3, "P", "F", "A2", "T1", "0-20", 5.4, 30, 28, 4.5, 45, 18, 30, 0, 8, 470, 2.0],
        [4, "P", "F", "A3", "T2", "0 a 20", 5.0, 12, 20, 2.0, 25, 9, 38, 1, 5, 180, 0.8],
    ]
    buf = io.BytesIO()
    pd.DataFrame([cab, uni] + lin).to_excel(buf, header=False, index=False)
    buf.seek(0)
    return ler_laudo(buf, "c.xlsx")


@pytest.mark.parametrize("prof,sub", [("0-20", False), ("20-40", True), ("20 a 40 cm", True),
                                      ("0 - 20", False), ("", False), ("40-60", True)])
def test_camada_subsuperficial(prof, sub):
    assert e_subsuperficial(prof) is sub


def test_subsuperficial_fora_da_conta_e_da_saida():
    L = _laudo_camadas()
    res = recomendar(L)
    sub = res[res["Camada subsuperficial"]]
    assert len(sub) == 1 and sub["Calcário (kg/ha)"].isna().all()
    xl = pd.read_excel(io.BytesIO(gerar_excel(res, L, Parametros(), "c.xlsx")), sheet_name="Recomendações",
                       header=5)
    assert "20-40" not in set(xl["Profundidade (cm)"].dropna())
    assert len(xl["N Lab"].dropna().loc[lambda s: s.astype(str).str.isdigit()]) == 3


def test_ajuste_percentual_e_media_alvo():
    L = _laudo_camadas()
    base = recomendar(L)
    r10 = recomendar(L, ajustes=Ajustes(kcl=AjusteProduto(pct=10)))
    ok = ~base["Camada subsuperficial"]
    assert r10.attrs["ajustes"]["KCl"]["fator"] == pytest.approx(1.10)
    alvo = recomendar(L, ajustes=Ajustes(calcario=AjusteProduto(media_alvo=2000)))
    # média exata antes do arredondamento; depois dele, diferença < passo de 10 kg
    assert alvo.loc[ok, "Calcário (kg/ha)"].mean() == pytest.approx(2000, abs=10)
    # proporções mantidas entre amostras
    b, a = base.loc[ok, "Calcário (kg/ha)"].values, alvo.loc[ok, "Calcário (kg/ha)"].values
    assert (a / b).std() < 0.02


def test_s_elementar_e_produto_p():
    L = _laudo_camadas()
    res = recomendar(L, ajustes=Ajustes(gesso_por_s_elementar=True, formula_p="11-52-00"))
    assert "S elementar (kg/ha)" in res and "Gesso (kg/ha)" not in res
    g = res.loc[0, "Gesso substituído (kg/ha)"]
    assert res.loc[0, "S elementar (kg/ha)"] == pytest.approx(round(4.1904 * g ** 0.3754), abs=1)
    col = "Produto 11-52-00 (kg/ha)"
    assert res.loc[0, col] == pytest.approx(round(res.loc[0, "P2O5 (kg/ha)"] / 0.52 / 5) * 5)
    assert regras.ler_formula("00-25-00") == (0, 25, 0)


def test_classes_de_teores():
    assert it.classificar("ph", 5.1) == "Bom"
    assert it.classificar("ph", 7.0) == "Muito alto"       # acima da maior faixa
    assert it.classificar("hal", 15) == "Excelente"        # escala invertida: o mais baixo é o melhor (v15)
    assert it.classificar("hal", 95) == "Crítico"
    assert it.classificar("ctc", 190) == "Excelente"       # faixa corrigida (180–205)
    assert it.classificar("p", 6) == "Muito baixo"         # limite inferior pertence à faixa de cima


def test_interpretacao_gera_excel():
    L = _laudo_camadas()
    res = recomendar(L)
    m = it.medias_por_talhao(L)
    assert list(m["Talhão"]) == ["T1", "T2"] and m.loc[0, "Amostras"] == 2
    abas = pd.read_excel(io.BytesIO(gerar_excel_interpretacao(res, L, "c.xlsx")), sheet_name=None)
    assert {"Diagnóstico 0-20 cm", "Camada 20-40 cm", "Gráficos", "Classes (lista)",
            "Faixas de referência"} <= set(abas)


def test_kcl_parcelado_soma_igual_ao_total():
    L = _laudo_camadas()
    res = recomendar(L, ajustes=Ajustes(kcl_parcelado=True, kcl_pct_1=60, kcl=AjusteProduto(pct=15)))
    ok = ~res["Camada subsuperficial"]
    k, k1, k2 = (res.loc[ok, c] for c in ("KCl (kg/ha)", "KCl 1ª aplicação (kg/ha)", "KCl 2ª aplicação (kg/ha)"))
    assert ((k1 + k2) == k).all()
    assert (abs(k1 / k - 0.60) < 0.05).all()
    assert res.loc[~ok, "KCl 1ª aplicação (kg/ha)"].isna().all()
    xl = pd.read_excel(io.BytesIO(gerar_excel(res, L, Parametros(), "c.xlsx")), sheet_name="Recomendações", header=5)
    assert {"KCl 1ª aplicação (kg/ha)", "KCl 2ª aplicação (kg/ha)"} <= set(xl.columns)
    assert recomendar(L).columns.str.contains("aplicação").sum() == 0   # desligado por padrão
