"""Aplica as regras a todas as amostras de um laudo.

Duas etapas:
1. Doses pelas regras (equações + limites) — só para amostras da camada 0-20.
2. Ajustes do usuário: substituição do gesso por S elementar, ajuste percentual
   ou para uma média-alvo (mantendo as proporções), produto fosfatado e arredondamento.
"""
from __future__ import annotations

import math

import pandas as pd

from . import regras
from .leitor import Laudo
from .parametros import Ajustes, AjusteProduto, Parametros

ID_COLS = ["N Lab", "Proprietário", "Propriedade", "Identificação", "Talhão", "Profundidade (cm)"]
COL_CAL = "Calcário (kg/ha)"
COL_GES = "Gesso (kg/ha)"
COL_S0 = "S elementar (kg/ha)"
COL_P = "P2O5 (kg/ha)"
COL_K = "KCl (kg/ha)"
COL_K1 = "KCl 1ª aplicação (kg/ha)"
COL_K2 = "KCl 2ª aplicação (kg/ha)"
COL_SUB = "Camada subsuperficial"
COL_PC = "P correção (kg/ha)"          # P2O5 de correção (além da manutenção)


def nome_correcao_p(formula: str) -> str | None:
    f = regras.ler_formula(formula)
    if not f or f[1] <= 0:
        return None
    return f"Correção {int(f[0]):02d}-{int(f[1]):02d}-{int(f[2]):02d} (kg/ha)"


def nome_produto_p(formula: str) -> str | None:
    f = regras.ler_formula(formula)
    if not f or f[1] <= 0:
        return None
    return f"Produto {int(f[0]):02d}-{int(f[1]):02d}-{int(f[2]):02d} (kg/ha)"


def colunas_doses(res: pd.DataFrame) -> list[str]:
    """Colunas de dose presentes, na ordem de exibição."""
    ordem = [COL_CAL, COL_GES, COL_S0, COL_P] + \
            [c for c in res.columns if c.startswith("Produto ")] + [COL_PC] + \
            [c for c in res.columns if c.startswith("Correção ")] + [COL_K, COL_K1, COL_K2]
    return [c for c in ordem if c in res.columns]


def colunas_doses_totais(res: pd.DataFrame) -> list[str]:
    """Doses sem as parcelas (para gráficos e médias)."""
    return [c for c in colunas_doses(res) if c not in (COL_K1, COL_K2)]


def colunas_recomendacao(res: pd.DataFrame) -> list[str]:
    extra = ["Arquivo"] if "Arquivo" in res else []
    return extra + ID_COLS + ["Abertura"] + colunas_doses(res) + ["Observações"]


def _escalar(valores: pd.Series, aj: AjusteProduto) -> tuple[pd.Series, float, float]:
    """Devolve (valores ajustados, fator, média original)."""
    media = valores.mean()
    if aj.media_alvo is not None and media and not math.isnan(media) and media > 0:
        fator = aj.media_alvo / media
    else:
        fator = 1 + aj.pct / 100
    return valores * fator, fator, media


def recomendar(laudo: Laudo, par: Parametros | None = None,
               abertura: dict[str, bool] | None = None,
               ajustes: Ajustes | None = None) -> pd.DataFrame:
    """Uma linha por amostra do laudo (inclusive 20-40, marcadas e sem dose).

    abertura: {talhão: True/False}. Talhões ausentes contam como área já cultivada.
    O resumo dos ajustes aplicados fica em `res.attrs["ajustes"]`.
    """
    par = par or Parametros()
    abertura = abertura or {}
    aj = ajustes or Ajustes()
    arr = par.arredondamento

    linhas = []
    for r in laudo.dados.to_dict("records"):
        sub = bool(r.get("subsuperficial", False))
        ab = bool(abertura.get(r["talhao"], False))
        base = {
            "N Lab": int(r["n_lab"]) if str(r["n_lab"]).isdigit() else r["n_lab"],
            "Proprietário": r["proprietario"], "Propriedade": r["propriedade"],
            "Identificação": r["identificacao"], "Talhão": r["talhao"],
            "Profundidade (cm)": r["profundidade"], COL_SUB: sub,
            "Abertura": "" if sub else ("Sim" if ab else "Não"),
            "Ca": r["ca"], "Mg": r["mg"], "K": r["k"], "H+Al": r["hal"], "Al": r["al"],
            "P resina": r["p"], "S-SO4": r["s"], "Argila (g/kg)": r["argila"],
            "Linha no laudo": r["_linha"],
        }
        if "_arquivo" in r:
            base["Arquivo"] = r["_arquivo"]
        if sub:
            base["Observações"] = "Camada subsuperficial: não entra na recomendação"
            linhas.append(base)
            continue
        cal = regras.calcario(r["ca"], r["mg"], r["hal"], r["al"], r["k"], ab, par.calcario)
        ges = regras.gesso(r["s"], r["argila"], par.gesso)
        fos = regras.p2o5(r["p"], par.fosforo) if not aj.bloquear_p else regras.Resultado(
            None, "não calculado", ["P2O5: P por Mehlich-1 — regra calibrada para P resina (dose não calculada)"])
        pot = regras.kcl(r["k"], par.potassio)
        k_ctc = r["k"] if (par.calcario.usar_k_medido and pd.notna(r["k"])) else par.calcario.k_ctc
        base.update({
            "_cal": cal.dose, "_ges": ges.dose, "_p": fos.dose, "_k": pot.dose,
            "_obs": (["Argila estimada pela CTC (laudo sem granulometria)"] if r.get("argila_estimada") else [])
                    + cal.obs + ges.obs + fos.obs + pot.obs,
            "Argila - origem": "estimada (CTC)" if r.get("argila_estimada") else "laudo",
            "CTC*": r["ca"] + r["mg"] + r["hal"] + k_ctc,
            "Calcário - equação": cal.bruto, "Calcário - critério": cal.criterio,
            "Gesso - equação": ges.bruto, "Gesso - critério": ges.criterio,
            "P2O5 - equação": fos.bruto, "P2O5 - critério": fos.criterio,
            "KCl - equação": pot.bruto, "KCl - critério": pot.criterio,
        })
        linhas.append(base)

    res = pd.DataFrame(linhas)
    for c in ("_cal", "_ges", "_p", "_k"):
        if c not in res:
            res[c] = float("nan")
        res[c] = pd.to_numeric(res[c], errors="coerce")
    if "_obs" not in res:
        res["_obs"] = None
    res["_obs"] = res["_obs"].apply(lambda v: list(v) if isinstance(v, list) else [])

    resumo: dict[str, dict] = {}

    def aplicar(col_base: str, col_final: str, ajuste: AjusteProduto, minimo, maximo, passo, nome):
        final, fator, media = _escalar(res[col_base], ajuste)
        res[col_final] = final.apply(lambda v: regras.arredondar(v, passo) if pd.notna(v) else None)
        res[f"{nome} - fator de ajuste"] = fator
        resumo[nome] = {"coluna": col_final, "media_regra": media, "fator": fator,
                        "media_final": res[col_final].mean(), "ajuste": ajuste}
        if ajuste.ativo():
            for i, v in final.items():
                if pd.isna(v):
                    continue
                if minimo is not None and v < minimo - 1e-9:
                    res.at[i, "_obs"].append(f"{nome} ajustado abaixo do mínimo usual ({minimo:.0f})")
                elif maximo is not None and v > maximo + 1e-9:
                    res.at[i, "_obs"].append(f"{nome} ajustado acima do máximo usual ({maximo:.0f})")

    aplicar("_cal", COL_CAL, aj.calcario, par.calcario.dose_min,
            None if any(abertura.values()) else par.calcario.dose_max, arr.calcario, "Calcário")

    if aj.gesso_por_s_elementar:
        res["_s0"] = res["_ges"].apply(lambda g: regras.s_elementar(g, par.enxofre.coef, par.enxofre.expoente))
        res["Gesso substituído (kg/ha)"] = res["_ges"].apply(
            lambda v: regras.arredondar(v, arr.gesso) if pd.notna(v) else None)
        aplicar("_s0", COL_S0, aj.gesso, None, None, arr.s_elementar, "S elementar")
    else:
        aplicar("_ges", COL_GES, aj.gesso, par.gesso.dose_min, par.gesso.dose_max, arr.gesso, "Gesso")

    aplicar("_p", COL_P, aj.p2o5, par.fosforo.dose_min, par.fosforo.dose_max, arr.p2o5, "P2O5")

    # formulação de P: dose do produto e, se tiver K2O, desconto do KCl (evita potássio em excesso)
    nome_p = nome_produto_p(aj.formula_p)
    if nome_p:
        n, p2o5, k2o = regras.ler_formula(aj.formula_p)
        dose_prod = res[COL_P] / (p2o5 / 100)
        res[nome_p] = dose_prod.apply(lambda v: regras.arredondar(v, arr.produto_p) if pd.notna(v) else None)
        prod_final = pd.to_numeric(res[nome_p], errors="coerce")
        if n > 0:
            res["N via produto P (kg/ha)"] = (prod_final * n / 100).round(1)
        if k2o > 0:
            res["K2O via produto P (kg/ha)"] = (prod_final * k2o / 100).round(1)
            equiv = regras.kcl_equivalente(prod_final, k2o, par.potassio.k2o_kcl)
            res["KCl pela regra (kg/ha)"] = res["_k"].apply(
                lambda v: regras.arredondar(v, arr.kcl) if pd.notna(v) else None)
            res["KCl equivalente descontado (kg/ha)"] = equiv.round(1)
            liquido = (res["_k"] - equiv.fillna(0)).clip(lower=0)
            res["_k"] = liquido.where(res["_k"].notna())
            for i in res.index[res["_k"].notna()]:
                if res.at[i, "_k"] <= 0:
                    res.at[i, "_obs"].append(f"K suprido pela formulação {int(n):02d}-{int(p2o5):02d}-{int(k2o):02d} "
                                             "(KCl = 0)")
                else:
                    res.at[i, "_obs"].append(f"KCl descontado: formulação fornece {prod_final[i] * k2o / 100:.0f} "
                                             f"kg/ha de K2O (= {equiv[i]:.0f} kg/ha de KCl)")
        resumo["Produto P"] = {"coluna": nome_p, "formula": regras.ler_formula(aj.formula_p),
                               "media_final": res[nome_p].mean()}

    # correção de P: dose média informada pelo usuário, além da manutenção (coluna própria)
    alvo_pc = aj.correcao_p2o5()
    if alvo_pc > 0:
        sup = ~res[COL_SUB].astype(bool)
        peso = pd.to_numeric(res.get("P2O5 - equação"), errors="coerce").where(sup)
        variavel = aj.p_correcao_variavel and not aj.bloquear_p and peso.notna().any()
        if variavel:
            # mesmo formato da curva de manutenção (sem os limites): mais onde o P do solo é menor
            peso = peso.fillna(peso.max()).where(sup)
            d_pc = alvo_pc * peso / peso.mean()
        else:
            d_pc = pd.Series(alvo_pc, index=res.index, dtype=float).where(sup)
        res[COL_PC] = d_pc.apply(lambda v: regras.arredondar(v, arr.p2o5) if pd.notna(v) else None)
        resumo["P correção"] = {"coluna": COL_PC, "media_final": pd.to_numeric(res[COL_PC]).mean(),
                                "variavel": bool(variavel), "alvo_p2o5": alvo_pc}
        nome_pc = nome_correcao_p(aj.p_correcao_formula)
        if nome_pc:
            n_c, p_c, k_c = regras.ler_formula(aj.p_correcao_formula)
            res[nome_pc] = (d_pc / (p_c / 100)).apply(
                lambda v: regras.arredondar(v, arr.produto_p) if pd.notna(v) else None)
            resumo["P correção"].update({"coluna_produto": nome_pc, "formula": (n_c, p_c, k_c),
                                         "media_produto": pd.to_numeric(res[nome_pc]).mean()})
            if k_c > 0:
                for i in res.index[sup]:
                    res.at[i, "_obs"].append(f"Correção de P com {int(n_c):02d}-{int(p_c):02d}-{int(k_c):02d}: o K2O "
                                             "desse produto NÃO foi descontado do KCl")

    minimo_k = par.potassio.dose_min if not (nome_p and regras.ler_formula(aj.formula_p)[2] > 0) else None
    aplicar("_k", COL_K, aj.kcl, minimo_k, par.potassio.dose_max, arr.kcl, "KCl")

    if aj.kcl_parcelado:
        pct1 = min(max(float(aj.kcl_pct_1), 0.0), 100.0)
        # 1ª parcela arredondada no mesmo passo do KCl; a 2ª é o restante → soma = dose total
        res[COL_K1] = res[COL_K].apply(
            lambda v: regras.arredondar(v * pct1 / 100, arr.kcl) if pd.notna(v) else None)
        res[COL_K2] = res[COL_K] - res[COL_K1]
        resumo["KCl parcelado"] = {"coluna": COL_K1, "pct_1": pct1, "pct_2": 100 - pct1,
                                   "media_final": res[COL_K].mean()}

    def obs_final(row):
        partes = list(row["_obs"])
        if isinstance(row.get("Observações"), str) and row["Observações"]:
            partes.insert(0, row["Observações"])
        return "; ".join(partes)

    res["Observações"] = res.apply(obs_final, axis=1)
    res = res.drop(columns=[c for c in res.columns if c.startswith("_")])
    res.attrs["ajustes"] = resumo
    return res


def resumo_por_talhao(res: pd.DataFrame) -> pd.DataFrame:
    rec = res[~res[COL_SUB]]
    doses = colunas_doses(rec)
    g = rec.groupby(["Talhão", "Abertura"], sort=False)
    out = g[doses].agg(["mean", "min", "max"]).round(0)
    out.columns = [f"{c} - {s}".replace("mean", "média").replace("min", "mín").replace("max", "máx")
                   for c, s in out.columns]
    out.insert(0, "Amostras", g.size())
    return out.reset_index()
