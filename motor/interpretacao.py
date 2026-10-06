"""Interpretação dos laudos por talhão: médias, classes de teores e gráficos.

As faixas seguem a tabela de legendas da equipe (8 classes). Cada atributo tem
intervalos [mín, máx); valores abaixo da menor faixa ou acima da maior recebem
a classe da faixa extrema. Para H+Al, Al e m% a escala é invertida (menor é melhor).
"""
from __future__ import annotations

import threading
import io

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .leitor import Laudo  # noqa: E402

CLASSES = ["Muito alto", "Excelente", "Muito bom", "Bom", "Regular", "Baixo", "Muito baixo", "Crítico"]
CORES = {  # mesma paleta dos mapas do book (vermelho escuro → azul escuro)
    "Muito alto": "#121D5E", "Excelente": "#0A6FF2", "Muito bom": "#008056", "Bom": "#1FD12A",
    "Regular": "#F2EF05", "Baixo": "#FF8F01", "Muito baixo": "#F20000", "Crítico": "#850000",
}


def cor_texto(fundo: str) -> str:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(fundo)
    return "#1A1A1A" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "white"


TEXTO_CLARO = {c for c, h in CORES.items() if cor_texto(h) == "white"}
ATENCAO = {"Baixo", "Muito baixo", "Crítico"}


def _faixas(*lims):
    """Recebe 8 pares (mín, máx) na ordem Muito alto → Crítico."""
    return [(c, lo, hi) for c, (lo, hi) in zip(CLASSES, zip(lims[::2], lims[1::2]))]


# chave: (rótulo, unidade, faixas)
# o pyplot não é seguro entre threads (sessões do Streamlit e downloads rodam em threads próprias)
TRAVA_MPL = threading.RLock()

ATRIBUTOS: dict[str, tuple[str, str, list]] = {
    "ph": ("pH CaCl2", "", _faixas(5.9, 6.4, 5.6, 5.9, 5.3, 5.6, 5.0, 5.3, 4.7, 5.0, 4.4, 4.7, 4.1, 4.4, 3.8, 4.1)),
    "mo": ("M.O.", "g/dm³", _faixas(40, 45, 35, 40, 30, 35, 25, 30, 20, 25, 15, 20, 10, 15, 5, 10)),
    "hal": ("H+Al", "mmolc/dm³", _faixas(10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 90)),
    "al": ("Al", "mmolc/dm³", _faixas(0, 0.3, 0.3, 0.6, 0.6, 1, 1, 1.4, 1.4, 1.9, 1.9, 2.4, 2.4, 3, 3, 3.6)),
    # na planilha original "Excelente" aparece como 108–205; corrigido para 180–205 (sequência das faixas)
    "ctc": ("CTC", "mmolc/dm³", _faixas(205, 230, 180, 205, 155, 180, 130, 155, 105, 130, 80, 105, 55, 80, 30, 55)),
    "v": ("V%", "%", _faixas(74, 81, 67, 74, 60, 67, 53, 60, 46, 53, 39, 46, 32, 39, 25, 32)),
    "m": ("m%", "%", _faixas(0, 2.5, 2.5, 5, 5, 7.5, 7.5, 10, 10, 12.5, 12.5, 15, 15, 17.5, 17.5, 20)),
    "ca": ("Ca", "mmolc/dm³", _faixas(56, 66, 46, 56, 38, 46, 30, 38, 22.5, 30, 15, 22.5, 7.5, 15, 0, 7.5)),
    "sat_ca": ("Ca na CTC", "%", _faixas(60.5, 70, 54, 60.5, 47.5, 54, 41, 47.5, 34.5, 41, 28, 34.5, 21.5, 28, 15, 21.5)),
    "mg": ("Mg", "mmolc/dm³", _faixas(21, 25, 18, 21, 15, 18, 12, 15, 9, 12, 6, 9, 3, 6, 0, 3)),
    "sat_mg": ("Mg na CTC", "%", _faixas(21, 24, 18, 21, 15, 18, 12, 15, 9, 12, 6, 9, 3, 6, 0, 3)),
    "k": ("K", "mmolc/dm³", _faixas(4.2, 4.8, 3.6, 4.2, 3, 3.6, 2.4, 3, 1.8, 2.4, 1.2, 1.8, 0.6, 1.2, 0, 0.6)),
    "sat_k": ("K na CTC", "%", _faixas(5, 6, 4, 5, 3.3, 4, 2.6, 3.3, 1.9, 2.6, 1.2, 1.9, 0.6, 1.2, 0, 0.6)),
    "prem": ("P-rem", "mg/L", _faixas(52.5, 60, 45, 52.5, 37.5, 45, 30, 37.5, 22.5, 30, 15, 22.5, 7.5, 15, 0, 7.5)),
    "p": ("P resina", "mg/dm³", _faixas(52, 70, 42, 52, 34, 42, 26, 34, 16, 26, 12, 16, 6, 12, 0, 6)),
    "s": ("S-SO4", "mg/dm³", _faixas(36, 41, 31, 36, 26, 31, 21, 26, 16, 21, 11, 16, 6, 11, 0, 6)),
    "b": ("B", "mg/dm³", _faixas(0.7, 0.8, 0.6, 0.7, 0.5, 0.6, 0.4, 0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2, 0, 0.1)),
    "zn": ("Zn", "mg/dm³", _faixas(3.5, 4, 3, 3.5, 2.5, 3, 2, 2.5, 1.5, 2, 1, 1.5, 0.5, 1, 0, 0.5)),
    "cu": ("Cu", "mg/dm³", _faixas(3.5, 4, 3, 3.5, 2.5, 3, 2, 2.5, 1.5, 2, 1, 1.5, 0.5, 1, 0, 0.5)),
    "fe": ("Fe", "mg/dm³", _faixas(50, 70, 30, 50, 20, 30, 10, 20, 6, 10, 4, 6, 2, 4, 0, 2)),
    "mn": ("Mn", "mg/dm³", _faixas(21, 28, 14, 21, 10, 14, 6, 10, 4.5, 6, 3, 4.5, 1.5, 3, 0, 1.5)),
}
INVERTIDOS = {"hal", "al", "m"}


def posicao_invertida(p: float) -> float:
    """Posição 0–8 na escala de condição para H+Al, Al e m% (menor é melhor): as duas faixas mais baixas de
    valor ocupam juntas a faixa "Excelente" (6–7); nunca chegam a "Muito alto"."""
    q = 8 - p
    return 6 + (q - 6) / 2 if q > 6 else q


def classificar(chave: str, valor) -> str | None:
    if valor is None or pd.isna(valor) or chave not in ATRIBUTOS:
        return None
    faixas = sorted(ATRIBUTOS[chave][2], key=lambda t: t[1])
    cl = faixas[0][0] if valor < faixas[0][1] else faixas[-1][0]
    for c, lo, hi in faixas:
        if lo <= valor < hi:
            cl = c
            break
    # H+Al, Al e m%: quanto menor, melhor — a faixa mais baixa é a melhor condição possível ("Excelente"),
    # nunca "Muito alto" (que nos outros atributos indica excesso)
    return "Excelente" if chave in INVERTIDOS and cl == "Muito alto" else cl


def textura(argila) -> str:
    """Classes texturais (Embrapa) pelo teor de argila em g/kg."""
    if argila is None or pd.isna(argila):
        return ""
    if argila < 150:
        return "Arenosa"
    if argila < 350:
        return "Média"
    if argila < 600:
        return "Argilosa"
    return "Muito argilosa"


def completar_derivados(d: pd.DataFrame) -> pd.DataFrame:
    """Calcula SB, CTC, V%, m% e saturações quando o laudo não traz a coluna."""
    d = d.copy()
    k = d["k"].fillna(0)
    sb_calc = d["ca"] + d["mg"] + k
    d["sb"] = d["sb"].fillna(sb_calc)
    d["ctc"] = d["ctc"].fillna(d["sb"] + d["hal"])
    d["v"] = d["v"].fillna(100 * d["sb"] / d["ctc"])
    d["m"] = d["m"].fillna(100 * d["al"].fillna(0) / (d["sb"] + d["al"].fillna(0)))
    d["sat_ca"] = 100 * d["ca"] / d["ctc"]
    d["sat_mg"] = 100 * d["mg"] / d["ctc"]
    d["sat_k"] = 100 * d["k"] / d["ctc"]
    return d


def medias_por_talhao(laudo: Laudo, subsuperficial: bool = False) -> pd.DataFrame:
    """Uma linha por talhão com a média de cada atributo (só atributos presentes no laudo)."""
    d = completar_derivados(laudo.dados)
    d = d[d["subsuperficial"] == subsuperficial]
    if d.empty:
        return pd.DataFrame()
    chaves = [k for k in ATRIBUTOS if d[k].notna().any()]
    g = d.groupby("talhao", sort=False)
    out = g[chaves + ["argila"]].mean()
    out.insert(0, "Amostras", g.size())
    out["Textura"] = out["argila"].apply(textura)
    if "argila_estimada" in d and d["argila_estimada"].any():
        est = g["argila_estimada"].mean() > 0.5
        out.loc[est, "Textura"] = out.loc[est, "Textura"] + " (est.)"
    out.index = [f"TH {int(float(i)):02d}" if str(i).replace(".0", "").isdigit() else i for i in out.index]
    out.index.name = "Talhão"
    return out.reset_index()


def classes_de(medias: pd.DataFrame) -> pd.DataFrame:
    """Mesmo formato de `medias`, com o nome da classe no lugar do valor."""
    cl = medias.copy()
    for k in ATRIBUTOS:
        if k in cl:
            cl[k] = cl[k].apply(lambda v, k=k: classificar(k, v))
    return cl


def pontos_de_atencao(medias: pd.DataFrame) -> dict[str, list[str]]:
    """Talhão → lista de 'atributo (classe)' em Baixo/Muito baixo/Crítico."""
    cl = classes_de(medias)
    out = {}
    for _, r in cl.iterrows():
        itens = [f"{ATRIBUTOS[k][0]} ({r[k].lower()})" for k in ATRIBUTOS if k in cl and r[k] in ATENCAO]
        out[r["Talhão"]] = itens
    return out


def formato(chave: str, v) -> str:
    if v is None or pd.isna(v):
        return "–"
    casas = 2 if chave in ("b", "k", "al") else (1 if v < 100 else 0)
    return f"{v:.{casas}f}".replace(".", ",")


# ------------------------------------------------------------------ gráficos
def figura_mapa(medias: pd.DataFrame, titulo: str = "") -> plt.Figure:
    """Mapa de classes: talhões × atributos, célula colorida pela classe e com a média."""
    chaves = [k for k in ATRIBUTOS if k in medias]
    n_l, n_c = len(medias), len(chaves)
    fig_w = max(9, 1.6 + 0.62 * n_c)
    fig_h = 1.9 + 0.42 * n_l
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    for i, (_, r) in enumerate(medias.iterrows()):
        for j, k in enumerate(chaves):
            cl = classificar(k, r[k])
            cor = CORES.get(cl, "#EEEEEE")
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, color=cor, lw=0))
            ax.text(j + 0.5, i + 0.52, formato(k, r[k]), ha="center", va="center", fontsize=7.5,
                    color="white" if cl in TEXTO_CLARO else "#1a1a1a", fontweight="bold")
    ax.set_xlim(0, n_c)
    ax.set_ylim(n_l, 0)
    ax.set_xticks(np.arange(n_c) + 0.5)
    ax.set_xticklabels([ATRIBUTOS[k][0] + (" ↓" if k in INVERTIDOS else "") for k in chaves],
                       fontsize=7.5, rotation=0)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_l) + 0.5)
    ax.set_yticklabels([f"{t}  (n={n})" for t, n in zip(medias["Talhão"], medias["Amostras"])], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    leg = [Patch(color=CORES[c], label=c) for c in CLASSES]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=8, fontsize=7.5,
              frameon=False, handlelength=1.2, columnspacing=1.2,
              title="Classes (cor = condição; número = média do talhão) · ↓ = menor valor é melhor",
              title_fontsize=7.5)
    if titulo:
        fig.suptitle(titulo, fontsize=11, fontweight="bold", color="#104A2A", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def figura_doses(res: pd.DataFrame, colunas: list[str]) -> plt.Figure:
    """Dose média recomendada por talhão — um painel por produto."""
    rec = res[~res["Camada subsuperficial"]]
    g = rec.groupby("Talhão", sort=False)[colunas].mean()
    g.index = [f"TH {int(float(i)):02d}" if str(i).replace(".0", "").isdigit() else i for i in g.index]
    n = len(colunas)
    fig, axes = plt.subplots(1, n, figsize=(max(9, 2.7 * n), 0.9 + 0.38 * len(g)), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, colunas):
        v = g[c]
        ax.barh(range(len(v)), v.values, color="#104A2A", height=0.62)
        for i, x in enumerate(v.values):
            if pd.notna(x):
                ax.text(x, i, f" {x:,.0f}".replace(",", "."), va="center", fontsize=7.5, color="#333333")
        ax.set_title(c.replace(" (kg/ha)", "\n(kg/ha, média)"), fontsize=8.5, color="#104A2A", fontweight="bold")
        ax.set_xlim(0, (np.nanmax(v.values) if len(v) else 1) * 1.28)
        ax.set_yticks(range(len(v)))
        ax.set_yticklabels(v.index, fontsize=8)
        ax.tick_params(axis="x", labelsize=7, colors="#777777")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
        ax.grid(axis="x", color="#E6E6E6", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#BBBBBB")
    axes[0].invert_yaxis()   # eixo y compartilhado: inverter uma vez só (1º talhão no topo)
    fig.tight_layout()
    return fig


def figura_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------ gráficos adicionais
NOMES_CURTOS = {"ph": "pH", "mo": "Matéria orgânica", "ctc": "CTC", "v": "Saturação por bases",
                "hal": "Acidez potencial", "al": "Acidez trocável", "m": "Saturação por Al", "ca": "Cálcio",
                "sat_ca": "Ca na CTC", "mg": "Magnésio", "sat_mg": "Mg na CTC", "k": "Potássio",
                "sat_k": "K na CTC", "prem": "P remanescente", "p": "Fósforo", "s": "Enxofre", "b": "Boro",
                "zn": "Zinco", "cu": "Cobre", "fe": "Ferro", "mn": "Manganês"}
ORDEM = ["ph", "mo", "ctc", "v", "hal", "al", "m", "ca", "sat_ca", "mg", "sat_mg", "k", "sat_k",
         "p", "s", "b", "zn", "mn", "cu", "fe"]


def pos_condicao(chave, v):
    """Posição 0–8 na escala de condição (0 = crítico … 8 = muito alto)."""
    faixas = sorted(ATRIBUTOS[chave][2], key=lambda t: t[1])
    lim = [faixas[0][1]] + [f[2] for f in faixas]
    p = float(np.interp(v, lim, np.arange(len(lim))))
    return posicao_invertida(p) if chave in INVERTIDOS else p


def _estilo(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def figura_panorama(medias: pd.DataFrame) -> plt.Figure:
    """Média de cada talhão marcada sobre a faixa de 8 classes de cada atributo."""
    chaves = [k for k in ORDEM if k in medias]
    n, nt = len(chaves), len(medias)
    fig, ax = plt.subplots(figsize=(9, 0.34 * n + 1.4), dpi=150)
    ordem = CLASSES[::-1]
    for i, k in enumerate(chaves):
        for j, cl in enumerate(ordem):
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.3), 0.96, 0.4, color=CORES[cl], alpha=0.85, lw=0))
    marc = ["o", "^", "s", "D", "v", "P", "X", "*"]
    for t, (_, r) in enumerate(medias.iterrows()):
        ys, xs = [], []
        for i, k in enumerate(chaves):
            if pd.notna(r[k]):
                xs.append(np.clip(pos_condicao(k, r[k]), 0.08, 7.92))
                ys.append(i + 0.5 + (t - (nt - 1) / 2) * min(0.16, 0.5 / max(nt, 1)))
        ax.scatter(xs, ys, marker=marc[t % len(marc)], s=38, color="white", edgecolor="#111111", lw=1.1,
                   zorder=5, label=r["Talhão"])
    ax.set_xlim(0, 8)
    ax.set_ylim(n, -0.2)
    ax.set_yticks(np.arange(n) + 0.5, [NOMES_CURTOS[k] + (" ↓" if k in INVERTIDOS else "") for k in chaves],
                  fontsize=8)
    ax.set_xticks(np.arange(8) + 0.5, [c.replace("Muito ", "M. ") for c in ordem], fontsize=7.5)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=min(nt, 6), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def figura_classes_amostras(laudo, subsuperficial: bool = False) -> plt.Figure:
    """Participação das amostras em cada classe, por atributo (100% empilhado)."""
    d = completar_derivados(laudo.dados)
    d = d[d["subsuperficial"] == subsuperficial]
    chaves = [k for k in ORDEM if k in d and d[k].notna().any()]
    ordem = CLASSES[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.33 * len(chaves) + 1.2), dpi=150)
    for i, k in enumerate(chaves):
        cls = [classificar(k, v) for v in d[k].dropna()]
        x = 0.0
        for cl in ordem:
            pct = 100 * cls.count(cl) / len(cls) if cls else 0
            if pct > 0:
                ax.barh(i, pct, left=x, color=CORES[cl], height=0.66, lw=0)
                if pct >= 9:
                    ax.text(x + pct / 2, i, f"{pct:.0f}%", ha="center", va="center", fontsize=6.8,
                            color=cor_texto(CORES[cl]))
                x += pct
    ax.set_yticks(range(len(chaves)), [NOMES_CURTOS[k] for k in chaves], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% das amostras", fontsize=8)
    ax.tick_params(labelsize=7.5, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    leg = [Patch(color=CORES[c], label=c) for c in ordem]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=8, fontsize=7, frameon=False,
              handlelength=1.1, columnspacing=1.0)
    fig.tight_layout()
    return fig


def figura_bases(medias: pd.DataFrame, alvo_ca=55.0, alvo_mg=19.4) -> plt.Figure:
    """Composição da CTC (Ca, Mg, K, H+Al) por talhão, com as metas da calagem e as relações entre bases."""
    tal = list(medias["Talhão"])
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(10, 0.55 * len(tal) + 1.7), dpi=150,
                                  gridspec_kw={"width_ratios": [3, 1.35]})
    cores = {"Ca": "#104A2A", "Mg": "#7FA88C", "K": "#E25A10", "H+Al": "#C4C4C4"}
    linhas = []
    for i, (_, r) in enumerate(medias.iterrows()):
        tot = r["ca"] + r["mg"] + r["k"] + r["hal"]
        x = 0
        for nome, v in (("Ca", r["ca"]), ("Mg", r["mg"]), ("K", r["k"]), ("H+Al", r["hal"])):
            p = 100 * v / tot
            ax.barh(i, p, left=x, color=cores[nome], height=0.6, edgecolor="white", lw=1, label=nome if i == 0 else None)
            if p >= 5:
                ax.text(x + p / 2, i, f"{p:.0f}%", ha="center", va="center", fontsize=7.5, color=cor_texto(cores[nome]))
            x += p
        linhas.append([r["Talhão"], f"{r['ca'] / r['mg']:.1f}", f"{r['ca'] / r['k']:.1f}", f"{r['mg'] / r['k']:.1f}"])
    for alvo, rot in ((alvo_ca, f"meta Ca {alvo_ca:.0f}%"), (alvo_ca + alvo_mg, f"meta Ca+Mg {alvo_ca + alvo_mg:.0f}%")):
        ax.axvline(alvo, color="#111111", lw=0.9, ls=(0, (3, 2)))
        ax.text(alvo, -0.62, rot, ha="center", va="bottom", fontsize=7)
    ax.set_yticks(range(len(tal)), tal, fontsize=8.5)
    ax.set_ylim(len(tal) - 0.5, -0.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% da CTC (Ca + Mg + K + H+Al)", fontsize=8)
    ax.tick_params(labelsize=7.5, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7.5, frameon=False)
    axt.axis("off")
    tb = axt.table(cellText=linhas, colLabels=["Talhão", "Ca/Mg", "Ca/K", "Mg/K"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    for (r_, c_), cel in tb.get_celld().items():
        cel.set_edgecolor("#DDDDDD")
        if r_ == 0:
            cel.set_facecolor("#104A2A")
            cel.set_text_props(color="white")
    fig.tight_layout()
    return fig
