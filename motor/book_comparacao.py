"""Book comparativo: dois laudos da mesma área (ano anterior × ano recente), no mesmo estilo do book.

Páginas de atributo com os dois mapas lado a lado (ou um sobre o outro, conforme o formato da área), mesma
legenda, áreas por classe de cada ano e a média de cada ano. Prescrições: só as do laudo mais recente.
"""
from __future__ import annotations

import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon as MPoly, Rectangle  # noqa: E402

from . import book as bk  # noqa: E402
from . import interpretacao as it  # noqa: E402
from .book import (A4, COR_CLASSE, CURTOS, F_BOLD, F_SEMI, F_XB, NOMES, ORDEM_FERT, TITULOS, br,  # noqa: E402
                   casas_de, cor_texto, curto, exibir)
from .comparacao import Comparacao, media_area, tabela_evolucao  # noqa: E402

CHAVES_VARIACAO = ["ph", "v", "ca", "mg", "k", "p", "s", "mo", "hal", "ctc"]
NOME_TAB = {"ph": "pH", "mo": "Matéria orgânica", "ctc": "CTC (pH 7)", "v": "Saturação por bases (V%)",
            "hal": "Acidez potencial (H+Al)", "al": "Alumínio (Al³⁺)", "m": "Saturação por Al (m%)",
            "ca": "Cálcio (Ca²⁺)", "sat_ca": "Ca na CTC", "mg": "Magnésio (Mg²⁺)", "sat_mg": "Mg na CTC",
            "k": "Potássio (K⁺)", "sat_k": "K na CTC", "p": "Fósforo (P)", "s": "Enxofre (S-SO₄)", "b": "Boro",
            "zn": "Zinco", "mn": "Manganês", "cu": "Cobre", "fe": "Ferro"}
VERDE_BOM, VERMELHO_RUIM, CINZA = "#1A9850", "#D73027", "#555555"
CMAP_DIF = LinearSegmentedColormap.from_list("dif", ["#A50026", "#F46D43", "#FEE08B", "#F7F7F7", "#D9EF8B",
                                                     "#66BD63", "#006837"])


# ------------------------------------------------------------------ utilidades
def _texto(fig, x, y, txt, larg=100, fs=8, cor="#333333", entre=0.015, **kw):
    for linha in textwrap.wrap(str(txt), larg):
        fig.text(x, y, linha, fontsize=fs, color=cor, va="top", **kw)
        y -= entre
    return y


def _escala_cor(chave, todos):
    if chave in it.ATRIBUTOS:
        limites, cores, nomes = bk.escala_classes(chave)
    else:
        limites, cores, nomes = bk.escala_intervalos(todos, chave)
    return limites, cores, nomes


def _valores(mapas, chave):
    return np.concatenate([exibir(chave, m.atributos[chave][m.grade.mascara]) for m in mapas])


def _areas(mapas, chave, limites, n):
    vals = _valores(mapas, chave)
    pix = np.concatenate([np.full(m.grade.mascara.sum(), m.grade.area_pixel_ha) for m in mapas])
    ok = np.isfinite(vals)
    cls = np.clip(np.searchsorted(limites, vals[ok], side="right") - 1, 0, n - 1)
    return np.array([pix[ok][cls == i].sum() for i in range(n)])


def _desenha(ax, mapas, chave, limites, cores, discreto, lim, rotulos=True):
    cmap = ListedColormap(cores) if discreto else bk._cmap_posicional(cores)
    larg_fig = ax.get_position().width * A4[0]
    for m in mapas:
        px_talhao = larg_fig * 200 * (m.grade.x[-1] - m.grade.x[0]) / (lim[2] - lim[0])
        fator = int(np.clip(round(px_talhao / max(m.grade.mascara.shape[1], 1)), 1, 4 if discreto else 3))
        pp = bk._patch(m.talhao.perimetro, fc="none", ec="none")
        ax.add_artist(pp)
        if discreto:
            if not bk.faixas_solidas(ax, m, exibir(chave, m.atributos[chave]), limites, cores, pp):
                continue
        else:
            Zs, ext = bk._raster_suave(m, exibir(chave, m.atributos[chave]), fator=fator)
            if ext is None:
                continue
            im = ax.imshow(bk._posicao(Zs, limites), extent=ext, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear",
                           zorder=2)
            im.set_clip_path(pp)
        ax.add_artist(bk._patch(m.talhao.perimetro, fc="none", ec="#111111", lw=0.7, zorder=5))
        if rotulos and len(mapas) > 1:
            c = m.talhao.perimetro.representative_point()
            ax.text(c.x, c.y, curto(m.talhao.nome), ha="center", va="center", fontsize=6 if len(mapas) > 6 else 7,
                    family=F_BOLD, weight="bold", color="#111111", zorder=6,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])


def arranjo(lim, topo=0.825, base=0.395):
    """Posição dos dois mapas: lado a lado (áreas mais altas que largas) ou um sobre o outro (áreas largas).

    Escolhe o arranjo em que o mapa fica maior. Devolve (modo, [caixa A, caixa B])."""
    w, h = lim[2] - lim[0], lim[3] - lim[1]
    alt = topo - base
    lado = [[0.095, base, 0.395, alt - 0.02], [0.51, base, 0.395, alt - 0.02]]
    meio = alt / 2
    emp = [[0.095, base + meio + 0.012, 0.81, meio - 0.034], [0.095, base, 0.81, meio - 0.034]]

    def escala(c):
        return min(c[2] * A4[0] / w, c[3] * A4[1] / h)
    return ("lado", lado) if escala(lado[0]) >= escala(emp[0]) else ("empilhado", emp)


def _cabecalho(fig, ctx, x, y, ano, media, unid, cl, casas, ha="center"):
    """'2024 · média 5,2 mg/dm³ [Bom]' acima de cada mapa."""
    cor = ctx.marca["cor"]
    u = "" if unid in ("", "adimensional") else f" {unid}"
    txt = f"{ano}  ·  média {br(media, casas)}{u}"
    t = fig.text(x, y, txt, ha=ha, va="center", fontsize=9.5, family=F_BOLD, weight="bold", color=cor)
    if cl:
        r = fig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.012, y, cl, ha="left", va="center", fontsize=7.5, family=F_SEMI,
                 color=cor_texto(COR_CLASSE[cl]), bbox=dict(boxstyle="round,pad=0.3", fc=COR_CLASSE[cl], ec="none"))


def _rotulo_faixa(i, n, limites, casas, nomes):
    if nomes:
        return nomes[i]
    lo, hi = limites[i], limites[i + 1]
    if i == n - 1:
        return f"> {br(lo, casas)}"
    if i == 0:
        return f"< {br(hi, casas)}"
    return f"{br(lo, casas)} – {br(hi, casas)}"


def _tabela_classes(fig, ctx, x, y, limites, cores, nomes, areas_a, areas_b, rot, casas, discreto):
    """Tabela: faixa/classe | área e % em cada ano | variação em pontos percentuais (maior classe no topo)."""
    cor = ctx.marca["cor"]
    n = len(cores)
    ta, tb = areas_a.sum() or 1, areas_b.sum() or 1
    cols = [(x + 0.035, "Classe" if nomes else "Faixa", "left"), (x + 0.3, rot[0], "right"),
            (x + 0.44, rot[1], "right"), (x + 0.535, "Variação", "right")]
    for cx, txt, al in cols:
        fig.text(cx, y, txt, ha=al, va="center", fontsize=7.8, family=F_BOLD, weight="bold", color=cor)
    fig.add_artist(plt.Line2D([x, x + 0.54], [y - 0.01] * 2, transform=fig.transFigure, color="#BBBBBB", lw=0.6))
    passo = min(0.021, 0.2 / n)
    yy = y - 0.01
    for i in range(n - 1, -1, -1):
        yy -= passo
        fig.add_artist(Rectangle((x, yy - passo * 0.36), 0.024, passo * 0.72, transform=fig.transFigure, fc=cores[i],
                                 ec="#666666", lw=0.3))
        vazio = areas_a[i] <= 0 and areas_b[i] <= 0
        ct = "#AAAAAA" if vazio else "#222222"
        fig.text(cols[0][0], yy, _rotulo_faixa(i, n, limites, casas, nomes), va="center", fontsize=7.6, color=ct)
        pa, pb = 100 * areas_a[i] / ta, 100 * areas_b[i] / tb
        for (cx, _, _), ar, pc in ((cols[1], areas_a[i], pa), (cols[2], areas_b[i], pb)):
            fig.text(cx, yy, "–" if ar <= 0 else f"{br(ar, 1)} ha · {pc:.0f}%", ha="right", va="center",
                     fontsize=7.4, color="#AAAAAA" if ar <= 0 else "#333333")
        d = pb - pa
        if abs(d) >= 0.5:
            fig.text(cols[3][0], yy, f"{'+' if d > 0 else '−'}{abs(d):.0f} p.p.", ha="right", va="center",
                     fontsize=7.4, family=F_SEMI, color="#333333")
    return yy


def transicoes(pares, chave, limites, n):
    """Matriz n × n de área (ha): classe no ano anterior (linha) → classe no ano recente (coluna), pixel a pixel."""
    T = np.zeros((n, n))
    for a, b in pares:
        m = a.grade.mascara
        va, vb = exibir(chave, a.atributos[chave][m]), exibir(chave, b.atributos[chave][m])
        ok = np.isfinite(va) & np.isfinite(vb)
        ia = np.clip(np.searchsorted(limites, va[ok], side="right") - 1, 0, n - 1)
        ib = np.clip(np.searchsorted(limites, vb[ok], side="right") - 1, 0, n - 1)
        np.add.at(T, (ia, ib), a.grade.area_pixel_ha)
    return T


def _barras_distribuicao(fig, rect, cores, T, rot):
    """Duas barras 100% (anterior e recente) e as faixas de transição entre elas: quanto da área de cada classe
    no ano anterior foi para cada classe no ano recente (mesmo pixel nos dois mapas)."""
    ax = fig.add_axes(rect)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(0, 100)
    ax.axis("off")
    tot = T.sum() or 1
    P = 100 * T / tot
    pa, pb = P.sum(axis=1), P.sum(axis=0)
    ca, cb = np.concatenate([[0], np.cumsum(pa)]), np.concatenate([[0], np.cumsum(pb)])
    xa1, xb0 = 0.5, 1.5
    sai = ca[:-1].copy()                 # posição corrente dentro de cada classe de origem
    chega = cb[:-1].copy()
    for i in range(len(cores)):
        for j in range(len(cores)):
            f = P[i, j]
            if f <= 0.05:
                continue
            y0a, y0b = sai[i], chega[j]
            xs = np.linspace(xa1, xb0, 30)
            t = (1 - np.cos(np.pi * (xs - xa1) / (xb0 - xa1))) / 2          # curva suave
            baixo = y0a + (y0b - y0a) * t
            ax.fill_between(xs, baixo, baixo + f, color=cores[i], alpha=0.35, lw=0)
            sai[i] += f
            chega[j] += f
    for i, c in enumerate(cores):
        if pa[i] > 0:
            ax.add_patch(Rectangle((0, ca[i]), xa1, pa[i], fc=c, ec="white", lw=0.6))
        if pb[i] > 0:
            ax.add_patch(Rectangle((xb0, cb[i]), 2 - xb0, pb[i], fc=c, ec="white", lw=0.6))
    ax.text(xa1 / 2, -3, rot[0], ha="center", va="top", fontsize=8, family=F_SEMI, color="#333333")
    ax.text((xb0 + 2) / 2, -3, rot[1], ha="center", va="top", fontsize=8, family=F_SEMI, color="#333333")
    return ax


# ------------------------------------------------------------------ páginas
def pagina_atributo_comp(pdf, ctx, c: Comparacao, chave: str, num: int, pares=None, rotulo_grupo=""):
    pares = [(a, b) for a, b in (pares or c.pares) if chave in a.atributos and chave in b.atributos]
    ma, mb = [a for a, _ in pares], [b for _, b in pares]
    rot = c.rotulos
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    t, s_, unid = TITULOS.get(chave, (chave.upper(), "", ""))
    sub = f"{s_}  ·  {rot[0]} × {rot[1]}".strip(" ·")
    if rotulo_grupo:
        sub = f"{sub}  ·  {rotulo_grupo}"
    bk.titulo(fig, ctx, t, sub, tam=24 if len(t) < 22 else 20)
    va, vb = _valores(ma, chave), _valores(mb, chave)
    limites, cores, nomes = _escala_cor(chave, np.concatenate([va, vb]))
    discreto = ctx.dados.cores_discretas
    lim = bk._limites(mb)
    modo, caixas = arranjo(lim)
    axs = [bk._caixa_mapa(fig, ctx, cx, lim) for cx in caixas]
    for ax, ms in zip(axs, (ma, mb)):
        _desenha(ax, ms, chave, limites, cores, discreto, lim)
    med_a, med_b = float(np.nanmean(va)), float(np.nanmean(vb))
    casas = casas_de(chave, max(abs(med_a), abs(med_b)))
    for ax, ano, med in zip(axs, rot, (med_a, med_b)):
        bb = ax.get_position()
        cl = it.classificar(chave, med / bk.FATOR_EXIBICAO.get(chave, 1.0)) if chave in it.ATRIBUTOS else None
        if modo == "lado":
            _cabecalho(fig, ctx, bb.x0 + bb.width / 2 - (0.04 if cl else 0), bb.y1 + 0.016, ano, med, unid, cl, casas)
        else:
            _cabecalho(fig, ctx, 0.1, bb.y1 + 0.013, ano, med, unid, cl, casas, ha="left")
    bk.escala(fig, axs[1], 0.115, 0.372, frac=0.12, fs=6.8)
    bk.norte(fig, 0.87, 0.352, 0.032)
    # variação da média
    d = med_b - med_a
    y = 0.338
    u = "" if unid in ("", "adimensional") else f" {unid}"
    melhor = None
    if chave in it.ATRIBUTOS:
        from .comparacao import posicao
        f = bk.FATOR_EXIBICAO.get(chave, 1.0)
        melhor = posicao(chave, med_b / f) - posicao(chave, med_a / f)
    cor_d = VERDE_BOM if (melhor or 0) > 0.05 else VERMELHO_RUIM if (melhor or 0) < -0.05 else CINZA
    seta = "▲" if d > 0 else "▼" if d < 0 else "="
    pct = f" ({'+' if d >= 0 else '−'}{abs(100 * d / med_a):.0f}%)" if med_a else ""
    fig.text(0.115, y, f"Média: {br(med_a, casas)} → {br(med_b, casas)}{u}", fontsize=10.5, family=F_BOLD,
             weight="bold", color=ctx.marca["cor"], va="center")
    fig.text(0.6, y, f"{seta} {'+' if d >= 0 else '−'}{br(abs(d), casas)}{u}{pct}", fontsize=10, family=F_BOLD,
             weight="bold", color=cor_d, va="center")
    T = transicoes(pares, chave, limites, len(cores))
    aa, ab = T.sum(axis=1), T.sum(axis=0)
    _tabela_classes(fig, ctx, 0.11, 0.3, limites, cores, nomes, aa, ab, rot, casas_de(chave, limites[-1]), discreto)
    _barras_distribuicao(fig, [0.685, 0.108, 0.2, 0.19], cores, T, rot)
    fig.text(0.785, 0.305, "para onde foi a área", ha="center", fontsize=6.6, color="#777777")
    pdf.savefig(fig)
    plt.close(fig)


def pagina_evolucao(pdf, ctx, c: Comparacao, num: int):
    """Tabela-resumo: média de cada atributo nos dois anos, classe, variação e se a condição melhorou."""
    rot = c.rotulos
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "EVOLUÇÃO DA FERTILIDADE", f"médias da área · {rot[0]} × {rot[1]} · camada 0–20 cm", tam=21)
    cor = ctx.marca["cor"]
    chaves = [k for k in ORDEM_FERT if all(k in a.atributos and k in b.atributos for a, b in c.pares)]
    tab = tabela_evolucao(c, chaves)
    area = sum(b.talhao.area_ha for b in c.mapas_b)
    fig.text(0.5, 0.838, f"{len(c.pares)} talhão(ões) · {br(area, 1)} ha comparados", ha="center", fontsize=9,
             color=CINZA)
    x_nome, x_a, x_ca, x_b, x_cb, x_d, x_s = 0.11, 0.47, 0.485, 0.66, 0.675, 0.855, 0.868
    y = 0.8
    fig.add_artist(FancyBboxPatch((0.1, y - 0.013), 0.8, 0.028, boxstyle="round,pad=0,rounding_size=0.01",
                                  transform=fig.transFigure, fc=cor, ec="none"))
    for x, txt, al in ((x_nome, "Atributo", "left"), (x_nome + 0.235, "Unidade", "left"), (x_a, rot[0], "right"),
                       (x_b, rot[1], "right"), (x_d, "Variação", "right")):
        fig.text(x, y + 0.001, txt, ha=al, va="center", fontsize=8.4, family=F_BOLD, weight="bold", color="white")
    passo = min(0.03, 0.62 / max(len(tab), 1))
    for i, r in tab.reset_index(drop=True).iterrows():
        y -= passo
        if i % 2 == 0:
            fig.add_artist(Rectangle((0.1, y - passo / 2), 0.8, passo, transform=fig.transFigure, fc="#F4F6F4",
                                     ec="none", zorder=0))
        k = r["chave"]
        unid = it.ATRIBUTOS[k][1] if k in it.ATRIBUTOS else ""
        cs = casas_de(k, max(abs(r["A"]), abs(r["B"])))
        fig.text(x_nome, y, NOME_TAB.get(k, NOMES[k]), va="center", fontsize=8.2, color="#222222")
        fig.text(x_nome + 0.235, y, unid, va="center", fontsize=6.6, color="#777777")
        for x, xc, v, cl in ((x_a, x_ca, r["A"], r.get("classe A")), (x_b, x_cb, r["B"], r.get("classe B"))):
            fig.text(x, y, br(v, cs), ha="right", va="center", fontsize=8.4, family=F_SEMI, color="#222222")
            if isinstance(cl, str):
                fig.text(xc, y, cl, ha="left", va="center", fontsize=6.4, color=cor_texto(COR_CLASSE[cl]),
                         bbox=dict(boxstyle="round,pad=0.25", fc=COR_CLASSE[cl], ec="none"))
        m = r.get("melhora", 0) or 0
        cd = VERDE_BOM if m > 0.05 else VERMELHO_RUIM if m < -0.05 else CINZA
        d = r["Δ"]
        fig.text(x_d, y, f"{'+' if d >= 0 else '−'}{br(abs(d), cs)}", ha="right", va="center", fontsize=8.4,
                 family=F_BOLD, weight="bold", color=cd)
        fig.text(x_s, y, "▲" if m > 0.05 else "▼" if m < -0.05 else "•", ha="left", va="center", fontsize=8,
                 color=cd)
    y -= passo + 0.01
    _texto(fig, 0.11, y, "Verde (▲) = a condição melhorou na escala de interpretação; vermelho (▼) = piorou. Para "
                         "H+Al, Al³⁺ e m%, cair é melhorar; para pH, V%, Ca na CTC e B, passar de 'Excelente' para "
                         "'Muito alto' não conta como melhora (pode ser excesso). Médias ponderadas pela área (mapas na mesma grade nos dois "
                         "anos).", larg=112, fs=7.2, cor="#666666", entre=0.013)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_evolucao_talhoes(pdf, ctx, c: Comparacao, num: int, parte=0, por_pagina=22):
    """Matriz talhão × atributo: valor anterior → recente, com fundo pela melhora na escala de interpretação."""
    from .comparacao import posicao
    rot = c.rotulos
    chaves = [k for k in CHAVES_VARIACAO if all(k in a.atributos and k in b.atributos for a, b in c.pares)][:8]
    pares = c.pares[parte * por_pagina:(parte + 1) * por_pagina]
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "EVOLUÇÃO POR TALHÃO", f"média de cada talhão · {rot[0]} → {rot[1]}"
              + (f" · parte {parte + 1}" if parte else ""), tam=21)
    cor = ctx.marca["cor"]
    x0, larg_nome = 0.1, 0.12
    wc = (0.9 - x0 - larg_nome) / max(len(chaves), 1)
    y = 0.82
    for j, k in enumerate(chaves):
        fig.text(x0 + larg_nome + wc * (j + 0.5), y, CURTOS.get(k, k), ha="center", va="center", fontsize=8.4,
                 family=F_BOLD, weight="bold", color=cor)
    passo = min(0.058, 0.66 / max(len(pares), 1))
    duas = passo >= 0.042                           # poucos talhões: célula maior, com a variação embaixo
    for a, b in pares:
        y -= passo
        fig.text(x0 + 0.005, y, curto(b.talhao.nome), va="center", fontsize=8.6 if duas else 8.2, family=F_SEMI,
                 color="#222222")
        for j, k in enumerate(chaves):
            va, vb = float(np.nanmean(a.atributos[k])), float(np.nanmean(b.atributos[k]))
            m = posicao(k, vb) - posicao(k, va) if k in it.ATRIBUTOS else 0
            inten = min(abs(m) / 1.5, 1.0)
            fundo = bk._mistura("white", VERDE_BOM if m > 0 else VERMELHO_RUIM, 0.15 + 0.6 * inten) \
                if abs(m) > 0.05 else "#F2F2F2"
            cx = x0 + larg_nome + wc * j
            fig.add_artist(FancyBboxPatch((cx + 0.003, y - passo * 0.42), wc - 0.006, passo * 0.84,
                                          boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                          fc=fundo, ec="none"))
            cs = casas_de(k, max(abs(va), abs(vb)))
            fs = (7.6 if duas else 6.9) if wc > 0.085 else 6.2
            fig.text(cx + wc / 2, y + (0.007 if duas else 0), f"{br(va, cs)} → {br(vb, cs)}", ha="center",
                     va="center", fontsize=fs, color="#1A1A1A")
            if duas:
                d = vb - va
                fig.text(cx + wc / 2, y - 0.01, f"{'+' if d >= 0 else '−'}{br(abs(d), cs)}", ha="center", va="center",
                         fontsize=fs - 0.6, family=F_BOLD, weight="bold", color="#1A1A1A")
    y -= passo + 0.012
    for i, (txt, cfundo) in enumerate((("melhorou", bk._mistura("white", VERDE_BOM, 0.6)),
                                       ("piorou", bk._mistura("white", VERMELHO_RUIM, 0.6)),
                                       ("estável", "#F2F2F2"))):
        fig.add_artist(Rectangle((0.11 + i * 0.14, y - 0.006), 0.022, 0.012, transform=fig.transFigure, fc=cfundo,
                                 ec="none"))
        fig.text(0.137 + i * 0.14, y, txt, va="center", fontsize=7.6, color="#333333")
    fig.text(0.56, y, "intensidade = tamanho da mudança de classe", va="center", fontsize=7, color="#777777")
    pdf.savefig(fig)
    plt.close(fig)


def pagina_area_classes_comp(pdf, ctx, c: Comparacao, num: int):
    """Participação da área em cada classe: barra do ano anterior sobre a do ano recente, por atributo."""
    rot = c.rotulos
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "ÁREA POR CLASSE", f"participação da área em cada condição · {rot[0]} × {rot[1]}", tam=22)
    chaves = [k for k in ORDEM_FERT if k in it.ATRIBUTOS and all(k in a.atributos and k in b.atributos
                                                                   for a, b in c.pares)]
    ordem = it.CLASSES[::-1]
    ax = fig.add_axes([0.35, 0.16, 0.54, 0.66])
    n = len(chaves)
    ax.set_xlim(0, 100)
    ax.set_ylim(n, -0.1)
    ax.axis("off")
    for i, k in enumerate(chaves):
        limites, cores_k, _ = bk.escala_classes(k)
        for j, (ms, ano) in enumerate(((c.mapas_a, rot[0]), (c.mapas_b, rot[1]))):
            ar = _areas(ms, k, limites, len(cores_k))
            cont = bk.somar_por_condicao(k, ar)
            tot = ar.sum() or 1
            x = 0.0
            y0 = i + 0.1 + j * 0.4
            for cl in ordem:
                pct = 100 * cont.get(cl, 0) / tot
                if pct <= 0:
                    continue
                ax.add_artist(Rectangle((x, y0), pct, 0.36, color=COR_CLASSE[cl], lw=0))
                if pct >= 11:
                    ax.text(x + pct / 2, y0 + 0.18, f"{pct:.0f}%", ha="center", va="center", fontsize=5.8,
                            color=cor_texto(COR_CLASSE[cl]), family=F_SEMI)
                x += pct
            ax.text(-0.8, y0 + 0.18, ano, ha="right", va="center", fontsize=5.8, color="#777777", clip_on=False)
        ax.text(-9, i + 0.5, NOME_TAB.get(k, NOMES[k]), ha="right", va="center", fontsize=7.4,
                color="#222222", clip_on=False)
    for i, cl in enumerate(ordem):
        x = 0.12 + i * 0.097
        fig.add_artist(Rectangle((x, 0.125), 0.016, 0.011, transform=fig.transFigure, color=COR_CLASSE[cl]))
        fig.text(x + 0.019, 0.1305, cl.replace("Muito ", "M. "), va="center", fontsize=6.4, color="#333333")
    fig.text(0.12, 0.1, f"Em cada atributo, a barra de cima é {rot[0]} e a de baixo, {rot[1]}. "
                        "Classes da esquerda para a direita: crítico → muito alto.", fontsize=7, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def pagina_variacao(pdf, ctx, c: Comparacao, num: int, pares=None, rotulo_grupo=""):
    """Onde mudou: mapas de diferença (recente − anterior) de até 6 atributos-chave."""
    pares = pares or c.pares
    rot = c.rotulos
    chaves = [k for k in CHAVES_VARIACAO if all(k in a.atributos and k in b.atributos for a, b in pares)][:6]
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "ONDE MUDOU", f"diferença {rot[1]} − {rot[0]}" + (f"  ·  {rotulo_grupo}" if rotulo_grupo
                                                                           else ""), tam=24)
    lim = bk._limites([b for _, b in pares])
    for idx, k in enumerate(chaves):
        col, lin = idx % 2, idx // 2
        y_t = 0.815 - lin * 0.235
        caixa = [0.1 + col * 0.41, y_t - 0.198, 0.39, 0.18]
        ax = bk._caixa_mapa(fig, ctx, caixa, lim)
        difs = []
        for a, b in pares:
            D = exibir(k, b.atributos[k] - a.atributos[k])
            difs.append(D[b.grade.mascara])
        todas = np.concatenate(difs)
        todas = todas[np.isfinite(todas)]
        amp = float(np.nanpercentile(np.abs(todas), 98)) if len(todas) else 1.0
        amp = amp if amp > 0 else 1.0
        inv = k in it.INVERTIDOS
        for a, b in pares:
            D = exibir(k, b.atributos[k] - a.atributos[k])
            Zs, ext = bk._raster_suave(b, -D if inv else D, fator=2)
            if ext is None:
                continue
            im = ax.imshow(Zs, extent=ext, cmap=CMAP_DIF, norm=TwoSlopeNorm(0, -amp, amp), interpolation="bilinear",
                           zorder=2)
            pp = bk._patch(b.talhao.perimetro, fc="none", ec="none")
            ax.add_artist(pp)
            im.set_clip_path(pp)
            ax.add_artist(bk._patch(b.talhao.perimetro, fc="none", ec="#111111", lw=0.5, zorder=5))
        unid = TITULOS.get(k, ("", "", ""))[2]
        u = "" if unid == "adimensional" else f" {unid}"
        md = float(np.nanmean(todas)) if len(todas) else np.nan
        cs = casas_de(k, max(abs(md), amp))
        fig.text(caixa[0], y_t, CURTOS.get(k, k), fontsize=10.5, family=F_BOLD, weight="bold",
                 color=ctx.marca["cor"], va="center")
        fig.text(caixa[0] + 0.085, y_t,
                 f"média {'+' if md >= 0 else '−'}{br(abs(md), cs)}{u}", fontsize=8, color="#333333", va="center")
        axc = fig.add_axes([caixa[0] + 0.08, caixa[1] - 0.011, caixa[2] - 0.16, 0.007])
        axc.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP_DIF, aspect="auto")
        esq, dir_ = (f"+{br(amp, cs)}", f"−{br(amp, cs)}") if inv else (f"−{br(amp, cs)}", f"+{br(amp, cs)}")
        axc.set_xticks([0, 128, 255], [esq, "0", dir_], fontsize=6)
        axc.set_yticks([])
        axc.tick_params(length=0, pad=1)
        for s in axc.spines.values():
            s.set_visible(False)
    _texto(fig, 0.11, 0.098, "Verde = melhorou (aumentou; para H+Al e m%, diminuiu); vermelho = piorou. Mapas de "
                             "cada ano interpolados na mesma grade; diferenças pequenas podem refletir só a variação da "
                             "amostragem.", larg=118, fs=7, cor="#666666", entre=0.012)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_pontos_comp(pdf, ctx, c: Comparacao, num: int, pares=None, rotulo_grupo=""):
    pares = pares or c.pares
    rot = c.rotulos
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "MAPA DE PONTOS", f"amostragem de {rot[0]} e de {rot[1]}"
              + (f"  ·  {rotulo_grupo}" if rotulo_grupo else ""))
    mb = [b for _, b in pares]
    lim = bk._limites(mb, 0.06)
    modo, caixas = arranjo(lim, topo=0.83, base=0.15)
    for (caixa, ano, ms, cor_pt) in zip(caixas, rot, ([a for a, _ in pares], mb), ("#1F78B4", "#D7263D")):
        ax = bk._caixa_mapa(fig, ctx, caixa, lim)
        ax.add_artist(Rectangle((lim[0], lim[1]), lim[2] - lim[0], lim[3] - lim[1], color="#EEF2EE", zorder=0))
        n = 0
        for m in ms:
            ax.add_artist(bk._patch(m.talhao.perimetro, fc="#BDBDBD", ec="#222222", lw=0.8, zorder=2))
            am = m.talhao.amostras
            ax.scatter(am["x"], am["y"], s=6, c=cor_pt, zorder=4, lw=0)
            n += len(am)
            if len(ms) > 1:
                cc = m.talhao.perimetro.representative_point()
                ax.text(cc.x, cc.y, curto(m.talhao.nome), ha="center", va="center", fontsize=7, family=F_BOLD,
                        weight="bold", color="white", zorder=6,
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="#333333")])
        bb = ax.get_position()
        fig.text(bb.x0 + bb.width / 2 if modo == "lado" else 0.1, bb.y1 + 0.014, f"{ano}  ·  {n} pontos",
                 ha="center" if modo == "lado" else "left", va="center", fontsize=10, family=F_BOLD, weight="bold",
                 color=ctx.marca["cor"])
        if caixa is caixas[1]:
            bk.escala(fig, ax, 0.115, 0.12, frac=0.12)
    bk.norte(fig)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_volumes_comp(pdf, ctx, c: Comparacao, num: int, parte=0, por_pagina=16):
    """Totais de produto por talhão: ano anterior (mesma regra) × recente (prescrito)."""
    rot = c.rotulos
    v = c.volumes
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "VOLUMES DE PRODUTO", f"total por talhão (t) · {rot[0]} × {rot[1]}"
              + (f" · parte {parte + 1}" if parte else ""), tam=22)
    cor = ctx.marca["cor"]
    prods = list(dict.fromkeys(v["Produto"]))
    talhoes = list(dict.fromkeys(v["Talhão"]))
    pag = talhoes[parte * por_pagina:(parte + 1) * por_pagina]
    ultima = (parte + 1) * por_pagina >= len(talhoes)
    rotp = lambda p: p.replace("P2O5", "P₂O₅").replace("Produto ", "")  # noqa: E731
    ta, tb = f"Total {rot[0]} (t)", f"Total {rot[1]} (t)"
    x0, xa = 0.11, 0.3
    larg = (0.9 - 0.32) / max(len(prods), 1)
    y = 0.8
    fig.add_artist(FancyBboxPatch((0.1, y - 0.03), 0.8, 0.048, boxstyle="round,pad=0,rounding_size=0.01",
                                  transform=fig.transFigure, fc=cor, ec="none"))
    fig.text(x0 + 0.005, y - 0.006, "Talhão", va="center", fontsize=8, family=F_BOLD, weight="bold", color="white")
    fig.text(xa, y - 0.006, "Área (ha)", ha="right", va="center", fontsize=8, family=F_BOLD, weight="bold",
             color="white")
    for j, p in enumerate(prods):
        cx = 0.32 + larg * j
        fig.text(cx + larg / 2, y + 0.006, rotp(p), ha="center", va="center", fontsize=7.8, family=F_BOLD,
                 weight="bold", color="white")
        fig.text(cx + larg * 0.3, y - 0.017, rot[0], ha="center", va="center", fontsize=6.6, color="white")
        fig.text(cx + larg * 0.72, y - 0.017, rot[1], ha="center", va="center", fontsize=6.6, color="white")
    y -= 0.03
    passo = min(0.022, 0.46 / max(len(pag) + 2, 1))

    def linha(y, rotulo, area, getter, negrito=False):
        kw = dict(family=F_BOLD, weight="bold", color=cor) if negrito else dict(color="#222222")
        fig.text(x0 + 0.005, y, rotulo, va="center", fontsize=8, **kw)
        fig.text(xa, y, br(area, 1), ha="right", va="center", fontsize=8, **kw)
        for j, p in enumerate(prods):
            a, b = getter(p)
            cx = 0.32 + larg * j
            fig.text(cx + larg * 0.3, y, br(a, 1) if np.isfinite(a) else "–", ha="center", va="center", fontsize=7.8,
                     **({**kw, "color": "#777777"} if not negrito else kw))
            fig.text(cx + larg * 0.72, y, br(b, 1) if np.isfinite(b) else "–", ha="center", va="center", fontsize=7.8,
                     **kw)
    for tal in pag:
        y -= passo
        t = v[v["Talhão"] == tal]

        def g(p, t=t):
            r = t[t["Produto"] == p]
            return (float(r[ta].iloc[0]), float(r[tb].iloc[0])) if len(r) else (np.nan, np.nan)
        linha(y, curto(tal), t["Área (ha)"].iloc[0], g)
    if ultima:
        y -= passo + 0.004
        fig.add_artist(plt.Line2D([0.1, 0.9], [y + passo * 0.65] * 2, transform=fig.transFigure, color="#BBBBBB",
                                  lw=0.6))
        area = v.drop_duplicates("Talhão")["Área (ha)"].sum()
        linha(y, "Propriedade", area, lambda p: (v.loc[v["Produto"] == p, ta].sum(min_count=1),
                                                  v.loc[v["Produto"] == p, tb].sum(min_count=1)), negrito=True)
        tot = v.groupby("Produto", sort=False)[[ta, tb]].sum(min_count=1)
        altura = min(0.04 * len(tot) + 0.03, y - 0.22)
        if altura > 0.08:
            ax = fig.add_axes([0.3, y - 0.07 - altura, 0.55, altura])
            fig.text(0.11, y - 0.045, "Total da propriedade por produto (t)", fontsize=9, family=F_BOLD, weight="bold",
                     color=cor)
            idx = np.arange(len(tot))
            ax.barh(idx - 0.19, tot[ta].fillna(0), 0.36, color=bk._mistura(cor, "white", 0.55), label=rot[0])
            ax.barh(idx + 0.19, tot[tb].fillna(0), 0.36, color=cor, label=rot[1])
            for i, (va, vb) in enumerate(zip(tot[ta], tot[tb])):
                if np.isfinite(va):
                    ax.text(va, i - 0.19, f"  {br(va, 1)} t", va="center", fontsize=7, color="#555555")
                if np.isfinite(vb):
                    txt = f"  {br(vb, 1)} t"
                    if np.isfinite(va) and va > 0:
                        txt += f" ({'+' if vb >= va else '−'}{abs(100 * (vb - va) / va):.0f}%)"
                    ax.text(vb, i + 0.19, txt, va="center", fontsize=7, color="#222222")
            ax.set_yticks(idx, [rotp(p) for p in tot.index], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, np.nanmax(tot.values) * 1.35 if np.isfinite(np.nanmax(tot.values)) else 1)
            ax.set_xticks([])
            for s in ("top", "right", "bottom"):
                ax.spines[s].set_visible(False)
            ax.tick_params(length=0)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    _texto(fig, 0.11, 0.1, f"{rot[1]}: soma dos mapas de prescrição (dose da zona × área). {rot[0]}: doses calculadas "
                           "pela mesma regra e parâmetros sobre o laudo anterior (sem o ajuste de volume comprado), só "
                           "para comparação.", larg=118, fs=7, cor="#666666", entre=0.012)
    pdf.savefig(fig)
    plt.close(fig)


# ------------------------------------------------------------------ montagem
def montar(projeto, mapas, dados, par, aj, c: Comparacao, laudo_dados=None, progresso=None, blocos=None):
    """Book comparativo completo. `mapas`/`blocos`: ano recente (prescrições); `c`: pares anterior × recente."""
    from .book import (Contexto, MARCAS, agrupar_por_proximidade, lista_talhoes, pagina_atributo, pagina_capa,
                       pagina_info, pagina_metodos, pagina_ndvi, pagina_prescricao, pagina_secao,
                       pagina_subsuperficial, pagina_sumario, preparar_externos, texto_parametros)
    ctx = Contexto(projeto, mapas, dados, MARCAS[dados.marca])
    if laudo_dados is not None and laudo_dados["subsuperficial"].any():
        from .interpretacao import completar_derivados
        subs = []
        for t in projeto.talhoes:
            if len(t.sub):
                s = completar_derivados(t.sub.assign(subsuperficial=True))
                s["_talhao"] = t.nome
                subs.append(s)
        ctx.subs = pd.concat(subs) if subs else None
    if laudo_dados is not None and "argila_estimada" in laudo_dados:
        sup = laudo_dados[~laudo_dados["subsuperficial"]]
        ctx.argila_est = float(sup["argila_estimada"].mean()) if len(sup) else 0.0
    ctx.tem_areia_silte = all(k in m.atributos for m in mapas for k in ("areia", "silte"))
    preparar_externos(ctx, progresso)
    ctx.avisos += c.avisos

    roteiro = []
    add = lambda t, n, f: roteiro.append((t, n, f))  # noqa: E731
    grupos_b = agrupar_por_proximidade(c.mapas_b) if c.pares else []
    par_de = {id(b): (a, b) for a, b in c.pares}
    grupos = [[par_de[id(b)] for b in g] for g in grupos_b]
    varios = len(grupos) > 1
    rot_g = [lista_talhoes([b for _, b in g]) if varios else "" for g in grupos]
    add("Informações técnicas", 0, lambda pdf, k: pagina_info(pdf, ctx, k))
    add("Metodologias e equações", 0, lambda pdf, k: pagina_metodos(pdf, ctx, k, texto_parametros(par, aj)))
    for gi, g in enumerate(grupos):
        add("Mapa de pontos" if gi == 0 else None, 0,
            lambda pdf, k, g=g, gi=gi: pagina_pontos_comp(pdf, ctx, c, k, g, rot_g[gi]))
    grupos_rec = agrupar_por_proximidade(mapas)
    rot_r = [lista_talhoes(g) if len(grupos_rec) > 1 else "" for g in grupos_rec]
    base_ch = (["alt"] if ctx.tem_alt else []) + (["argila"] if ctx.argila_est <= 0.5 else []) \
        + (["areia", "silte"] if ctx.tem_areia_silte else [])
    for ch in base_ch:
        if all(ch in m.atributos for m in mapas):
            for gi, g in enumerate(grupos_rec):
                add(NOMES[ch] if gi == 0 else None, 0,
                    lambda pdf, k, ch=ch, g=g, gi=gi: pagina_atributo(pdf, ctx, ch, k, mapas=g, rotulo_grupo=rot_r[gi]))
    if ctx.ndvi is not None:
        for gi, g in enumerate(grupos_rec):
            add("Índice de vegetação (NDVI)" if gi == 0 else None, 0,
                lambda pdf, k, g=g, gi=gi: pagina_ndvi(pdf, ctx, k, g, rot_r[gi]))
    add("Mapas de fertilidade", 0, None)
    if c.pares:
        add("Evolução da fertilidade", 1, lambda pdf, k: pagina_evolucao(pdf, ctx, c, k))
        if len(c.pares) > 1:
            for parte in range(int(np.ceil(len(c.pares) / 22))):
                add("Evolução por talhão" if parte == 0 else None, 1,
                    lambda pdf, k, parte=parte: pagina_evolucao_talhoes(pdf, ctx, c, k, parte))
        add("Área por classe", 1, lambda pdf, k: pagina_area_classes_comp(pdf, ctx, c, k))
        for gi, g in enumerate(grupos):
            add("Onde mudou" if gi == 0 else None, 1,
                lambda pdf, k, g=g, gi=gi: pagina_variacao(pdf, ctx, c, k, g, rot_g[gi]))
        for ch in ORDEM_FERT:
            if all(ch in a.atributos and ch in b.atributos for a, b in c.pares):
                for gi, g in enumerate(grupos):
                    add(NOMES[ch] if gi == 0 else None, 1,
                        lambda pdf, k, ch=ch, g=g, gi=gi: pagina_atributo_comp(pdf, ctx, c, ch, k, g, rot_g[gi]))
    if ctx.subs is not None and len(ctx.subs):
        add(f"Camada 20–40 cm ({c.rotulos[1]})", 1, lambda pdf, k: pagina_subsuperficial(pdf, ctx, k))
    # prescrições: só do laudo mais recente (mesma lógica do book normal)
    add("Mapas de prescrição", 0, None)
    from .prescricao import Bloco
    from .regras import ler_formula
    f_ = ler_formula(aj.formula_p)
    form = f"{int(f_[0]):02d}-{int(f_[1]):02d}-{int(f_[2]):02d}" if f_ and f_[1] > 0 else None
    sub_p = (f"Formulação {form} · dose do produto comercial" if form else
             "P₂O₅ (referência — converter para o fertilizante utilizado)")
    subs_prod = {"Calcário": dados.produto_calcario, "Gesso": dados.produto_gesso,
                 "S elementar": "Enxofre elementar", "P2O5": sub_p, "KCl": dados.produto_k,
                 "KCl 1ª aplicação": f"{dados.produto_k} · {aj.kcl_pct_1:.0f}% da dose total",
                 "KCl 2ª aplicação": f"{dados.produto_k} · {100 - aj.kcl_pct_1:.0f}% da dose total"}
    blocos = blocos or [Bloco(m.talhao.nome, [m], m.zonas) for m in mapas]
    for rot_sum, b, prod, sub in bk.ordem_prescricoes(blocos, aj, sub_p, subs_prod):
        add(rot_sum, 1, lambda pdf, k, b=b, prod=prod, sub=sub: pagina_prescricao(pdf, ctx, b, prod, k, sub))
    if len(c.volumes):
        n_tal = c.volumes["Talhão"].nunique()
        for parte in range(int(np.ceil(n_tal / 16))):
            add("Volumes de produto" if parte == 0 else None, 0,
                lambda pdf, k, parte=parte: pagina_volumes_comp(pdf, ctx, c, k, parte))

    itens, num = [], 1
    for t, n, f in roteiro:
        if t:
            itens.append((t, num, n))
        num += 1
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"Book comparativo – {dados.propriedade}",
                                 "Author": MARCAS[dados.marca]["nome"]}) as pdf:
        if progresso:
            progresso("Montando páginas…")
        pagina_capa(pdf, ctx)
        pagina_sumario(pdf, ctx, itens)
        num = 1
        for t, n, f in roteiro:
            if f is None:
                pagina_secao(pdf, ctx, "MAPAS DE\nFERTILIDADE" if "fertilidade" in t else "MAPAS DE\nPRESCRIÇÃO",
                             "fertilidade" if "fertilidade" in t else "prescricao")
            else:
                f(pdf, num)
            num += 1
    return buf.getvalue(), ctx.avisos
