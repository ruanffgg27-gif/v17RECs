"""Book de semeadura em taxa variável (mesmo estilo dos books de fertilidade)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from . import book as bk  # noqa: E402
from .book import A4, F_BOLD, F_SEMI, br, curto  # noqa: E402
from .sementes import ConfigSementes, ResultadoSementes, tabela_resumo  # noqa: E402

CMAP_IF = LinearSegmentedColormap.from_list("if", ["#A50026", "#F46D43", "#FEE08B", "#A6D96A", "#1A9850", "#004529"])


def _mil(v):
    return f"{br(v / 1000, 0)} mil"


def _desenha_if(ax, m, vmin, vmax):
    Zs, ext = bk._raster_suave(m, m.atributos["if_sem"], fator=3)
    if ext is None:
        return
    im = ax.imshow(Zs, extent=ext, cmap=CMAP_IF, vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=2)
    pp = bk._patch(m.talhao.perimetro, fc="none", ec="none")
    ax.add_artist(pp)
    im.set_clip_path(pp)
    ax.add_artist(bk._patch(m.talhao.perimetro, fc="none", ec="#111111", lw=0.8, zorder=5))


def _barra_if(fig, rect, vmin, vmax, fs=7):
    ax = fig.add_axes(rect)
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP_IF, aspect="auto")
    ax.set_xticks([0, 255], [f"{br(vmin, 0)}%  menos fértil", f"mais fértil  {br(vmax, 0)}%"], fontsize=fs)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def _faixa_if(mapas):
    v = np.concatenate([m.atributos["if_sem"][m.grade.mascara] for m in mapas if "if_sem" in m.atributos])
    v = v[np.isfinite(v)]
    return float(np.floor(np.percentile(v, 1))), float(np.ceil(np.percentile(v, 99)))


def pagina_indice(pdf, ctx, num, mapas, res: ResultadoSementes, cfg: ConfigSementes, rotulo_grupo=""):
    """Índice de fertilidade (IF) da fazenda/grupo, com a regra e os máximos usados."""
    mapas = [m for m in mapas if "if_sem" in m.atributos]
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "ÍNDICE DE FERTILIDADE", "base do mapa de semeadura · regra em avaliação" + (f"  ·  {rotulo_grupo}" if rotulo_grupo
                                                                                  else ""), tam=22)
    vmin, vmax = _faixa_if(ctx.mapas)
    lim = bk._limites(mapas)
    ax = bk._caixa_mapa(fig, ctx, [0.1, 0.36, 0.8, 0.46], lim)
    for m in mapas:
        _desenha_if(ax, m, vmin, vmax)
        if len(mapas) > 1:
            c = m.talhao.perimetro.representative_point()
            ax.text(c.x, c.y, curto(m.talhao.nome), ha="center", va="center", fontsize=8, family=F_BOLD,
                    weight="bold", color="#111111", zorder=6,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="white")])
    bk.escala(fig, ax, 0.115, 0.33, frac=0.12)
    _barra_if(fig, [0.42, 0.335, 0.4, 0.012], vmin, vmax)
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    y = 0.285
    fig.text(0.12, y, "Como o índice é calculado", fontsize=10, family=F_BOLD, weight="bold", color=cor)
    y -= 0.024
    mx = res.maximos
    linhas = [
        "Cada atributo é convertido em % do maior valor encontrado nas amostras (regra de três): "
        f"P {br(mx['p'], 1)} mg/dm³ · M.O. {br(mx['mo'], 0)} g/dm³ · "
        + (f"argila {br(mx['argila'] / 10, 1)}% · " if np.isfinite(mx["argila"]) else "")
        + f"soma de bases (Ca+Mg+K) {br(mx['sb'], 1)} mmolc/dm³ = 100%.",
        "IF = (P% × 5 + M.O.% × 20 + argila% × 40 + SB% × 35) ÷ 100. Sem argila medida, a soma de bases "
        "recebe peso 75.",
        "Quanto maior o índice, mais fértil o ambiente: ali caem menos sementes, porque cada planta encontra mais "
        "recursos; nas áreas menos férteis caem mais sementes para equilibrar o estande e o potencial.",
    ]
    for l in linhas:
        fig.text(0.115, y, "›", fontsize=10, color=ac, family=F_BOLD, va="top")
        y = bk_texto(fig, 0.135, y, l, 110) - 0.008
    bk.norte(fig)
    pdf.savefig(fig)
    plt.close(fig)


def bk_texto(fig, x, y, txt, larg=100, fs=8.4, cor="#333333", entre=0.016):
    import textwrap
    for linha in textwrap.wrap(txt, larg):
        fig.text(x, y, linha, fontsize=fs, color=cor, va="top")
        y -= entre
    return y


def pagina_sementes(pdf, ctx, num, m, cfg: ConfigSementes, sub="", manuais=None):
    """Mapa de taxa de sementes de um talhão (zonas coloridas do verde escuro = menos sementes).

    `manuais`: [(dose recomendada, dose ajustada)] das faixas alteradas à mão neste talhão."""
    z = m.zonas["Sementes"]
    cs = cfg.casas(cfg.media_de(m.talhao.nome))
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    nome = curto(m.talhao.nome)
    bk.titulo(fig, ctx, f"SEMEADURA - {nome}", sub or f"taxa variável · média {br(z.dose_media, cs)} sementes/m",
              tam=24 if len(nome) < 12 else 18)
    per = m.talhao.perimetro
    xs = per.bounds
    mx, my = (xs[2] - xs[0]) * 0.04, (xs[3] - xs[1]) * 0.04
    ax = bk._caixa_mapa(fig, ctx, [0.12, 0.33, 0.76, 0.49], (xs[0] - mx, xs[1] - my, xs[2] + mx, xs[3] + my))
    cor_de = bk.cores_doses(d for d, _ in z.poligonos)
    for d, g in z.poligonos:
        ax.add_artist(bk._patch(g, fc=cor_de[float(d)], ec="none", zorder=2))
    ax.add_artist(bk._patch(per, fc="none", ec="#111111", lw=0.9, zorder=5))
    bk.escala(fig, ax, 0.83, 0.095, alinhar="dir")
    cor = ctx.marca["cor"]
    tab = z.tabela().sort_values("Dose (kg/ha)", ascending=False)
    L, y = 0.12, 0.3
    fig.text(L, y, "sementes/m", fontsize=9.5, family=F_BOLD, weight="bold", color=cor)
    fig.text(L + 0.135, y, "por ha", fontsize=8, family=F_SEMI, color="#555555")
    fig.text(L + 0.235, y, "área", fontsize=8, family=F_SEMI, color="#555555")
    passo = min(0.021, 0.19 / max(len(tab), 1))
    for _, r in tab.iterrows():
        y -= passo
        d = float(r["Dose (kg/ha)"])
        fig.add_artist(Rectangle((L, y - 0.007), 0.022, 0.014, transform=fig.transFigure, fc=cor_de[d],
                                 ec="#333333", lw=0.4))
        fig.text(L + 0.03, y, br(d, cs), va="center", fontsize=9, family=F_SEMI, color=cor)
        fig.text(L + 0.135, y, _mil(cfg.por_ha(d)), va="center", fontsize=8, color="#444444")
        fig.text(L + 0.235, y, f"{br(r['Área (ha)'], 2)} ha", va="center", fontsize=8, color="#444444")
    # resumo à direita
    x2, y2 = 0.56, 0.3
    total = cfg.por_ha(z.dose_media) * z.area_ha
    for rot, val in (("Média do mapa", f"{br(z.dose_media, cs)} sementes/m · {_mil(cfg.por_ha(z.dose_media))}/ha"),
                     ("Média comprada", f"{br(cfg.media_de(m.talhao.nome), cs)} sementes/m"),
                     ("Área", f"{br(z.area_ha, 2)} ha"),
                     ("Total de sementes", f"{br(total / 1e6, 2)} milhões"),
                     ("Espaçamento", f"{br(cfg.espacamento, 2)} m entre linhas")):
        fig.text(x2, y2, rot, fontsize=7.6, color="#666666")
        fig.text(x2, y2 - 0.017, val, fontsize=9.5, family=F_BOLD, weight="bold", color=cor)
        y2 -= 0.042
    if manuais:
        txt = "Doses ajustadas à mão (recomendado → usado): " + "; ".join(
            f"{br(a, cs)} → {br(b, cs)}" for a, b in manuais) + "."
        bk.texto_corrido(fig, 0.12, 0.083, txt, larg=0.62, y_min=bk.MOLD_Y0 + 0.012, fs=6.6, cor="#8A4B00")
    bk.norte(fig, 0.845, 0.115)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_resumo(pdf, ctx, num, mapas, cfg: ConfigSementes, manuais: dict | None = None):
    t = tabela_resumo(mapas, cfg)
    cs = cfg.casas()
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "RESUMO DA SEMEADURA", "sementes por talhão e total da propriedade", tam=22)
    cor = ctx.marca["cor"]
    cols = [("Talhão", 0.12, "l"), ("Área (ha)", 0.33, "r"), ("Média (sem/m)", 0.47, "r"), ("Faixa (sem/m)", 0.62, "r"),
            ("Mil sem/ha", 0.74, "r"), ("Milhões", 0.87, "r")]
    y = 0.8
    fig.add_artist(FancyBboxPatch((0.11, y - 0.012), 0.78, 0.028, boxstyle="round,pad=0,rounding_size=0.01",
                                  transform=fig.transFigure, fc=cor, ec="none"))
    for rot, x, al in cols:
        fig.text(x, y + 0.002, rot, fontsize=8.2, family=F_BOLD, weight="bold", color="white", va="center",
                 ha="left" if al == "l" else "right")
    passo = min(0.022, 0.6 / max(len(t) + 1, 1))
    for _, r in t.iterrows():
        y -= passo
        vals = [curto(r["Talhão"]), br(r["Área (ha)"], 2), br(r["Média do mapa (sem/m)"], cs),
                f"{br(r['Mínima (sem/m)'], cs)} – {br(r['Máxima (sem/m)'], cs)}", br(r["Sementes/ha (média)"] / 1000, 0),
                br(r["Total (milhões de sementes)"], 2)]
        for (rot, x, al), v in zip(cols, vals):
            fig.text(x, y, v, fontsize=8.4, color="#222222", va="center", ha="left" if al == "l" else "right")
    y -= passo + 0.006
    fig.add_artist(plt.Line2D([0.12, 0.88], [y + passo * 0.6] * 2, transform=fig.transFigure, color="#BBBBBB",
                              lw=0.6))
    area = t["Área (ha)"].sum()
    tot = t["Total (milhões de sementes)"].sum()
    media = float(np.average(t["Média do mapa (sem/m)"], weights=t["Área (ha)"])) if area else np.nan
    for (rot, x, al), v in zip(cols, ["Propriedade", br(area, 2), br(media, cs), "", br(cfg.por_ha(media) / 1000, 0),
                                      br(tot, 2)]):
        fig.text(x, y, v, fontsize=8.6, family=F_BOLD, weight="bold", color=cor, va="center",
                 ha="left" if al == "l" else "right")
    y -= 0.06
    editados = [n for n, v in (manuais or {}).items() if v]
    regra = ("usando toda a faixa permitida (o ponto mais fértil recebe a menor taxa e o menos fértil, a maior)"
             if cfg.esticar else "proporcionais ao inverso do índice de fertilidade")
    bk_texto(fig, 0.12, y, (f"Taxas {regra}, limitadas a ±{cfg.variacao_max:g}% da média comprada e divididas em até "
                            f"{cfg.classes} faixas aplicáveis; a média de cada mapa foi recalibrada para a média "
                            "comprada" + (" — exceto onde as doses foram ajustadas à mão ("
                                          + ", ".join(curto(n) for n in editados) + "), em que a média é a das "
                                          "doses escolhidas" if editados else " (não falta nem sobra semente)")
                            + ". Sementes/ha = sementes/m × 10.000 ÷ espaçamento "
                            f"({br(cfg.espacamento, 2)} m). Confira a regulagem da plantadeira e a germinação do lote."),
             105, fs=7.8, cor="#555555")
    y -= 0.075
    fig.text(0.12, y, "Regra da equipe em avaliação", fontsize=9.5, family=F_BOLD, weight="bold", color=cor)
    bk_texto(fig, 0.12, y - 0.02, ("A resposta da população de plantas à fertilidade depende da cultura, da cultivar, da "
                                   "época, da germinação, do estabelecimento e da disponibilidade de água. Recomenda-se "
                                   "deixar faixas com a taxa fixa (média comprada) em ambientes contrastantes para "
                                   "comparar na colheita e confirmar o ganho da taxa variável."), 105, fs=7.8,
             cor="#555555")
    pdf.savefig(fig)
    plt.close(fig)


CAPA = ("book/capa_sementes.jpg", 0.14)          # arte e quanto ela desce para abrir espaço ao título


def pagina_capa(pdf, ctx, mapas, cfg: ConfigSementes, cultura: str = ""):
    """Capa com a arte de semeadura: logo à esquerda, título à direita, cartão com produtor/propriedade."""
    d = ctx.dados
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    if not (bk.ASSETS / CAPA[0]).exists():
        bk.pagina_capa(pdf, ctx)
        return
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    img = bk._img(CAPA[0])
    fig.patch.set_facecolor(tuple(img[:20].reshape(-1, 4)[:, :3].mean(0) / 255))
    bk.imagem_capa(ax, img, CAPA[1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lg = bk._img(ctx.marca.get("logo_capa", ctx.marca["logo"]))
    hl = 0.07
    wl = hl * lg.shape[1] / lg.shape[0] * A4[1] / A4[0]
    if wl > 0.3:
        wl = 0.3
        hl = wl * lg.shape[0] / lg.shape[1] * A4[0] / A4[1]
    ax.imshow(lg, extent=(0.07, 0.07 + wl, 0.955 - hl, 0.955), aspect="auto", zorder=3)
    ax.text(0.93, 0.93, "MAPA DE", ha="right", va="center", fontsize=26, family=bk.F_XB, color=cor, zorder=3)
    ax.text(0.93, 0.893, "SEMEADURA", ha="right", va="center", fontsize=26, family=bk.F_XB, color=cor, zorder=3)
    ax.plot([0.63, 0.93], [0.866, 0.866], color=ac, lw=2.6, solid_capstyle="round", zorder=3)
    ax.text(0.93, 0.848, " · ".join(x for x in ("taxa variável", cultura) if x), ha="right", va="center", fontsize=9.5,
            family=F_SEMI, color="#444444", zorder=3)
    area = sum(m.zonas["Sementes"].area_ha for m in mapas)
    itens = [(r, v) for r, v in (("Produtor", d.produtor), ("Propriedade", d.propriedade),
                                  ("Área", f"{len(mapas)} {'talhão' if len(mapas) == 1 else 'talhões'} · {br(area, 1)} ha")) if v]
    alt = 0.03 + 0.047 * len(itens) + (0.022 if d.municipio else 0)
    x0, y0, larg = 0.055, 0.045, 0.44
    ax.add_artist(FancyBboxPatch((x0, y0), larg, alt, boxstyle="round,pad=0,rounding_size=0.018", fc=bk.CREME,
                                alpha=0.95, ec="none", mutation_aspect=A4[0] / A4[1], zorder=4))
    ax.add_artist(Rectangle((x0, y0 + 0.012), 0.008, alt - 0.024, fc=ac, ec="none", zorder=5))
    y = y0 + alt - 0.026
    for rot, val in itens:
        ax.text(x0 + 0.03, y, rot.upper(), fontsize=7.5, family=F_SEMI, color=ac, va="center", zorder=5)
        ax.text(x0 + 0.03, y - 0.02, val, fontsize=12, family=F_BOLD, weight="bold", color="#222222", va="center",
                zorder=5)
        y -= 0.047
    if d.municipio:
        ax.text(x0 + 0.03, y + 0.006, d.municipio, fontsize=9.5, color="#555555", va="center", zorder=5)
    pdf.savefig(fig)
    plt.close(fig)


def montar(projeto, mapas, dados, cfg: ConfigSementes, res: ResultadoSementes, progresso=None,
           cultura: str = "", com_indice: bool = False, manuais: dict | None = None):
    """PDF simples de semeadura: capa, um mapa por talhão e o resumo. `com_indice=True` acrescenta a página do
    índice de fertilidade (base da taxa)."""
    ctx = bk.Contexto(projeto, mapas, dados, bk.MARCAS[dados.marca])
    com = [m for m in mapas if "Sementes" in m.zonas]
    roteiro = []
    if com_indice:
        grupos = bk.agrupar_por_proximidade(com)
        for g in grupos:
            roteiro.append(lambda pdf, k, g=g: pagina_indice(pdf, ctx, k, g, res, cfg,
                                                             bk.lista_talhoes(g) if len(grupos) > 1 else ""))
    for m in com:
        roteiro.append(lambda pdf, k, m=m: pagina_sementes(pdf, ctx, k, m, cfg,
                                                           manuais=(manuais or {}).get(m.talhao.nome)))
    roteiro.append(lambda pdf, k: pagina_resumo(pdf, ctx, k, com, cfg, manuais))
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"Semeadura – {dados.propriedade}", "Author": ctx.marca["nome"]}) as pdf:
        if progresso:
            progresso("Montando páginas…")
        pagina_capa(pdf, ctx, com, cfg, cultura)
        for i, f in enumerate(roteiro):
            f(pdf, i + 1)
    return buf.getvalue()
