"""PDF do plano de amostragem: mapa dos pontos (com a grade ou as zonas) e a tabela de coordenadas."""
from __future__ import annotations

import io
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import book as bk  # noqa: E402
from .amostragem import COR_ZONA, NOMES_ZONA, Plano  # noqa: E402
from .book import A4, F_BOLD, F_SEMI, br  # noqa: E402

LINHAS_POR_COLUNA = 46


def _info(fig, ctx, plano: Plano, y0: float):
    cor = ctx.marca["cor"]
    fig.text(0.10, y0, "Resumo", fontsize=10, family=F_BOLD, weight="bold", color=cor)
    itens = [("Pontos", str(len(plano.pontos))), ("Área", f"{br(plano.area_ha, 2)} ha"),
             ("Densidade", f"1 ponto a cada {br(plano.ha_por_ponto, 2)} ha")]
    for i, (r, v) in enumerate(itens):
        fig.text(0.10, y0 - 0.02 - i * 0.016, r, fontsize=7.6, color="#666666")
        fig.text(0.19, y0 - 0.02 - i * 0.016, v, fontsize=7.8, family=F_SEMI, color="#222222")
    # por talhão (até 12 linhas em 2 colunas)
    fig.text(0.47, y0, "Por talhão" + (" e zona" if plano.zonas else ""), fontsize=10, family=F_BOLD,
             weight="bold", color=cor)
    if plano.zonas:
        n = len({z for _, z, _ in plano.zonas})
        cores = dict(zip(NOMES_ZONA[n], COR_ZONA[n])) if n in NOMES_ZONA else {}
        linhas = []
        for tal, zona, g in plano.zonas:
            k = int(((plano.pontos["Talhão"] == tal) & (plano.pontos["Zona"] == zona)).sum())
            linhas.append((cores.get(zona), f"{bk.curto(tal)} · {zona}: {k} ponto(s) · {br(g.area / 1e4, 1)} ha"))
    else:
        linhas = [(None, f"{bk.curto(t.nome)}: {int((plano.pontos['Talhão'] == t.nome).sum())} pontos · "
                         f"{br(t.area_ha, 1)} ha") for t in plano.talhoes]
    por_col = 6
    for i, (c, txt) in enumerate(linhas[:12]):
        x = 0.47 + (i // por_col) * 0.225
        y = y0 - 0.02 - (i % por_col) * 0.0145
        if c:
            fig.add_artist(Rectangle((x, y - 0.004), 0.011, 0.008, transform=fig.transFigure, fc=c, ec="#555555",
                                     lw=0.3))
        fig.text(x + (0.016 if c else 0), y, txt, fontsize=6.8, va="center", color="#333333")
    if len(linhas) > 12:
        fig.text(0.47, y0 - 0.02 - por_col * 0.0145, f"… e mais {len(linhas) - 12} (ver planilha)", fontsize=6.5,
                 color="#777777")


def pagina_mapa(pdf, ctx, num: int, plano: Plano, satelite=None):
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "PONTOS DE AMOSTRAGEM", "zonas de manejo" if plano.zonas else "grade amostral")
    from pyproj import Transformer
    from shapely.ops import transform as sh_tr
    geoms, tr = [], {}
    for t in plano.talhoes:
        tr[t.nome] = Transformer.from_crs(t.epsg, 3857, always_xy=True)
        geoms.append(sh_tr(tr[t.nome].transform, t.poligono))
    b = np.array([g.bounds for g in geoms])
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    mx, my = (x1 - x0) * 0.06, (y1 - y0) * 0.06
    lim = (x0 - mx, y0 - my, x1 + mx, y1 + my)
    ax = bk._caixa_mapa(fig, ctx, [0.10, 0.25, 0.80, 0.60], lim)
    com_img = satelite is not None
    if com_img:
        img, ext, cred = satelite
        ax.imshow(img, extent=ext, zorder=1, interpolation="bilinear")
        ax.set_xlim(lim[0], lim[2])
        ax.set_ylim(lim[1], lim[3])
        fig.text(0.90, 0.238, cred[:90], ha="right", va="top", fontsize=5.2, color="#777777")
    n_z = len({z for _, z, _ in plano.zonas})
    cores = dict(zip(NOMES_ZONA.get(n_z, []), COR_ZONA.get(n_z, [])))
    for t, g in zip(plano.talhoes, geoms):
        if not plano.zonas:
            ax.add_artist(bk._patch(g, fc="#DDE8D8" if not com_img else "#FFFFFF", alpha=0.9 if not com_img else 0.18,
                                    ec="none", zorder=2))
    for tal, zona, g in plano.zonas:
        ax.add_artist(bk._patch(sh_tr(tr[tal].transform, g), fc=cores.get(zona, "#CCCCCC"),
                                alpha=0.55 if com_img else 0.8, ec="white", lw=0.4, zorder=3))
    if not plano.zonas:
        for (_, r), c in zip(plano.pontos.iterrows(), plano.celulas):
            if c is not None and not c.is_empty:
                ax.add_artist(bk._patch(sh_tr(tr[r["Talhão"]].transform, c), fc="none",
                                        ec="#FFFFFF" if com_img else "#7A8C7A", lw=0.45, zorder=3))
    n = len(plano.pontos)
    fs = 6.4 if n <= 60 else (5.0 if n <= 160 else (3.9 if n <= 320 else 3.0))
    tam = 13 if n <= 60 else (8 if n <= 160 else (5 if n <= 320 else 3))
    for t, g in zip(plano.talhoes, geoms):
        ax.add_artist(bk._patch(g, fc="none", ec="#111111", lw=1.1, zorder=5))
        d = plano.pontos[plano.pontos["Talhão"] == t.nome]
        px, py = tr[t.nome].transform(d["x"].values, d["y"].values)
        ax.scatter(px, py, s=tam, c="#D7263D", ec="white", lw=0.35, zorder=6)
        if n <= 700:
            for xx, yy, k in zip(px, py, d["Ponto"]):
                ax.text(xx, yy + (lim[3] - lim[1]) * 0.006, str(int(k)), ha="center", va="bottom", fontsize=fs,
                        family=F_SEMI, color="#111111", zorder=7,
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=1.3, foreground="white")])
        if len(plano.talhoes) > 1:
            c = g.representative_point()
            ax.text(c.x, c.y, bk.curto(t.nome), ha="center", va="center", fontsize=11, family=F_BOLD, weight="bold",
                    color="#111111", alpha=0.55, zorder=4,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="white")])
    bk.escala(fig, ax, 0.115, 0.236)
    _info(fig, ctx, plano, 0.205)
    import textwrap
    for i, l in enumerate(textwrap.wrap(plano.descricao, 118)[:2]):
        fig.text(0.10, 0.098 - i * 0.012, l, fontsize=6.8, color="#555555")
    bk.norte(fig, y=0.245)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_tabela(pdf, ctx, num: int, plano: Plano, linhas, parte: int, partes: int):
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, "COORDENADAS DOS PONTOS",
              "graus decimais · WGS84" + (f" · {parte}/{partes}" if partes > 1 else ""), tam=20)
    cor = ctx.marca["cor"]
    zonas = bool(plano.zonas)
    colunas = 2
    for c in range(colunas):
        bloco = linhas[c * LINHAS_POR_COLUNA:(c + 1) * LINHAS_POR_COLUNA]
        if not bloco:
            continue
        x = 0.105 + c * 0.405
        cab = [(x, "Ponto", "left"), (x + 0.055, "Talhão", "left"), (x + 0.255, "Latitude", "right"),
               (x + 0.365, "Longitude", "right")]
        if zonas:
            cab.insert(2, (x + 0.115, "Zona", "left"))
        for cx, txt, ha in cab:
            fig.text(cx, 0.845, txt, ha=ha, fontsize=7.2, family=F_BOLD, weight="bold", color=cor)
        fig.add_artist(plt.Line2D([x, x + 0.365], [0.838, 0.838], color=cor, lw=0.8, transform=fig.transFigure))
        for i, r in enumerate(bloco):
            y = 0.827 - i * 0.0158
            if i % 2:
                fig.add_artist(Rectangle((x - 0.004, y - 0.0079), 0.373, 0.0158, transform=fig.transFigure,
                                         fc="#F2F5F0", ec="none", zorder=0))
            fig.text(x, y, str(int(r["Ponto"])), fontsize=6.9, va="center", family=F_SEMI, color="#222222")
            fig.text(x + 0.055, y, bk.curto(str(r["Talhão"]))[:9], fontsize=6.7, va="center", color="#333333")
            if zonas:
                fig.text(x + 0.115, y, str(r["Zona"])[:13], fontsize=6.3, va="center", color="#333333")
            fig.text(x + 0.255, y, f"{r['Latitude']:.6f}", ha="right", fontsize=6.9, va="center", color="#222222")
            fig.text(x + 0.365, y, f"{r['Longitude']:.6f}", ha="right", fontsize=6.9, va="center", color="#222222")
    fig.text(0.5, 0.082, "No laboratório, identifique cada amostra com o número do ponto (AMOSTRA 01, 02…): é por ele "
                         "que o laudo é ligado ao mapa.", ha="center", fontsize=6.8, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def montar(plano: Plano, marca: str = "Atria", propriedade: str = "", satelite=None) -> bytes:
    """`satelite`: (imagem RGB, extent em EPSG:3857, crédito) ou None."""
    ctx = SimpleNamespace(marca=bk.MARCAS[marca], dados=SimpleNamespace(propriedade=propriedade))
    linhas = [r for _, r in plano.pontos.iterrows()]
    por_pag = LINHAS_POR_COLUNA * 2
    partes = max(1, int(np.ceil(len(linhas) / por_pag)))
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"Amostragem – {propriedade}", "Author": ctx.marca["nome"]}) as pdf:
        pagina_mapa(pdf, ctx, 1, plano, satelite)
        for p in range(partes):
            pagina_tabela(pdf, ctx, 2 + p, plano, linhas[p * por_pag:(p + 1) * por_pag], p + 1, partes)
    return buf.getvalue()


def fundo_satelite(plano: Plano, fonte: str = "Esri World Imagery", chave_google: str | None = None):
    """Imagem de fundo para o mapa (None se não houver internet)."""
    from shapely.ops import unary_union

    from . import externos
    from .geo import projetar
    try:
        ll = unary_union([projetar(t.poligono, t.epsg, inverso=True) for t in plano.talhoes])
        x0, y0, x1, y1 = ll.bounds
        mx, my = (x1 - x0) * 0.08, (y1 - y0) * 0.08
        return externos.imagem_satelite(x0 - mx, y0 - my, x1 + mx, y1 + my, fonte, chave_google=chave_google)
    except Exception:  # noqa: BLE001
        return None
