"""Relatório em PDF do módulo Satélites, no estilo dos books (moldura, marca, mapas com norte e escala)."""
from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .. import book as bk  # noqa: E402
from ..book import A4, F_BOLD, F_REG, F_SEMI, F_XB, br  # noqa: E402
from . import clima as cl  # noqa: E402
from . import historico as hi  # noqa: E402
from . import lavoura as lv  # noqa: E402
from . import relevo as rl  # noqa: E402
from .diagnostico import Diagnostico  # noqa: E402

# limites internos da moldura (fração da largura da página): nada de texto ou eixo passa das linhas verdes
TEXTO_DIR, EIXO_ESQ, EIXO_DIR = 0.912, 0.14, 0.895
CAPA = ("book/capa_satelites.jpg", 0.14)          # arquivo e quanto a arte desce para abrir espaço ao título
CINZA = "#555555"


@dataclass
class DadosRelatorio:
    marca: str = "Atria"
    produtor: str = ""
    propriedade: str = ""
    talhao: str = ""
    municipio: str = ""
    data: str = ""


# ------------------------------------------------------------------ utilidades gráficas
def _ctx(d: DadosRelatorio):
    return SimpleNamespace(marca=bk.MARCAS[d.marca], dados=d)


def _eixos_limpos(ax, fs=7.5):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(labelsize=fs, colors="#333333", length=2.5, width=0.6)
    ax.grid(False)
    pos = ax.get_position()
    x0, x1 = max(pos.x0, EIXO_ESQ), min(pos.x1, EIXO_DIR)            # gráfico (e rótulos) dentro da moldura
    if (x0, x1) != (pos.x0, pos.x1) and x1 - x0 > 0.1:
        ax.set_position([x0, pos.y0, x1 - x0, pos.height])
    larg_pol = ax.get_position().width * A4[0]
    ax.yaxis.set_label_coords(-0.42 / max(larg_pol, 0.5), 0.5)       # rótulo do eixo y sem encostar nos números


def _texto(fig, x, y, txt, larg=95, fs=8.2, cor="#333333", entre=0.0155, **kw):
    """Parágrafo com quebra automática; devolve o y final."""
    for par in str(txt).split("\n"):
        for linha in textwrap.wrap(par, larg) or [""]:
            fig.text(x, y, linha, fontsize=fs, color=cor, va="top", **kw)
            y -= entre
    return y


def _subtitulo(fig, ctx, x, y, txt, fs=10):
    fig.text(x, y, txt, fontsize=fs, family=F_BOLD, weight="bold", color=ctx.marca["cor"], va="center")


def _limites(area, margem=0.06, geom=None):
    x0, y0, x1, y1 = (geom or area.poligono).bounds
    mx, my = (x1 - x0) * margem + 20, (y1 - y0) * margem + 20
    return x0 - mx, y0 - my, x1 + mx, y1 + my


def _mapa(fig, ctx, rect, area, grade=None, camada=None, cmap=None, norm=None, recortar=True, fundo=None,
          lim=None, interp="nearest"):
    lim = lim or _limites(area)
    ax = bk._caixa_mapa(fig, ctx, rect, lim)
    if fundo is not None:
        ax.imshow(fundo[0], extent=fundo[1], interpolation="bilinear", zorder=0)
    if camada is not None:
        im = ax.imshow(camada, extent=grade.extent, cmap=cmap, norm=norm, interpolation=interp, zorder=2)
        if recortar:
            pp = bk._patch(area.poligono, fc="none", ec="none")
            ax.add_artist(pp)
            im.set_clip_path(pp)
    ax.add_artist(bk._patch(area.poligono, fc="none", ec="#111111", lw=1.0, zorder=6))
    ax.set_xlim(lim[0], lim[2])
    ax.set_ylim(lim[1], lim[3])
    return ax


def _legenda(fig, x, y, itens, fs=7.4, passo=0.019, larg_col=0.2, colunas=1, titulo=None, ctx=None):
    """itens: [(cor, rótulo)]"""
    if titulo:
        fig.text(x, y + 0.018, titulo, fontsize=fs + 0.6, family=F_SEMI, color=ctx.marca["cor"] if ctx else "#222")
    por_col = int(np.ceil(len(itens) / colunas))
    for i, (cor, rot) in enumerate(itens):
        cx = x + (i // por_col) * larg_col
        cy = y - (i % por_col) * passo
        fig.add_artist(Rectangle((cx, cy - 0.0055), 0.016, 0.011, transform=fig.transFigure, fc=cor, ec="#666666",
                                 lw=0.3))
        fig.text(cx + 0.022, cy, rot, fontsize=fs, va="center", color="#222222")


def _tabela(fig, x, y, df: pd.DataFrame, larguras, fs=7.2, passo=0.0165, cor_cab=None, alinh=None, max_linhas=30):
    cor_cab = cor_cab or "#104A2A"
    xs = np.cumsum([0] + list(larguras))[:-1] + x
    for j, c in enumerate(df.columns):
        fig.text(xs[j] + (larguras[j] if (alinh and alinh[j] == "r") else 0), y, str(c), fontsize=fs,
                 family=F_BOLD, weight="bold", color=cor_cab, va="center",
                 ha="right" if (alinh and alinh[j] == "r") else "left")
    fig.add_artist(plt.Line2D([x, x + sum(larguras)], [y - 0.009, y - 0.009], transform=fig.transFigure,
                              color="#BBBBBB", lw=0.6))
    for i, (_, r) in enumerate(df.head(max_linhas).iterrows()):
        yy = y - (i + 1) * passo - 0.003
        for j, v in enumerate(r.values):
            fig.text(xs[j] + (larguras[j] if (alinh and alinh[j] == "r") else 0), yy, str(v), fontsize=fs,
                     va="center", color="#222222", ha="right" if (alinh and alinh[j] == "r") else "left")
    return y - (min(len(df), max_linhas) + 1) * passo


def _pagina(pdf, ctx, num, titulo_txt, sub=""):
    fig = plt.figure(figsize=A4)
    bk.base(fig, ctx, num)
    bk.titulo(fig, ctx, titulo_txt, sub, tam=22 if len(titulo_txt) < 24 else 18)
    return fig


def _encaixar(fig):
    """Rede de segurança: texto de página que passaria da linha direita da moldura tem a fonte reduzida (até 70%)
    e, se ainda não couber, é encurtado com reticências. Os valores mudam a cada talhão — nomes, anos e números
    mais longos não podem vazar."""
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for t in fig.texts:
        txt = t.get_text()
        if not txt.strip() or t.get_rotation() not in (0, 0.0):
            continue
        fs0 = t.get_fontsize()
        for _ in range(7):
            if t.get_window_extent(r).transformed(inv).x1 <= TEXTO_DIR:
                break
            t.set_fontsize(t.get_fontsize() * 0.95)
        while t.get_window_extent(r).transformed(inv).x1 > TEXTO_DIR and len(t.get_text()) > 8:
            t.set_text(t.get_text()[:-2].rstrip(" ·,;(") + "…")
        if t.get_fontsize() < fs0 * 0.7:
            t.set_fontsize(fs0 * 0.7)


def _fechar(pdf, fig):
    if fig.texts:
        _encaixar(fig)
    pdf.savefig(fig)
    plt.close(fig)


def _pct(v, total):
    return 100 * v / total if total else 0


# ------------------------------------------------------------------ capa
def pagina_capa(pdf, ctx, d: DadosRelatorio, diag: Diagnostico):
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    arq = bk.ASSETS / CAPA[0]
    if arq.exists():
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
    ax.text(0.93, 0.93, "DIAGNÓSTICO", ha="right", va="center", fontsize=26, family=F_XB, color=cor, zorder=3)
    ax.text(0.93, 0.893, "POR SATÉLITE", ha="right", va="center", fontsize=26, family=F_XB, color=cor, zorder=3)
    ax.plot([0.63, 0.93], [0.866, 0.866], color=ac, lw=2.6, solid_capstyle="round", zorder=3)
    ax.text(0.93, 0.848, "histórico · lavoura · relevo · clima · solo · vento", ha="right", va="center",
            fontsize=9, family=F_SEMI, color="#444444", zorder=3)
    itens = [(r, v) for r, v in (("Produtor", d.produtor), ("Propriedade", d.propriedade),
                                  ("Talhão", f"{d.talhao} · {br(diag.area.area_ha, 1)} ha" if d.talhao else
                                   f"{br(diag.area.area_ha, 1)} ha")) if v]
    alt = 0.03 + 0.047 * len(itens) + (0.022 if d.municipio else 0) + (0.02 if d.data else 0)
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
        ax.text(x0 + 0.03, y + 0.006, d.municipio, fontsize=9.5, color=CINZA, va="center", zorder=5)
        y -= 0.022
    if d.data:
        ax.text(x0 + 0.03, y + 0.006, f"Gerado em {d.data}", fontsize=8.5, color=CINZA, va="center", zorder=5)
    _fechar(pdf, fig)


# ------------------------------------------------------------------ resumo
def indicadores(diag: Diagnostico) -> list[tuple[str, str, str]]:
    A = diag.area.area_ha
    out = [("Área do talhão", f"{br(A, 1)} ha", f"centro {diag.area.centro[1]:.5f}, {diag.area.centro[0]:.5f}")]
    R = diag.relevo
    if R is not None:
        dom = max(R.decliv_ha, key=R.decliv_ha.get)
        out.append(("Altitude", f"{br(R.estat['media'], 0)} m", f"amplitude de {br(R.estat['amplitude'], 0)} m"))
        out.append(("Declividade predominante", dom.split(" (")[0], f"{_pct(R.decliv_ha[dom], A):.0f}% da área · "
                                                                    f"média {br(R.estat['decliv_media'], 1)}%"))
    H = diag.historico
    if H is not None and H.eventos:
        ab = next((e for e in H.eventos if e[0].startswith("Abertura") or e[0].startswith("Área já aberta")), None)
        if ab:
            ano = "".join(ch for ch in ab[1].split("em ")[1][:4] if ch.isdigit()) if "em " in ab[1] else ""
            out.append(("Abertura da área", ano if ab[0].startswith("Abertura") else f"antes de {H.grupos_ha.index[0]}",
                        ab[0].lower()))
        uso = H.grupos_ha.iloc[-1]
        out.append((f"Uso em {H.grupos_ha.index[-1]}", uso.idxmax(), f"{_pct(uso.max(), uso.sum()):.0f}% da área"))
    L = diag.lavoura
    if L is not None:
        if L.uniformidade:
            out.append(("Uniformidade da lavoura", f"{L.uniformidade['uniforme_pct']:.0f}%",
                        f"da área a ±10% da mediana ({L.uniformidade['ano']})"))
        if L.estab_area:
            b = L.estab_area.get("Baixo e estável", 0)
            out.append(("Vigor baixo persistente", f"{br(b, 1)} ha", f"{_pct(b, A):.0f}% da área"))
    C = diag.clima
    if C is not None and not C.anual.empty:
        out.append(("Chuva média anual", f"{br(C.anual['Chuva (mm)'].mean(), 0)} mm",
                    f"{len(C.anual)} anos agrícolas ({C.anual['Ano agrícola'].iloc[0]} a {C.anual['Ano agrícola'].iloc[-1]})"))
        est = cl.texto_estacao(C)
        if "Início" in est:
            out.append(("Início típico das chuvas", est["Início"]["mediana"],
                        f"em metade dos anos entre {est['Início']['p25']} e {est['Início']['p75']}"))
        if not C.veranicos.empty:
            v = C.veranicos.set_index("Mês")
            mes = "Jan" if "Jan" in v.index else v.index[0]
            out.append((f"Veranico ≥ 10 dias em {mes.lower()}", f"{v.loc[mes, '≥ 10 dias secos (%)']:.0f}% dos anos",
                        "frequência histórica"))
    S = diag.solo
    if S is not None and S.textura:
        out.append(("Textura estimada", S.textura.split(" (")[0].capitalize(), "SoilGrids · 250 m (contexto)"))
    V = C.vento if C is not None else None
    if V is not None and V.predominante:
        out.append(("Vento predominante", f"de {V.predominante}", f"{V.predominante_pct:.0f}% do tempo · média "
                                                                  f"{br(V.media_kmh, 1)} km/h a 2 m"))
    T = diag.termico
    if T is not None and len(out) < 12:
        from . import termico as tm
        fino, ext, res = tm.superficie(T)
        rs = tm.resumo_superficie(T, fino, ext, res, diag.area)
        out.append(("Variação de temperatura", f"{br(rs['amplitude'], 1)} °C", "na superfície do talhão (P95–P5)"))
    return out


def alertas(diag: Diagnostico) -> list[str]:
    A = diag.area.area_ha
    out = []
    L = diag.lavoura
    if L is not None:
        b = L.estab_area.get("Baixo e estável", 0) if L.estab_area else 0
        if b > 0.05 * A:
            out.append(f"{br(b, 1)} ha ({_pct(b, A):.0f}%) com vigor abaixo do restante do talhão em vários anos "
                       "— investigar causas (química, física, drenagem, histórico de manejo).")
        if L.anomalia_info and L.anomalia_info.get("queda_forte_ha", 0) >= 1:
            out.append(f"Queda recente de vigor em {br(L.anomalia_info['queda_forte_ha'], 1)} ha na imagem de "
                       f"{L.anomalia_info['data']:%d/%m/%Y} em relação ao esperado para a época.")
        if len(L.pontos):
            out.append(f"{len(L.pontos)} ponto(s) prioritário(s) de visita com coordenadas (página de pontos).")
    R = diag.relevo
    if R is not None:
        ac8 = sum(v for k, v in R.decliv_ha.items() if not k.startswith(("Plano", "Suave")))
        if ac8 > 0.05 * A:
            out.append(f"{_pct(ac8, A):.0f}% da área acima de 8% de declividade — reforçar práticas de conservação.")
        er = R.erosao_ha.get("Alta", 0) + R.erosao_ha.get("Muito alta", 0)
        if er > 0.05 * A:
            out.append(f"{br(er, 1)} ha com suscetibilidade relativa alta ou muito alta à erosão (triagem).")
        if len(R.saidas):
            out.append(f"Escoamento concentrado deixa o talhão em {len(R.saidas)} ponto(s); o maior drena "
                       f"~{br(R.saidas['Contribuição (ha)'].iloc[0], 0)} ha.")
    C = diag.clima
    if C is not None and C.atual:
        out.append(f"Safra {C.atual['ano']}: {br(C.atual['acumulado'], 0)} mm até {C.atual['ate']:%d/%m} "
                   f"({C.atual['pct_normal']:.0f}% da mediana histórica para o período).")
    T = diag.termico
    if T is not None:
        from . import termico as tm
        fino, ext, res = tm.superficie(T)
        rs = tm.resumo_superficie(T, fino, ext, res, diag.area)
        if rs["quente_ha"] > 0.05 * A:
            out.append(f"{br(rs['quente_ha'], 1)} ha com superfície ≥ 1 °C mais quente que o restante do talhão — "
                       "cruzar com vigor, relevo e cobertura.")
    return out


def pagina_resumo(pdf, ctx, num, diag: Diagnostico, satelite=None):
    fig = _pagina(pdf, ctx, num, "RAIO-X DO TALHÃO", "principais números e alertas")
    area = diag.area
    fundo = None
    if satelite is not None:
        fundo = satelite
    lim = _limites(area, 0.25)
    ax = _mapa(fig, ctx, [0.09, 0.53, 0.38, 0.33], area, lim=lim, fundo=fundo)
    if fundo is None:
        ax.add_artist(bk._patch(area.poligono, fc=bk._mistura(ctx.marca["cor"], "white", 0.8), ec="none", zorder=1))
    else:
        ax.add_artist(bk._patch(area.poligono, fc="none", ec="white", lw=2.2, zorder=5))
    bb = ax.get_position()
    bk.escala(fig, ax, bb.x0, 0.506, frac=0.07, fs=6.5)
    bk.norte(fig, bb.x1 - 0.022, 0.492, 0.028)
    if fundo is not None and len(satelite) > 2 and satelite[2]:
        cred = textwrap.wrap(str(satelite[2]), 70)[:2]
        for k, linha in enumerate(cred):
            fig.text(0.09, 0.487 - k * 0.008, linha, ha="left", va="center", fontsize=5, color="#888888")
    ind = indicadores(diag)
    x0, y0 = 0.5, 0.855
    for i, (rot, val, sub) in enumerate(ind[:12]):
        cx = x0 + (i % 2) * 0.215
        cy = y0 - (i // 2) * 0.052
        fig.add_artist(FancyBboxPatch((cx, cy - 0.043), 0.2, 0.046, boxstyle="round,pad=0,rounding_size=0.008",
                                      transform=fig.transFigure, fc=bk._mistura(ctx.marca["cor"], "white", 0.92),
                                      ec="none"))
        fig.text(cx + 0.008, cy - 0.008, rot, fontsize=6.4, color=CINZA, va="center")
        fig.text(cx + 0.008, cy - 0.022, val, fontsize=9.5, family=F_BOLD, weight="bold", color=ctx.marca["cor"],
                 va="center")
        fig.text(cx + 0.008, cy - 0.035, sub if len(sub) <= 42 else sub[:41] + "…", fontsize=5.6, color=CINZA,
                 va="center")
    y = 0.47
    _subtitulo(fig, ctx, 0.09, y, "Pontos de atenção")
    y -= 0.022
    for a in alertas(diag) or ["Nenhum alerta automático — ver as páginas de cada tema."]:
        fig.text(0.1, y, "›", fontsize=10, color=ctx.marca["acento"], family=F_BOLD, va="top")
        y = _texto(fig, 0.12, y, a, larg=100, fs=8.2) - 0.006
    y -= 0.01
    _subtitulo(fig, ctx, 0.09, y, "Como ler este relatório")
    _texto(fig, 0.09, y - 0.02, (
        "As análises combinam imagens de satélite, modelos de relevo e bases climáticas e de solo públicas. "
        "Elas mostram padrões e onde investigar; não substituem a visita a campo, a análise de solo em laboratório "
        "nem o levantamento topográfico. Vigor baixo, sozinho, não identifica a causa (nutrição, compactação, "
        "doença, falha de plantio, encharcamento etc.). As bases climáticas e de solo são grades regionais: "
        "descrevem o contexto do local, não medições dentro do talhão."), larg=108, fs=7.6, cor=CINZA)
    _fechar(pdf, fig)


# ------------------------------------------------------------------ histórico
def pagina_historico(pdf, ctx, num, diag: Diagnostico):
    H = diag.historico
    fig = _pagina(pdf, ctx, num, "HISTÓRICO DA ÁREA", f"uso e cobertura do solo {H.grupos_ha.index[0]}–"
                                                        f"{H.grupos_ha.index[-1]} · {H.colecao}")
    pct = H.grupos_ha.div(H.grupos_ha.sum(axis=1), axis=0) * 100
    ax = fig.add_axes([0.11, 0.62, 0.8, 0.2])
    ax.stackplot(pct.index, pct.T.values, colors=[hi.COR_GRUPO[g] for g in pct.columns], lw=0)
    ax.set_xlim(pct.index[0], pct.index[-1])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% da área", fontsize=7.5)
    _eixos_limpos(ax)
    _legenda(fig, 0.11, 0.595, [(hi.COR_GRUPO[g], g) for g in pct.columns], colunas=4, larg_col=0.2, fs=7)
    # mini mapas
    anos = list(H.mapas)
    esc = sorted(set([anos[0], anos[len(anos) // 3], anos[2 * len(anos) // 3], anos[-1]]))
    cores = {g: i for i, g in enumerate(hi.GRUPOS)}
    cmap = ListedColormap([hi.COR_GRUPO[g] for g in hi.GRUPOS])
    norm = BoundaryNorm(np.arange(len(hi.GRUPOS) + 1) - 0.5, cmap.N)
    larg = 0.8 / len(esc)
    for k, a in enumerate(esc):
        m = H.mapas[a]
        idx = np.full(m.shape, np.nan)
        ok = np.isfinite(m)
        idx[ok] = [cores[hi.grupo_de(v)] for v in m[ok]]
        axm = _mapa(fig, ctx, [0.1 + k * larg, 0.37, larg - 0.02, 0.17], diag.area, H.grade, idx, cmap, norm)
        fig.text(0.1 + k * larg + (larg - 0.02) / 2, 0.36, str(a), ha="center", fontsize=9, family=F_BOLD,
                 weight="bold", color=ctx.marca["cor"])
    y = 0.325
    _subtitulo(fig, ctx, 0.09, y, "Linha do tempo")
    y -= 0.02
    for t, desc in H.eventos:
        y0 = y
        yt = _texto(fig, 0.1, y, t, larg=28, fs=7.8, cor=ctx.marca["cor"], family=F_SEMI)
        y = min(yt, _texto(fig, 0.36, y0, desc, larg=74, fs=7.8)) - 0.004
    if H.prodes is not None:
        y -= 0.008
        if len(H.prodes):
            txt = "; ".join(f"{int(r['Ano'])}: {br(r['Área (ha)'], 1)} ha" for _, r in H.prodes.iterrows())
            _texto(fig, 0.1, y, f"PRODES/INPE ({H.prodes_bioma}) — desmatamento detectado no talhão: {txt}.",
                   larg=110, fs=7.6, cor=CINZA)
        else:
            _texto(fig, 0.1, y, "PRODES/INPE: nenhum polígono de desmatamento anual registrado dentro do talhão "
                                "no período monitorado.", larg=110, fs=7.6, cor=CINZA)
    fig.text(0.09, 0.075, f"Fonte: Projeto {H.colecao} (mapas anuais de 30 m). Classes agrupadas; proporções "
                          "calculadas dentro do perímetro.", fontsize=6.4, color="#777777")
    _fechar(pdf, fig)


# ------------------------------------------------------------------ lavoura
def pagina_lavoura_serie(pdf, ctx, num, diag: Diagnostico):
    L = diag.lavoura
    fig = _pagina(pdf, ctx, num, "COMPORTAMENTO DA LAVOURA",
                  f"evolução do vigor · Sentinel-2 · {len(L.cenas)} imagens sem nuvens")
    s = L.serie
    ax = fig.add_axes([0.12, 0.56, 0.78, 0.25])
    dt = pd.to_datetime(s["data"])
    for k, (ano, g) in enumerate(s.groupby("ano_agricola")):
        d0, d1 = pd.Timestamp(int(ano[:4]), 7, 1), pd.Timestamp(int(ano[:4]) + 1, 6, 30)
        if k % 2 == 0:
            ax.axvspan(d0, d1, color="#F2F2F2", lw=0, zorder=0)
        ax.text(d0 + (d1 - d0) / 2, 0.985, ano, ha="center", va="top", fontsize=6.8, color=CINZA,
                transform=ax.get_xaxis_transform())
    ax.fill_between(dt, s["p10"], s["p90"], color=ctx.marca["cor"], alpha=0.15, lw=0, label="10% a 90% da área")
    ax.plot(dt, s["media"], color=ctx.marca["cor"], lw=1.8, marker="o", ms=2.8, label="NDVI médio do talhão")
    ax.set_ylim(0, 1)
    ax.set_ylabel("NDVI", fontsize=7.5)
    _eixos_limpos(ax)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left", ncol=2)
    # índices por ano
    if not L.indices_ano.empty:
        _subtitulo(fig, ctx, 0.09, 0.495, "Índices no pico de cada ano agrícola")
        t = L.indices_ano.copy()
        tab = pd.DataFrame({"Ano agrícola": t["Ano agrícola"],
                            "Data do pico": pd.to_datetime(t["Data do pico"]).dt.strftime("%d/%m/%Y"),
                            **{k: t[k].map(lambda v: br(v, 2)) for k in ("NDVI", "NDRE", "NDMI", "MSAVI2")},
                            "CV NDVI": t["CV do NDVI (%)"].map(lambda v: f"{br(v, 1)}%"),
                            "Imagens": t["Imagens"]})
        y = _tabela(fig, 0.1, 0.47, tab, [0.12, 0.13, 0.09, 0.09, 0.09, 0.1, 0.1, 0.08], cor_cab=ctx.marca["cor"],
                    alinh=["l", "l", "r", "r", "r", "r", "r", "r"])
        axb = fig.add_axes([0.12, max(0.14, y - 0.2), 0.78, min(0.17, y - 0.16)])
        x = np.arange(len(t))
        cores = [ctx.marca["cor"], bk._mistura(ctx.marca["cor"], "white", 0.35), ctx.marca["acento"],
                 bk._mistura(ctx.marca["acento"], "white", 0.45)]
        for k, ind in enumerate(("NDVI", "NDRE", "NDMI", "MSAVI2")):
            axb.bar(x + (k - 1.5) * 0.19, t[ind], 0.18, color=cores[k], label=ind)
        axb.set_xticks(x, t["Ano agrícola"], fontsize=7)
        axb.set_ylim(0, 1)
        _eixos_limpos(axb)
        axb.legend(fontsize=6.6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _texto(fig, 0.09, 0.105, "NDVI: vigor/biomassa · NDRE (borda do vermelho): clorofila, útil com dossel fechado · "
                             "NDMI: água na vegetação · MSAVI2: vigor com correção do solo exposto. Anos com culturas ou "
                             "estágios diferentes não são diretamente comparáveis.", larg=118, fs=6.8, cor="#777777")
    _fechar(pdf, fig)


DESC_INDICE = {
    "ndvi": ("NDVI", "vigor e biomassa; satura quando a lavoura fecha"),
    "ndre": ("NDRE", "clorofila (nitrogênio); sensível com dossel fechado"),
    "ndmi": ("NDMI", "água na vegetação"),
    "msavi2": ("MSAVI2", "vigor com correção do solo exposto"),
}
# uma paleta por índice: cada um mede uma coisa diferente (menor → maior)
CMAP_INDICE = {
    "ndvi": LinearSegmentedColormap.from_list("ndvi", ["#A50026", "#F46D43", "#FEE08B", "#D9EF8B", "#66BD63",
                                                       "#1A9850", "#004529"]),            # vermelho → verde (vigor)
    "ndre": LinearSegmentedColormap.from_list("ndre", ["#FCE4EC", "#F8BBD9", "#CE93D8", "#9C5FC4", "#6A2C91",
                                                       "#3A0F5C"]),                       # rosa → roxo (clorofila)
    "ndmi": LinearSegmentedColormap.from_list("ndmi", ["#8C510A", "#D8B365", "#F6E8C3", "#C7E3F0", "#5BA3D0",
                                                       "#1B5E9E", "#0B2F5E"]),            # marrom (seco) → azul (água)
    "msavi2": LinearSegmentedColormap.from_list("msavi2", ["#FDE725", "#5DC863", "#21908C", "#3B528B",
                                                           "#440154"]),                   # amarelo → verde → roxo
}
LEGENDA_INDICE = {"ndvi": "vermelho = menor · verde = maior", "ndre": "rosa claro = menor · roxo = maior",
                  "ndmi": "marrom = mais seco · azul = mais água", "msavi2": "amarelo = menor · roxo = maior"}


def faixa_do_talhao(v: np.ndarray, minimo: float = 0.02) -> tuple[float, float]:
    """Limites da legenda ajustados à variação DENTRO do talhão (percentis 2 e 98), com amplitude mínima.

    Com limites fixos (0–1), um NDVI de lavoura fechada fica todo na cor máxima e não se vê diferença; esticando
    a escala entre os percentis do próprio talhão, as diferenças internas aparecem — e os valores reais ficam
    escritos na legenda."""
    v = v[np.isfinite(v)]
    lo, hi = (float(np.percentile(v, 2)), float(np.percentile(v, 98))) if len(v) else (0.0, 1.0)
    if hi - lo < minimo:
        c = (hi + lo) / 2
        lo, hi = c - minimo / 2, c + minimo / 2
    return lo, hi


def pagina_lavoura_indices(pdf, ctx, num, diag: Diagnostico):
    """Os quatro índices lado a lado: média histórica do pico de cada ano agrícola."""
    L = diag.lavoura
    g = L.grade
    n = len(L.hist_anos)
    periodo = f"{L.hist_anos[0]} a {L.hist_anos[-1]}" if n > 1 else (L.hist_anos[0] if n else "")
    fig = _pagina(pdf, ctx, num, "ÍNDICES DE VEGETAÇÃO",
                  (f"média histórica de {n} anos agrícolas · {periodo}" if n > 1 else f"ano agrícola {periodo}"))
    for i, k in enumerate(lv.INDICES):
        col, lin = i % 2, i // 2
        x0, y_t = 0.09 + col * 0.42, 0.815 - lin * 0.365
        nome, desc = DESC_INDICE[k]
        fig.text(x0, y_t, nome, fontsize=12, family=F_BOLD, weight="bold", color=ctx.marca["cor"], va="center")
        fig.text(x0 + 0.03 + 0.0135 * len(nome), y_t - 0.001, desc, fontsize=6.8, color=CINZA, va="center")
        Z = L.hist[k]
        v = Z[g.mascara]
        lo, hi = faixa_do_talhao(v)
        falta = ~np.isfinite(Z)
        if falta.any() and not falta.all():        # estende até a borda (o recorte é pelo perímetro, sem serrilhado)
            from scipy import ndimage
            idx = ndimage.distance_transform_edt(falta, return_distances=False, return_indices=True)
            Z = Z[tuple(idx)]
        _mapa(fig, ctx, [x0, y_t - 0.285, 0.39, 0.265], diag.area, g, Z, CMAP_INDICE[k],
              matplotlib.colors.Normalize(lo, hi), interp="bilinear")
        ax_cb = fig.add_axes([x0 + 0.06, y_t - 0.305, 0.27, 0.009])
        ax_cb.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP_INDICE[k], aspect="auto")
        fig.text(x0 + 0.195, y_t - 0.33, LEGENDA_INDICE[k], ha="center", va="center", fontsize=6, color="#777777")
        ax_cb.set_xticks([0, 127.5, 255], [br(lo, 2), f"média {br(float(np.nanmean(v)), 2)}", br(hi, 2)], fontsize=6.4)
        ax_cb.set_yticks([])
        ax_cb.tick_params(length=0, pad=2)
        for s_ in ax_cb.spines.values():
            s_.set_visible(False)
    _texto(fig, 0.09, 0.098, ("Cada mapa é a média, entre os anos, do maior valor do índice em cada ponto no ano agrícola "
                              "(pico da lavoura). A escala de cores de cada mapa vai do percentil 2 ao 98 do próprio "
                              "talhão, para evidenciar as diferenças internas: compare as cores dentro de um mapa e use "
                              "os números da legenda para comparar entre talhões. Cada índice tem a sua paleta porque mede uma "
                              "coisa diferente."),
           larg=128, fs=6.6, cor="#666666", entre=0.0115)
    _fechar(pdf, fig)


def pagina_lavoura_mapas(pdf, ctx, num, diag: Diagnostico):
    L = diag.lavoura
    A = diag.area.area_ha
    fig = _pagina(pdf, ctx, num, "ESTABILIDADE E ANOMALIAS", "onde o vigor se repete e o que mudou recentemente")
    g = L.grade
    if L.estab_classe is not None:
        cmap = ListedColormap([lv.COR_ESTAB[c] for c in lv.CLASSES_ESTAB])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
        ax = _mapa(fig, ctx, [0.09, 0.47, 0.5, 0.34], diag.area, g, L.estab_classe, cmap, norm)
        _subtitulo(fig, ctx, 0.62, 0.8, "Estabilidade do vigor")
        y_anos = _texto(fig, 0.62, 0.787, f"{len(L.vigor_rel)} anos agrícolas: {', '.join(L.vigor_rel)}", larg=46,
                        fs=6.6, cor=CINZA, entre=0.011)
        _legenda(fig, 0.62, min(0.75, y_anos - 0.012), [(lv.COR_ESTAB[c], f"{c}: {br(L.estab_area[c], 1)} ha ({_pct(L.estab_area[c], A):.0f}%)")
                                   for c in lv.CLASSES_ESTAB], fs=7.2, passo=0.022)
        _texto(fig, 0.62, 0.64, "Vigor relativo de cada ano (em relação à mediana do talhão). Estável = mesmo "
                                "comportamento todos os anos; instável = muda de um ano para outro.", larg=44, fs=6.8,
               cor=CINZA)
        _pontos_no_mapa(ax, L)
        bk.escala(fig, ax, 0.09, 0.455)
    if L.anomalia is not None:
        cmap = LinearSegmentedColormap.from_list("anom", ["#B2182B", "#EF8A62", "#F7F7F7", "#67A9CF", "#2166AC"])
        norm = BoundaryNorm([-1, -0.15, -0.07, 0.07, 0.15, 1], cmap.N)
        ax2 = _mapa(fig, ctx, [0.09, 0.1, 0.5, 0.32], diag.area, g, L.anomalia, cmap, norm)
        i = L.anomalia_info
        _subtitulo(fig, ctx, 0.62, 0.41, "Anomalia recente")
        _texto(fig, 0.62, 0.392, f"Imagem de {i['data']:%d/%m/%Y} comparada com a {i['base']}.", larg=44, fs=6.8,
               cor=CINZA)
        _legenda(fig, 0.62, 0.33, [("#B2182B", f"Queda forte (< −0,15): {br(i['queda_forte_ha'], 1)} ha"),
                                   ("#EF8A62", f"Queda (−0,15 a −0,07): {br(i['queda_ha'], 1)} ha"),
                                   ("#F7F7F7", "Dentro do esperado"),
                                   ("#67A9CF", f"Acima (> 0,07): {br(i['alta_ha'], 1)} ha")], fs=7, passo=0.021)
        _texto(fig, 0.62, 0.235, f"Diferença de NDVI descontada a variação do talhão inteiro "
                                 f"({'+' if i['desvio_geral'] >= 0 else ''}{br(i['desvio_geral'], 2)}), para destacar "
                                 "mudanças localizadas. Mudança recente pede investigação diferente de uma mancha "
                                 "que se repete há anos.", larg=44, fs=6.8, cor=CINZA)
        bk.escala(fig, ax2, 0.09, 0.088)
    bk.norte(fig)
    _fechar(pdf, fig)


def _pontos_no_mapa(ax, L, cor="#111111"):
    if L.pontos is None or not len(L.pontos):
        return
    g = L.grade
    for _, r in L.pontos.iterrows():
        x = g.x0 + (r["_col"] + 0.5) * g.res
        y = g.y1 - (r["_lin"] + 0.5) * g.res
        ax.plot(x, y, marker="o", ms=9, mfc="white", mec=cor, mew=1.2, zorder=8)
        ax.text(x, y, r["Ponto"][1:], ha="center", va="center", fontsize=5.8, family=F_BOLD, weight="bold",
                color=cor, zorder=9)


def pagina_lavoura_pontos(pdf, ctx, num, diag: Diagnostico):
    L = diag.lavoura
    A = diag.area.area_ha
    fig = _pagina(pdf, ctx, num, "UNIFORMIDADE E PONTOS", "vigor atual, baixa cobertura recorrente e onde visitar")
    g = L.grade
    if L.uniformidade:
        u = L.uniformidade
        pico = L.picos[u["ano"]]
        rel = 100 * pico / u["mediana"]
        cores = ["#D73027", "#FC8D59", "#FFFFBF", "#91CF60", "#1A9850"]
        ax = _mapa(fig, ctx, [0.1, 0.52, 0.37, 0.28], diag.area, g, rel, ListedColormap(cores),
                   BoundaryNorm([0, 80, 90, 110, 120, 1000], 5))
        _subtitulo(fig, ctx, 0.1, 0.83, f"Vigor no pico de {u['ano']}")
        fig.text(0.1, 0.812, "% da mediana do talhão", fontsize=6.8, color=CINZA)
        _legenda(fig, 0.1, 0.495, [(c, f"{k}: {_pct(v, A):.0f}%") for c, (k, v) in zip(cores, u["classes"].items())],
                 colunas=2, larg_col=0.185, fs=6.8, passo=0.018)
        fig.text(0.1, 0.43, f"CV {br(u['cv'], 1)}% · {u['uniforme_pct']:.0f}% da área a ±10% da mediana",
                 fontsize=7.4, family=F_SEMI, color=ctx.marca["cor"])
    if L.baixa_cobertura is not None:
        cmap = LinearSegmentedColormap.from_list("bc", ["#F7F7F7", "#FEE08B", "#F46D43", "#A50026"])
        _mapa(fig, ctx, [0.54, 0.52, 0.37, 0.28], diag.area, g, L.baixa_cobertura, cmap,
              BoundaryNorm([0, 0.25, 0.5, 0.75, 1.01], cmap.N))
        _subtitulo(fig, ctx, 0.54, 0.83, "Baixa cobertura recorrente")
        fig.text(0.54, 0.812, "% das imagens em que o ponto ficou abaixo de 70% da mediana", fontsize=6.3,
                 color=CINZA)
        rec = float(np.nansum(L.baixa_cobertura >= 0.5) * g.area_pixel_ha)
        _legenda(fig, 0.54, 0.495, [("#F7F7F7", "< 25% das imagens"), ("#FEE08B", "25–50%"), ("#F46D43", "50–75%"),
                                    ("#A50026", "> 75%")], colunas=2, larg_col=0.185, fs=6.8, passo=0.018)
        fig.text(0.54, 0.43, f"{br(rec, 1)} ha ({_pct(rec, A):.0f}%) em metade ou mais das imagens",
                 fontsize=7.4, family=F_SEMI, color=ctx.marca["cor"])
    _subtitulo(fig, ctx, 0.1, 0.395, "Pontos prioritários para visita")
    if len(L.pontos):
        cmap = ListedColormap([lv.COR_ESTAB[c] for c in lv.CLASSES_ESTAB])
        base = L.estab_classe if L.estab_classe is not None else None
        ax3 = _mapa(fig, ctx, [0.1, 0.11, 0.33, 0.26], diag.area, g, base, cmap,
                    BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
        _pontos_no_mapa(ax3, L)
        bk.escala(fig, ax3, 0.1, 0.095)
        t = L.pontos
        tab = pd.DataFrame({"": t["Ponto"], "Tipo": t["Tipo"], "ha": t["Área (ha)"].map(lambda v: br(v, 1)),
                            "Latitude": t["Latitude"].map(lambda v: f"{v:.6f}"),
                            "Longitude": t["Longitude"].map(lambda v: f"{v:.6f}")})
        _tabela(fig, 0.47, 0.36, tab, [0.03, 0.15, 0.045, 0.095, 0.105], fs=6.5, passo=0.0165,
                cor_cab=ctx.marca["cor"], alinh=["l", "l", "r", "r", "r"])
        _texto(fig, 0.47, 0.155, "Mapa de fundo: estabilidade do vigor. Coordenadas em graus decimais (WGS84); o "
                                 "arquivo KML com os pontos pode ser baixado no app para abrir no celular.",
               larg=62, fs=6.3, cor="#777777")
    else:
        fig.text(0.08, 0.37, "Nenhuma mancha relevante (≥ 0,3 ha) encontrada pelos critérios.", fontsize=8, color=CINZA)
    bk.norte(fig)
    _fechar(pdf, fig)


# ------------------------------------------------------------------ relevo
def pagina_relevo_1(pdf, ctx, num, diag: Diagnostico):
    R = diag.relevo
    A = diag.area.area_ha
    fig = _pagina(pdf, ctx, num, "RELEVO", f"altitude e declividade · {R.fonte}")
    g = R.grade
    lim = _limites(diag.area, 0.08)
    # altitude com sombreamento e curvas de nível
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    z = R.dem
    zmin, zmax = np.nanpercentile(z[g.mascara], [1, 99])
    cm = LinearSegmentedColormap.from_list("hip", ["#1A9641", "#A6D96A", "#FFFFBF", "#FDAE61", "#A6611A"])
    rgb = ls.shade(z, cmap=cm, vert_exag=3, blend_mode="soft", vmin=zmin, vmax=zmax, dx=g.res, dy=g.res)
    ax = _mapa(fig, ctx, [0.09, 0.47, 0.5, 0.34], diag.area, lim=lim)
    im = ax.imshow(rgb, extent=g.extent, zorder=2, interpolation="bilinear")
    pp = bk._patch(diag.area.poligono, fc="none", ec="none")
    ax.add_artist(pp)
    im.set_clip_path(pp)
    X, Y = g.malha()
    passo = max(1, float(_passo_curvas(zmax - zmin)))
    cs = ax.contour(X, Y, np.where(g.mascara, z, np.nan), levels=np.arange(np.floor(zmin), zmax + passo, passo),
                    colors="#333333", linewidths=0.4, alpha=0.6, zorder=3)
    ax.clabel(cs, fontsize=5, fmt="%d", inline=True)
    ax.set_xlim(lim[0], lim[2])
    ax.set_ylim(lim[1], lim[3])
    bk.escala(fig, ax, 0.09, 0.455)
    _subtitulo(fig, ctx, 0.62, 0.8, "Altitude")
    e = R.estat
    for k, (r, v) in enumerate((("Mínima", e["min"]), ("Média", e["media"]), ("Máxima", e["max"]),
                                ("Amplitude", e["amplitude"]))):
        fig.text(0.62, 0.77 - k * 0.022, f"{r}: {br(v, 0)} m", fontsize=8, color="#222222")
    fig.text(0.62, 0.675, f"Curvas de nível a cada {br(passo, 0)} m", fontsize=6.8, color=CINZA)
    ax_cb = fig.add_axes([0.62, 0.62, 0.25, 0.015])
    ax_cb.imshow(np.linspace(0, 1, 256)[None, :], cmap=cm, aspect="auto")
    ax_cb.set_xticks([0, 255], [f"{zmin:.0f} m", f"{zmax:.0f} m"], fontsize=6.5)
    ax_cb.set_yticks([])
    for s in ax_cb.spines.values():
        s.set_visible(False)
    # declividade
    cmap = ListedColormap(rl.DECLIV_COR)
    norm = BoundaryNorm(rl.DECLIV_LIM[:-1] + [1000], cmap.N)
    from scipy.ndimage import zoom
    fino = SimpleNamespace(extent=g.extent)
    ax2 = _mapa(fig, ctx, [0.09, 0.1, 0.5, 0.32], diag.area, fino, zoom(R.declividade, 3, order=1), cmap, norm,
                lim=lim)
    bk.escala(fig, ax2, 0.09, 0.088)
    _subtitulo(fig, ctx, 0.62, 0.41, "Declividade (classes Embrapa)")
    _legenda(fig, 0.62, 0.38, [(c, f"{r}: {br(R.decliv_ha[r], 1)} ha ({_pct(R.decliv_ha[r], A):.0f}%)")
                               for c, r in zip(rl.DECLIV_COR, rl.DECLIV_ROT) if R.decliv_ha[r] > 0 or
                               rl.DECLIV_ROT.index(r) < 3], fs=7, passo=0.022)
    fig.text(0.62, 0.235, f"Declividade média: {br(e['decliv_media'], 1)}%", fontsize=6.8, color=CINZA)
    fig.text(0.62, 0.221, f"90% da área abaixo de {br(e['decliv_p90'], 1)}%", fontsize=6.8, color=CINZA)
    bk.norte(fig)
    _fechar(pdf, fig)


def _passo_curvas(amp):
    for p in (1, 2, 5, 10, 20, 25, 50):
        if amp / p <= 14:
            return p
    return 100


def pagina_relevo_2(pdf, ctx, num, diag: Diagnostico):
    R = diag.relevo
    A = diag.area.area_ha
    g = R.grade
    fig = _pagina(pdf, ctx, num, "PAISAGEM E ÁGUA", "posição no relevo, escoamento provável e erosão")
    lim = _limites(diag.area, 0.12)
    # posição na paisagem
    cmap = ListedColormap([rl.COR_POS[p] for p in rl.POSICOES])
    norm = BoundaryNorm(np.arange(len(rl.POSICOES) + 1) - 0.5, cmap.N)
    _mapa(fig, ctx, [0.1, 0.585, 0.37, 0.23], diag.area, g, R.posicao.astype(float), cmap, norm, lim=lim)
    _subtitulo(fig, ctx, 0.1, 0.835, "Posição na paisagem")
    _legenda(fig, 0.1, 0.56, [(rl.COR_POS[p], f"{p}: {_pct(R.posicao_ha[p], A):.0f}%") for p in rl.POSICOES],
             colunas=2, larg_col=0.185, fs=6.6, passo=0.017)
    # escoamento
    ax2 = _mapa(fig, ctx, [0.54, 0.585, 0.37, 0.23], diag.area, lim=lim)
    acc = R.acumulacao
    X, Y = g.malha()
    ax2.imshow(np.where(g.mascara, 0.98, 0.9)[..., None] * np.ones(3), extent=g.extent, zorder=1,
               interpolation="nearest")
    rio = np.where(acc >= 2, np.log10(acc), np.nan)
    if np.isfinite(rio).any():
        ax2.imshow(rio, extent=g.extent, cmap=LinearSegmentedColormap.from_list("agua", ["#9ECAE1", "#08306B"]),
                   vmin=np.log10(2), vmax=np.nanmax(rio), zorder=3, interpolation="nearest")
    passo = max(2, int(min(g.largura, g.altura) / 12))
    gy, gx = np.gradient(R.dem, g.res)
    sl = (slice(passo // 2, None, passo), slice(passo // 2, None, passo))
    mag = np.hypot(gx, gy)[sl] + 1e-9
    ax2.quiver(X[sl], Y[sl], -gx[sl] / mag, gy[sl] / mag, color="#666666", scale=30, width=0.0035, headwidth=4,
               zorder=4, alpha=0.6)
    for k, r in R.saidas.iterrows():
        x = g.x0 + (r["_col"] + 0.5) * g.res
        y = g.y1 - (r["_lin"] + 0.5) * g.res
        ax2.plot(x, y, marker="v", ms=8, color=ctx.marca["acento"], mec="white", zorder=7)
        ax2.text(x, y + g.res * 2.5, f"S{k + 1}", ha="center", fontsize=6.5, family=F_BOLD, weight="bold",
                 color=ctx.marca["acento"], zorder=7)
    ax2.set_xlim(lim[0], lim[2])
    ax2.set_ylim(lim[1], lim[3])
    _subtitulo(fig, ctx, 0.54, 0.835, "Escoamento provável")
    _texto(fig, 0.54, 0.568, "Azul: caminhos de concentração da água (≥ 2 ha de contribuição; mais escuro = mais "
                             "área). Setas: queda do terreno. S: onde o fluxo deixa o talhão.", larg=60, fs=6.4,
           cor=CINZA)
    # perfis
    _subtitulo(fig, ctx, 0.1, 0.47, "Perfis de elevação")
    ax3 = fig.add_axes([0.14, 0.25, 0.32, 0.19])
    for k, p in enumerate(R.perfis):
        ax3.plot(p["dist"], p["z"], color=[ctx.marca["cor"], ctx.marca["acento"]][k % 2], lw=1.5,
                 label=f"{p['nome']} ({br(p['decliv_media'], 1)}%)")
    ax3.set_xlabel("Distância (m)", fontsize=7)
    ax3.set_ylabel("Altitude (m)", fontsize=7)
    _eixos_limpos(ax3, 6.6)
    ax3.legend(fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    # erosão
    cmap = ListedColormap(rl.EROSAO_COR)
    _mapa(fig, ctx, [0.54, 0.235, 0.37, 0.21], diag.area, g, R.erosao.astype(float), cmap,
          BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), lim=lim)
    _subtitulo(fig, ctx, 0.54, 0.47, "Suscetibilidade relativa à erosão")
    _legenda(fig, 0.54, 0.212, [(c, f"{r}: {br(R.erosao_ha[r], 1)} ha") for c, r in zip(rl.EROSAO_COR, rl.EROSAO_ROT)],
             colunas=2, larg_col=0.185, fs=6.6, passo=0.017)
    _texto(fig, 0.1, 0.155, "Erosão: triagem pelo fator topográfico LS (Moore & Burch, 1986)"
                             + (", ponderado pela cobertura vegetal média observada pelo Sentinel-2" if R.cobertura_usada
                                else "") + ". Indica prioridade para conservação; não é perda de solo medida. "
                             "Modelo de superfície de 30 m: inclui vegetação e estruturas e perde precisão em áreas "
                             "muito planas — para terraços e drenagem, fazer levantamento topográfico.",
           larg=125, fs=6.4, cor="#777777")
    for k, a in enumerate(R.avisos):
        fig.text(0.1, 0.105 - k * 0.013, "⚠ " + a, fontsize=6.4, color="#B2182B")
    _fechar(pdf, fig)


# ------------------------------------------------------------------ clima
def pagina_clima_1(pdf, ctx, num, diag: Diagnostico):
    C = diag.clima
    anos = f"{C.mensal.index.min()}–{C.mensal.index.max()}" if not C.mensal.empty else ""
    fig = _pagina(pdf, ctx, num, "PERFIL CLIMÁTICO", f"chuva {anos} · {C.fonte_chuva.split(' (')[0]}")
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    # normais mensais (ano agrícola)
    ax = fig.add_axes([0.12, 0.62, 0.78, 0.18])
    ordem = [m for m in cl.ORDEM_AGRICOLA if m in C.normal.index]
    x = np.arange(len(ordem))
    n = C.normal.loc[ordem]
    ax.bar(x, n["media"], 0.65, color=bk._mistura("#2B7BB9", "white", 0.25))
    ax.errorbar(x, n["media"], yerr=[n["media"] - n["p20"], n["p80"] - n["media"]], fmt="none", ecolor="#08306B",
                elinewidth=0.8, capsize=2)
    for i, (v, p8) in enumerate(zip(n["media"], n["p80"])):
        ax.text(i, p8 + 5, f"{v:.0f}", ha="center", fontsize=6.3, color="#08306B")
    ax.set_xticks(x, [cl.MESES[m - 1] for m in ordem], fontsize=7)
    ax.set_ylabel("mm/mês", fontsize=7.5)
    _eixos_limpos(ax)
    _subtitulo(fig, ctx, 0.09, 0.825, "Chuva mensal: média e faixa de 20% a 80% dos anos")
    # anual
    ax2 = fig.add_axes([0.12, 0.36, 0.78, 0.19])
    a = C.anual
    med = a["Chuva (mm)"].median()
    cores = ["#D6604D" if v < a["Chuva (mm)"].quantile(0.2) else "#4393C3" if v > a["Chuva (mm)"].quantile(0.8)
             else "#BDBDBD" for v in a["Chuva (mm)"]]
    ax2.bar(np.arange(len(a)), a["Chuva (mm)"], 0.75, color=cores)
    ax2.axhline(med, color="#333333", lw=0.8, ls="--")
    ax2.text(len(a) - 0.5, med, f" mediana {med:,.0f} mm".replace(",", "."), fontsize=6.5, va="bottom", ha="right")
    step = max(1, int(np.ceil(len(a) / 8)))
    ax2.set_xticks(np.arange(len(a))[::step], a["Ano agrícola"].iloc[::step], fontsize=6.4, rotation=0)
    ax2.set_ylabel("mm/ano agrícola", fontsize=7.5)
    _eixos_limpos(ax2)
    _subtitulo(fig, ctx, 0.09, 0.575, "Chuva por ano agrícola (jul–jun)")
    _legenda(fig, 0.58, 0.575, [("#D6604D", "20% mais secos"), ("#4393C3", "20% mais chuvosos")], colunas=2,
             larg_col=0.16, fs=6.6)
    # safra atual
    if C.atual:
        at = C.atual
        ax3 = fig.add_axes([0.12, 0.125, 0.5, 0.155])
        dias = np.arange(len(at["p50"]))
        ax3.fill_between(dias, at["p10"], at["p90"], color="#9ECAE1", alpha=0.5, lw=0, label="10% a 90% dos anos")
        ax3.plot(dias, at["p50"], color="#08519C", lw=1, ls="--", label="mediana")
        ax3.plot(np.arange(len(at["curva_atual"])), at["curva_atual"], color=ac, lw=2, label=f"safra {at['ano']}")
        ax3.set_xticks([0, 92, 184, 273, 364], ["jul", "out", "jan", "abr", "jun"], fontsize=7)
        ax3.set_ylabel("mm acumulados", fontsize=7.5)
        _eixos_limpos(ax3)
        ax3.legend(fontsize=6.4, frameon=False, loc="upper left")
        _subtitulo(fig, ctx, 0.09, 0.305, "Safra atual × histórico (acumulado desde 1º de julho)")
        y = 0.27
        for t in (f"Até {at['ate']:%d/%m/%Y}: {br(at['acumulado'], 0)} mm",
                  f"{at['pct_normal']:.0f}% da mediana do período",
                  f"Mais chuvoso que {at['percentil']:.0f}% dos anos",
                  "Anos mais secos no mesmo período: " + ", ".join(C.anos_extremos.get("secos", [])),
                  "Mais chuvosos: " + ", ".join(C.anos_extremos.get("umidos", []))):
            y = _texto(fig, 0.66, y, t, larg=40, fs=7.2) - 0.004
    fig.text(0.09, 0.075, f"Fonte: {C.fonte_chuva}. Grade regional — talhões vizinhos compartilham a mesma célula; "
                          "não é chuva medida no talhão.", fontsize=6.3, color="#777777")
    _fechar(pdf, fig)


def pagina_clima_2(pdf, ctx, num, diag: Diagnostico):
    C = diag.clima
    fig = _pagina(pdf, ctx, num, "RISCOS CLIMÁTICOS", "veranicos, estação chuvosa e temperaturas extremas")
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    if not C.veranicos.empty:
        v = C.veranicos
        ax = fig.add_axes([0.12, 0.62, 0.4, 0.18])
        x = np.arange(len(v))
        ax.bar(x - 0.18, v["≥ 10 dias secos (%)"], 0.35, color=ac, label="≥ 10 dias")
        ax.bar(x + 0.18, v["≥ 15 dias secos (%)"], 0.35, color=bk._mistura(ac, "black", 0.35), label="≥ 15 dias")
        for i, val in enumerate(v["≥ 10 dias secos (%)"]):
            ax.text(i - 0.18, val + 1.5, f"{val:.0f}", ha="center", fontsize=6.2)
        ax.set_xticks(x, v["Mês"], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% dos anos", fontsize=7.5)
        _eixos_limpos(ax)
        ax.legend(fontsize=6.5, frameon=False)
        _subtitulo(fig, ctx, 0.09, 0.825, "Veranicos na estação chuvosa")
        _texto(fig, 0.56, 0.8, "Frequência histórica de anos com ao menos uma sequência de dias secos (chuva < 1 mm) "
                               "começando no mês. Ex.: “em quantos anos houve 10 ou mais dias secos seguidos em "
                               "janeiro?”. É estatística do passado, não previsão.", larg=48, fs=7, cor=CINZA)
    est = cl.texto_estacao(C)
    if est:
        ax2 = fig.add_axes([0.12, 0.425, 0.78, 0.11])
        for k, (col, corl) in enumerate((("Início", "#2B7BB9"), ("Fim", "#D6604D"))):
            if col in est:
                vals = est[col]["valores"]
                ax2.scatter(vals, np.full(len(vals), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(vals)),
                            s=10, color=corl, alpha=0.5, lw=0)
                ax2.plot([np.percentile(vals, 25), np.percentile(vals, 75)], [k, k], color=corl, lw=5, alpha=0.4)
                ax2.plot(np.median(vals), k, marker="|", ms=16, color=corl, mew=2.5)
        ax2.set_yticks([0, 1], ["Início", "Fim"], fontsize=7.5)
        ticks = [62, 92, 123, 153, 184, 215, 243, 274, 304]
        ax2.set_xticks(ticks, ["set", "out", "nov", "dez", "jan", "fev", "mar", "abr", "mai"], fontsize=7)
        ax2.set_ylim(-0.6, 1.6)
        _eixos_limpos(ax2)
        _subtitulo(fig, ctx, 0.09, 0.56, "Início e fim da estação chuvosa")
        txt = []
        if "Início" in est:
            txt.append(f"Início típico {est['Início']['mediana']} (metade dos anos entre {est['Início']['p25']} e "
                       f"{est['Início']['p75']})")
        if "Fim" in est:
            txt.append(f"fim típico {est['Fim']['mediana']} ({est['Fim']['p25']} a {est['Fim']['p75']})")
        fig.text(0.12, 0.39, "; ".join(txt) + ".", fontsize=7.2, color="#222222")
        fig.text(0.12, 0.375, "Início: ≥ 20 mm em 3 dias sem 10 dias secos nos 30 seguintes (a partir de 1/set). "
                              "Fim: 20 dias com menos de 10 mm (a partir de 1/mar).", fontsize=6.3, color=CINZA)
    if not C.extremos.empty:
        e = C.extremos
        ax3 = fig.add_axes([0.14, 0.12, 0.70, 0.17])       # mais estreito: o eixo de °C fica à direita, dentro da moldura
        x = np.arange(len(e))
        ax3.bar(x - 0.2, e["Dias ≥ 35 °C (por ano)"], 0.38, color="#D6604D", label="dias com máx ≥ 35 °C")
        ax3.bar(x + 0.2, e["Dias ≤ 5 °C (por ano)"], 0.38, color="#4393C3", label="dias com mín ≤ 5 °C")
        ax3.set_xticks(x, e["Mês"], fontsize=7)
        ax3.set_ylabel("dias por ano (média)", fontsize=7.5)
        _eixos_limpos(ax3)
        ax3b = ax3.twinx()
        ax3b.plot(x, e["Tmáx média (°C)"], color="#B2182B", lw=1.2, marker="o", ms=2.5)
        ax3b.plot(x, e["Tmín média (°C)"], color="#2166AC", lw=1.2, marker="o", ms=2.5)
        ax3b.set_ylabel("°C (médias)", fontsize=7.5, labelpad=3)
        ax3b.tick_params(labelsize=7, pad=2)
        for s in ("top",):
            ax3b.spines[s].set_visible(False)
        ax3.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.2))
        _subtitulo(fig, ctx, 0.09, 0.335, "Calor e frio extremos")
        geada = e["Dias ≤ 2 °C (por ano)"].sum()
        fig.text(0.12, 0.09, f"Dias com mínima ≤ 2 °C (risco de geada, temperatura a 2 m): {br(geada, 1)} por ano "
                             f"em média. Fonte: {C.fonte_temp}.", fontsize=6.4, color=CINZA)
    _fechar(pdf, fig)


# ------------------------------------------------------------------ solo
def pagina_solo(pdf, ctx, num, diag: Diagnostico):
    """Só granulometria (argila, silte, areia) e grupo textural — o que o SoilGrids estima com mais segurança."""
    from . import solo as so
    S = diag.solo
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    fig = _pagina(pdf, ctx, num, "CONTEXTO DO SOLO", "granulometria estimada · SoilGrids 2.0 (ISRIC) · 250 m")
    t = S.tabela
    cam = S.camada
    # --- cartão com o grupo textural
    if "clay" in cam:
        a, a5, a95 = cam["clay"]
        grupo = so.grupo_textural(a)
        fig.add_artist(FancyBboxPatch((0.09, 0.705), 0.3, 0.115, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=bk._mistura(cor, "white", 0.9), ec="none"))
        fig.add_artist(Rectangle((0.09, 0.715), 0.006, 0.095, transform=fig.transFigure, fc=ac, ec="none"))
        fig.text(0.11, 0.8, "GRUPO TEXTURAL ESTIMADO", fontsize=7, family=F_SEMI, color=ac, va="center")
        fig.text(0.11, 0.772, grupo, fontsize=17, family=F_XB, color=cor, va="center")
        fig.text(0.11, 0.742, f"argila ≈ {a:.0f}% na camada 0–30 cm", fontsize=8.2, color="#222222", va="center")
        fig.text(0.11, 0.724, f"faixa provável {a5:.0f}% a {a95:.0f}% (90%)", fontsize=7, color=CINZA, va="center")
        # régua de argila com os grupos Embrapa
        x0, x1, yr = 0.45, 0.9, 0.748
        esc = lambda v: x0 + (x1 - x0) * v / 100  # noqa: E731
        fig.text(x0, 0.8, "Teor de argila (0–30 cm) e grupos texturais", fontsize=8, family=F_SEMI, color=cor,
                 va="center")
        for k, (nome, lo, hi, c) in enumerate(so.GRUPOS):
            fig.add_artist(Rectangle((esc(lo), yr - 0.011), esc(hi) - esc(lo), 0.022, transform=fig.transFigure,
                                     fc=c, ec="white", lw=1.2))
            fig.text((esc(lo) + esc(hi)) / 2, yr, nome, ha="center", va="center", fontsize=6.3,
                     color="white" if k >= 2 else "#333333", family=F_SEMI)
        for v in (0, 15, 35, 60, 100):
            fig.text(esc(v), yr + 0.019, f"{v}%", ha="center", va="center", fontsize=6, color=CINZA)
        yb = yr - 0.02
        kw = dict(transform=fig.transFigure, color="#111111", lw=0.9)
        fig.add_artist(plt.Line2D([esc(a5), esc(a95)], [yb, yb], **kw))
        for xx in (esc(a5), esc(a95)):
            fig.add_artist(plt.Line2D([xx, xx], [yb - 0.004, yb + 0.004], **kw))
        fig.add_artist(plt.Line2D([esc(a)], [yr - 0.017], transform=fig.transFigure, marker="^", ms=7,
                                  color="#111111", lw=0))
        fig.text(esc(a), yb - 0.015, f"{a:.0f}%", ha="center", va="center", fontsize=8.5, family=F_BOLD,
                 weight="bold", color="#111111")
        fig.text(esc(a95) + 0.008, yb, "faixa provável", ha="left", va="center", fontsize=5.8, color=CINZA)
    # --- granulometria por profundidade (barras empilhadas)
    _subtitulo(fig, ctx, 0.09, 0.655, "Argila, silte e areia por profundidade")
    profs = [p for p in ["0-5cm", "5-15cm", "15-30cm", "30-60cm"] if p in set(t["profundidade"])]
    cores = {"clay": "#A0522D", "silt": "#C9A66B", "sand": "#EFD9A7"}
    ax = fig.add_axes([0.16, 0.44, 0.44, 0.195])
    for k, p in enumerate(profs):
        esq = 0.0
        vals = {}
        for prop in ("clay", "silt", "sand"):
            sub = t[(t["propriedade"] == prop) & (t["profundidade"] == p)]
            vals[prop] = float(sub["media"].iloc[0]) if len(sub) else 0.0
        tot = sum(vals.values()) or 1.0
        for prop in ("clay", "silt", "sand"):
            w = 100 * vals[prop] / tot                          # normaliza para 100% (predições independentes)
            ax.barh(k, w, left=esq, height=0.62, color=cores[prop], ec="white", lw=0.8)
            if w >= 7:
                ax.text(esq + w / 2, k, f"{w:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if prop == "clay" else "#333333", family=F_SEMI)
            esq += w
    ax.set_yticks(range(len(profs)), [p.replace("-", "–").replace("cm", " cm") for p in profs], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], fontsize=6.8)
    _eixos_limpos(ax, 7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _legenda(fig, 0.16, 0.415, [(cores["clay"], "Argila"), (cores["silt"], "Silte"), (cores["sand"], "Areia")],
             colunas=3, larg_col=0.1, fs=7)
    # --- mapa com o tamanho real do pixel do SoilGrids
    ax2 = _mapa(fig, ctx, [0.66, 0.43, 0.25, 0.21], diag.area, lim=_limites(diag.area, 0.35))
    ax2.add_artist(bk._patch(diag.area.poligono, fc=bk._mistura(cor, "white", 0.85), ec="none", zorder=1))
    from ..geo import projetar
    from shapely.geometry import Point
    for lon, lat in S.pontos:
        try:
            pt = projetar(Point(lon, lat), diag.area.epsg)
        except Exception:  # noqa: BLE001
            continue
        ax2.add_artist(Rectangle((pt.x - 125, pt.y - 125), 250, 250, fc="none", ec=ac, lw=1.4, zorder=7))
        ax2.plot(pt.x, pt.y, "o", ms=3, color=ac, zorder=8)
    fig.text(0.785, 0.418, "□ pixel do SoilGrids (250 × 250 m)", ha="center", fontsize=6.3, color=CINZA)
    # --- leitura
    y = 0.37
    _subtitulo(fig, ctx, 0.09, y, "Como usar")
    y -= 0.022
    itens = [
        "A análise granulométrica do laboratório sempre prevalece. Quando o laudo não traz argila, a aba "
        "Recomendação estima pela CTC; esta página serve como segunda referência de ordem de grandeza.",
        "Os grupos texturais (arenosa < 15%, média 15–35%, argilosa 35–60%, muito argilosa > 60% de argila) "
        "orientam a interpretação de P e K, a dose de gesso e a capacidade de retenção de água.",
        "Cada pixel cobre 6,25 ha: a variação de textura DENTRO do talhão não aparece. Para isso, use a amostragem "
        "em grade (aba Recomendação e book) ou mapas de condutividade elétrica.",
        f"Estimativa de {len(S.pontos)} ponto(s) do talhão; faixa provável = intervalo de 90% do modelo. "
        "Carbono, CTC, pH, N e densidade não são mostrados: variam muito com o manejo e as predições globais "
        "ficam distantes das lavouras."]
    for it_ in itens:
        fig.text(0.1, y, "›", fontsize=10, color=ac, family=F_BOLD, va="top")
        y = _texto(fig, 0.12, y, it_, larg=108, fs=7.6) - 0.008
    _fechar(pdf, fig)


# ------------------------------------------------------------------ temperatura e vento
CMAP_LST = LinearSegmentedColormap.from_list("lst", ["#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C"])


def _seta_vento(ax, graus_de, cor="#111111"):
    """Seta no canto do mapa apontando PARA onde o vento sopra (graus_de = de onde vem)."""
    rad = np.deg2rad((graus_de + 180) % 360)
    dx, dy = np.sin(rad), np.cos(rad)
    cx, cy, L = 0.1, 0.88, 0.07
    ax.annotate("", xy=(cx + dx * L, cy + dy * L * 0.9), xytext=(cx - dx * L, cy - dy * L * 0.9),
                xycoords="axes fraction", zorder=9,
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6", color=cor, lw=2.2))
    ax.annotate("vento", xy=(cx, cy), xycoords="axes fraction", xytext=(0, -17), textcoords="offset points",
                ha="center", va="top", fontsize=6, color=cor, zorder=9)


def _rosa(fig, rect, V, ctx):
    from . import vento as vt
    ax = fig.add_axes(rect, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    th = np.deg2rad(np.arange(16) * 22.5)
    base = np.zeros(16)
    for (nome, _, _, c) in vt.CLASSES:
        v = V.rosa[nome].values
        ax.bar(th, v, width=np.deg2rad(20), bottom=base, color=c, ec="white", lw=0.4)
        base += v
    ax.set_xticks(th[::2], [vt.SETORES[k] for k in range(0, 16, 2)], fontsize=6.8)
    ax.tick_params(axis="x", pad=-1)
    mx = base.max()
    passo = 5 if mx <= 25 else 10
    ticks = np.arange(passo, mx + passo, passo)
    ax.set_yticks(ticks, [f"{t:.0f}%" for t in ticks], fontsize=5.5, color=CINZA)
    calmo = int(np.argmin(np.convolve(np.r_[base[-1], base, base[0]], np.ones(3), "valid")))
    ax.set_rlabel_position(calmo * 22.5 + 11)
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.spines["polar"].set_color("#BBBBBB")
    return ax


def _janelas(fig, rect, V):
    from . import vento as vt
    ax = fig.add_axes(rect)
    ordem = cl.ORDEM_AGRICOLA
    M = V.janela.loc[ordem].values
    cmap = LinearSegmentedColormap.from_list("jan", ["#F7F7F7", "#C7E9C0", "#74C476", "#238B45", "#00441B"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(12), [vt.MESES[m - 1] for m in ordem], fontsize=6.4)
    hs = list(V.janela.columns)
    ax.set_xticks(range(0, len(hs), 2), [f"{h}h" for h in hs[::2]], fontsize=6.4)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    for (r, c), v in np.ndenumerate(M):
        if np.isfinite(v) and v >= 60:
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=4.6, color="white")
    return ax, cmap


def pagina_temp_vento(pdf, ctx, num, diag: Diagnostico):
    from . import termico as tm
    from . import vento as vt
    T = diag.termico
    V = diag.clima.vento if diag.clima is not None else None
    A = diag.area.area_ha
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    partes = []
    if T is not None:
        partes.append(f"Landsat 8/9 ({len(T.cenas)} passagens)")
    if V is not None:
        partes.append(f"vento NASA POWER {V.periodo[0]}–{V.periodo[1]}")
    fig = _pagina(pdf, ctx, num, "TEMPERATURA E VENTO" if T is not None else "VENTO E PULVERIZAÇÃO",
                  " · ".join(partes))
    y_topo = 0.815
    if T is not None:
        fino, ext, res = tm.superficie(T)
        rs = tm.resumo_superficie(T, fino, ext, res, diag.area)
        lim = max(0.8, round(max(abs(rs["p05"]), abs(rs["p95"])) * 1.1, 1))
        ax = _mapa(fig, ctx, [0.09, 0.505, 0.5, 0.31], diag.area)
        x0, x1, y0, y1 = ext
        h, w = fino.shape
        X = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
        Y = y1 - (np.arange(h) + 0.5) * (y1 - y0) / h
        niveis = np.linspace(-lim, lim, 17)
        cf = ax.contourf(X, Y, np.clip(fino, -lim, lim), levels=niveis, cmap=CMAP_LST, zorder=2, extend="both")
        cl_ = ax.contour(X, Y, fino, levels=[l for l in np.arange(-3, 3.01, 0.5) if abs(l) > 1e-9 and abs(l) < lim],
                         colors="#333333", linewidths=0.35, alpha=0.55, zorder=3)
        pp = bk._patch(diag.area.poligono, fc="none", ec="none")
        ax.add_artist(pp)
        for col in (cf, cl_):
            col.set_clip_path(pp)
        if V is not None and not V.rosa.empty:
            _seta_vento(ax, V.graus_predominante)
        bk.escala(fig, ax, ax.get_position().x0, 0.49, frac=0.1, fs=6.8)
        bk.norte(fig, 0.53, 0.478, 0.03)
        _subtitulo(fig, ctx, 0.62, 0.8, "Temperatura da superfície")
        y = _texto(fig, 0.62, 0.78, "Média das passagens sem nuvens (~10h30), em relação à mediana do talhão. "
                                    "Vermelho = relativamente mais quente.", larg=44, fs=6.8, cor=CINZA)
        ax_cb = fig.add_axes([0.62, y - 0.02, 0.26, 0.013])
        ax_cb.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP_LST, aspect="auto")
        ax_cb.set_xticks([0, 128, 255], [f"−{br(lim, 1)}", "0", f"+{br(lim, 1)} °C"], fontsize=6.5)
        ax_cb.set_yticks([])
        for s_ in ax_cb.spines.values():
            s_.set_visible(False)
        y -= 0.065
        linhas = [(f"{br(T.media_c, 1)} °C", "temperatura média do talhão nas passagens"),
                  (f"{br(rs['amplitude'], 1)} °C", "diferença entre as partes mais quentes e mais frias (P95–P5)"),
                  (f"{br(rs['quente_ha'], 1)} ha", f"≥ 1 °C acima da mediana ({_pct(rs['quente_ha'], A):.0f}% da área)")]
        for val, txt in linhas:
            fig.text(0.62, y, val, fontsize=10.5, family=F_BOLD, weight="bold", color=cor, va="center")
            y = _texto(fig, 0.62, y - 0.014, txt, larg=44, fs=6.6, cor=CINZA, entre=0.011) - 0.012
        if V is not None and not V.rosa.empty:
            fig.text(0.62, y - 0.004, "➜", fontsize=9, color="#111111", va="center")
            _texto(fig, 0.645, y + 0.002, f"seta no mapa: vento predominante, de {vt.por_extenso(V.predominante)} "
                                          "(regional, igual em todo o talhão)", larg=38, fs=6.4, cor=CINZA,
                   entre=0.0105)
        y_topo = 0.448
    if V is not None:
        # rosa dos ventos
        if not V.rosa.empty:
            _subtitulo(fig, ctx, 0.09, y_topo, "Rosa dos ventos")
            fig.text(0.09, y_topo - 0.016, "% do tempo, de onde o vento sopra (vento a 2 m)", fontsize=6.5,
                     color=CINZA, va="center")
            _rosa(fig, [0.1, y_topo - 0.225, 0.3, 0.17], V, ctx)
            _legenda(fig, 0.1, y_topo - 0.252, [(c, n) for n, _, _, c in vt.CLASSES], colunas=4, larg_col=0.1,
                     fs=6.2, passo=0.016)
        # janelas de pulverização
        xj = 0.5
        if V.janela is not None:
            _subtitulo(fig, ctx, xj, y_topo, "Janelas de pulverização")
            c = vt.CRITERIOS
            fig.text(xj, y_topo - 0.016, f"% das horas com vento {c['vento_min']:.0f}–{c['vento_max']:.0f} km/h, "
                                         f"< {c['temp_max']:.0f} °C e UR > {c['ur_min']:.0f}%", fontsize=6.5,
                     color=CINZA, va="center")
            axj, cmj = _janelas(fig, [xj + 0.045, y_topo - 0.225, 0.36, 0.185], V)
            ax_cb = fig.add_axes([xj + 0.045, y_topo - 0.247, 0.18, 0.008])
            ax_cb.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmj, aspect="auto")
            ax_cb.set_xticks([0, 255], ["0%", "100% das horas"], fontsize=5.8)
            ax_cb.set_yticks([])
            for s_ in ax_cb.spines.values():
                s_.set_visible(False)
        # números-chave
        yk = y_topo - 0.272
        chave = []
        if not V.rosa.empty:
            ext_ = vt.por_extenso(V.predominante)
            chave.append(f"Vento predominante de {ext_}" + (f" ({V.predominante})" if ext_ != V.predominante else "")
                         + f" em {V.predominante_pct:.0f}% do tempo; média de {br(V.media_kmh, 1)} km/h a 2 m.")
        if not V.mensal.empty:
            mx = V.mensal.sort_values("Vento médio (km/h)", ascending=False).iloc[:2]
            chave.append(f"Meses mais ventosos: {' e '.join(mx['Mês'].str.lower())}. Calmaria (< 3 km/h) em "
                         f"{V.calmaria_pct:.0f}% do tempo: vento fraco também é ruim (inversão térmica, deriva).")
        mh = vt.melhores_horas(V)
        if mh:
            chave.append(f"Out–mar: melhores horários (mais da metade das horas adequadas): {mh}.")
        for t_ in chave:
            fig.text(0.1, yk, "›", fontsize=10, color=ac, family=F_BOLD, va="top")
            yk = _texto(fig, 0.12, yk, t_, larg=125, fs=7, entre=0.0135) - 0.003
    _texto(fig, 0.09, 0.1, ("Temperatura: sensor térmico do Landsat (100 m), suavizada; temperatura alta, sozinha, não "
                              "comprova falta de água — solo exposto, palhada e relevo também aquecem a superfície. "
                              if T is not None else "") +
           ("Vento, temperatura do ar e umidade: NASA POWER (~50 km, hora local), iguais para toda a região; "
            "quebra-ventos, matas e relevo alteram o vento no campo. Confirme sempre na hora da aplicação."
            if V is not None else ""), larg=135, fs=6.0, cor="#777777", entre=0.0105)
    _fechar(pdf, fig)


# ------------------------------------------------------------------ métodos
# ------------------------------------------------------------------ água no solo
CMAP_UMIDADE = LinearSegmentedColormap.from_list("umid", ["#8C510A", "#D8B365", "#F6E8C3", "#9FD3E6", "#3F8FC4",
                                                          "#0B3C7A"])


def _cartao_agua(fig, ctx, x, y, larg, rotulo, valor, nota=""):
    fig.add_artist(FancyBboxPatch((x, y), larg, 0.043, boxstyle="round,pad=0,rounding_size=0.008",
                                  transform=fig.transFigure, fc="#EEF3EE", ec="none"))
    fig.text(x + 0.01, y + 0.033, rotulo, fontsize=6.3, color=CINZA, va="center")
    fig.text(x + 0.01, y + 0.018, valor, fontsize=9.6, family=F_BOLD, weight="bold", color=ctx.marca["cor"],
             va="center")
    if nota:
        fig.text(x + 0.01, y + 0.006, nota, fontsize=5.4, color="#777777", va="center")


def pagina_agua(pdf, ctx, num, diag: Diagnostico):
    """Água no solo: balanço hídrico dos últimos 90 dias (dados públicos; armazenamento só com os parâmetros do
    usuário) e umidade superficial NISAR (produto oficial). Os dois não são misturados."""
    from pyproj import Transformer

    from . import agua as ag
    from . import nisar as ns
    W = diag.agua
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    s, r, par = W.serie, W.resumo, W.par
    fig = _pagina(pdf, ctx, num, "ÁGUA NO SOLO", f"balanço hídrico de {r['dias']} dias · umidade superficial NISAR")

    # ---- origem das informações
    y = 0.853
    fig.text(0.09, y, "DADOS PÚBLICOS", fontsize=6.6, family=F_SEMI, color=ac, va="center")
    fig.text(0.56, y, "INFORMADO PELO USUÁRIO", fontsize=6.6, family=F_SEMI, color=ac, va="center")
    inf = par.informados()
    so_irrig = len(inf) == 1 and par.irrigacao == "nao_informado"
    txt_inf = ("Nenhum parâmetro de cultura ou solo informado · irrigação não informada." if so_irrig
               else " · ".join(f"{a}: {b}" for a, b in inf) + ".")
    bk.texto_corrido(fig, 0.09, y - 0.013, " · ".join(W.fontes or ["—"]) + ".", larg=0.44, y_min=0.818, fs=6.0,
                     fs_min=5.2, cor="#444444", entre=1.32)
    bk.texto_corrido(fig, 0.56, y - 0.013, txt_inf, larg=0.34, y_min=0.818, fs=6.0, fs_min=5.2, cor="#444444",
                     entre=1.32)

    # ---- A. chuva, ET0 e saldo
    _subtitulo(fig, ctx, 0.09, 0.800, f"Chuva e evapotranspiração de referência (ET₀) · {W.inicio:%d/%m/%Y} a "
                                      f"{W.fim:%d/%m/%Y}", fs=9.5)
    ax = fig.add_axes([0.14, 0.652, 0.755, 0.128])
    x = np.arange(len(s))
    ch = s["chuva"].to_numpy(float)
    chirps = (s["fonte_chuva"] == "CHIRPS").to_numpy()
    ax.bar(x[chirps], ch[chirps], width=0.85, color="#2C7FB8", label="chuva CHIRPS")
    if (~chirps & np.isfinite(ch)).any():
        ax.bar(x[~chirps & np.isfinite(ch)], ch[~chirps & np.isfinite(ch)], width=0.85, color="#9ECAE1",
               label="chuva NASA POWER")
    ax.plot(x, s["et0"].to_numpy(float), color="#E6550D", lw=1.3, label="ET₀")
    topo = max(np.nanmax(ch) if np.isfinite(ch).any() else 1, np.nanmax(s["et0"]) if s["et0"].notna().any() else 1)
    sem = ~np.isfinite(ch) | s["et0"].isna().to_numpy()
    if sem.any():
        ax.bar(x[sem], np.full(sem.sum(), topo * 1.3), width=1.0, color="#DDDDDD", alpha=0.6, label="sem dado",
               zorder=0)
    ax.set_xlim(-0.8, len(s) - 0.2)
    ax.set_ylim(0, topo * 1.3)
    marcas = [i for i, d in enumerate(s.index) if d.day in (1, 15)]
    ax.set_xticks(marcas, [f"{s.index[i]:%d/%m}" for i in marcas])
    ax.set_ylabel("mm/dia", fontsize=7)
    _eixos_limpos(ax, 6.8)
    ax.legend(fontsize=6.0, ncol=4, loc="upper right", handlelength=1.4, columnspacing=1.1, frameon=True,
              framealpha=0.9, edgecolor="none", borderpad=0.3)
    f1 = lambda v: "—" if not np.isfinite(v) else f"{br(v, 0)} mm"                       # noqa: E731
    sinal = "" if not np.isfinite(r["saldo"]) else ("+" if r["saldo"] >= 0 else "−")
    cart = [("Chuva", f1(r["chuva"]), f"{r['dias_chuva']} de {r['dias']} dias com dado"),
            ("ET₀", f1(r["et0"]), f"{r['dias_et0']} de {r['dias']} dias com dado"),
            ("Saldo P − ET₀", "—" if not np.isfinite(r["saldo"]) else f"{sinal}{br(abs(r['saldo']), 0)} mm",
             f"soma dos {r['dias_saldo']} dias com chuva e ET₀"),
            ("Última chuva ≥ 1 mm", f"{r['ultima_chuva']:%d/%m/%Y}" if r.get("ultima_chuva") else "—",
             f"há {(W.fim - r['ultima_chuva']).days} dia(s)" if r.get("ultima_chuva") else "nenhuma na janela")]
    for i, (a, b, c) in enumerate(cart):
        _cartao_agua(fig, ctx, 0.09 + i * 0.2075, 0.582, 0.195, a, b, c)
    _texto(fig, 0.09, 0.574, "O saldo P − ET₀ compara a chuva com a demanda da atmosfera sobre uma superfície de "
                             "referência: é um indicador climático. Sozinho, NÃO é a água disponível às plantas — isso "
                             "depende do solo, da cultura, das raízes e da irrigação.", larg=128, fs=6.5, cor=CINZA,
           entre=0.0115)

    # ---- B. armazenamento na zona radicular
    _subtitulo(fig, ctx, 0.09, 0.532, "Armazenamento de água na zona radicular", fs=9.5)
    arm = W.armazenamento
    if arm is None:
        fig.add_artist(FancyBboxPatch((0.09, 0.405), 0.82, 0.105, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc="#F6F6F3", ec="#CCCCCC", lw=0.6))
        fig.text(0.11, 0.49, "Não calculado", fontsize=10, family=F_BOLD, weight="bold", color=cor, va="center")
        _texto(fig, 0.11, 0.476, (f"Motivo — {W.motivo_sem_armazenamento}. " if W.motivo_sem_armazenamento else "")
               + "O armazenamento só é estimado quando o usuário informa cultura, data de semeadura, profundidade "
                 "radicular e capacidade de água disponível (CAD). Sem esses parâmetros, qualquer número de “água "
                 "disponível” seria um palpite: ficam apenas a chuva, a ET₀ e o saldo acima.",
               larg=118, fs=7.2, cor="#444444", entre=0.0135)
        y_c = 0.385
    else:
        k = W.kc_info
        ax2 = fig.add_axes([0.14, 0.405, 0.755, 0.105])
        xa = np.arange(len(arm))
        ax2.fill_between(xa, arm["arm_min"], arm["arm_max"], color="#9ECAE1", alpha=0.55, lw=0,
                         label="faixa da condição inicial (solo seco ↔ cheio)")
        ax2.plot(xa, arm["arm"], color="#08519C", lw=1.5, label="armazenamento (mm)")
        lim = (1 - k["p"]) * k["cad_total"]
        ax2.axhline(k["cad_total"], color="#555555", lw=0.7, ls=":")
        ax2.axhline(lim, color="#CB181D", lw=0.8, ls="--")
        ax2.text(len(arm) - 0.5, k["cad_total"], f" CAD {br(k['cad_total'], 0)} mm", fontsize=5.8, va="bottom",
                 ha="right", color="#555555")
        ax2.text(0.2, lim - k["cad_total"] * 0.02, "abaixo desta linha: estresse hídrico simulado (Ks < 1)",
                 fontsize=5.8, va="top", ha="left", color="#CB181D")
        irr = arm["irrigacao"].to_numpy(float)
        if (irr > 0).any():
            ax2.scatter(xa[irr > 0], np.full((irr > 0).sum(), k["cad_total"] * 1.04), marker="v", s=14,
                        color="#238B45", label="irrigação informada", zorder=5)
        ax2.set_xlim(-0.8, len(arm) - 0.2)
        ax2.set_ylim(0, k["cad_total"] * 1.14)
        marcas = [i for i, d in enumerate(arm.index) if d.day in (1, 15)]
        ax2.set_xticks(marcas, [f"{arm.index[i]:%d/%m}" for i in marcas])
        ax2.set_ylabel("mm", fontsize=7)
        _eixos_limpos(ax2, 6.8)
        ax2.legend(fontsize=5.8, ncol=3, loc="lower left", handlelength=1.4, frameon=True, framealpha=0.9,
                   edgecolor="none", borderpad=0.3)
        ult = arm.iloc[-1]
        das = int(ult["dias_apos_semeadura"])
        fase = (f"{das} dias após a semeadura (fase {ult['fase']}, Kc {br(ult['kc'], 2)})" if ult["fase"] !=
                "sem cultura" else f"fora do ciclo da cultura (Kc {br(ult['kc'], 2)})")
        _texto(fig, 0.09, 0.378,
               f"CENÁRIO: {W.cenario}. Em {W.fim:%d/%m}: {br(ult['arm'], 0)} mm ({br(ult['arm_pct'], 0)}% da CAD de "
               f"{br(k['cad_total'], 0)} mm), {fase}. Na janela: {k['dias_estresse']} dia(s) com estresse simulado, "
               f"consumo {br(k['eta'], 0)} mm de {br(k['etc'], 0)} mm possíveis, excedente (drenagem/escoamento) "
               f"{br(k['excedente'], 0)} mm. {k['cultura']}: Kc {br(k['kc'][0], 2)}/{br(k['kc'][1], 2)}/"
               f"{br(k['kc'][2], 2)}, fases de {'/'.join(str(v) for v in k['fases_dias'])} dias e p = "
               f"{br(k['p'], 2)} (FAO-56, valores genéricos). Faixa da condição inicial no fim: "
               f"{br(k['faixa_fim'], 0)} mm" + (f" · {k['dias_sem_dado']} dia(s) sem dado mantidos sem atualização"
                                                 if k["dias_sem_dado"] else "") + ". Resultado de simulação, não medida.",
               larg=132, fs=6.3, cor="#444444", entre=0.0112)
        y_c = 0.385

    # ---- C. umidade superficial NISAR
    N = W.nisar
    _subtitulo(fig, ctx, 0.09, 0.312, "Umidade superficial do solo · NISAR (NASA/ISRO)", fs=9.5)
    u = N.ultima_valida if N is not None else None
    if u is None:
        fig.add_artist(FancyBboxPatch((0.09, 0.185), 0.82, 0.105, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc="#F6F6F3", ec="#CCCCCC", lw=0.6))
        fig.text(0.11, 0.27, ns.SEM_OBS, fontsize=10, family=F_BOLD, weight="bold", color=cor, va="center")
        msg = (N.mensagem if N is not None else "").replace(ns.SEM_OBS + " — ", "")
        _texto(fig, 0.11, 0.256, (msg[:1].upper() + msg[1:] + " " if msg else "")
               + f"Produto consultado: {ns.PRODUTO}, 200 m, {ns.MATURIDADE}. O NDMI e outros índices de vegetação "
                 "não medem umidade do solo e não são usados como substitutos.",
               larg=118, fs=7.2, cor="#444444", entre=0.0135)
    else:
        g = u.grade
        t = Transformer.from_crs(ns.EPSG_GRADE, diag.area.epsg, always_xy=True)
        validos = g["sm"][g["dentro"] & g["valido"]]
        lo = float(np.floor(max(0.0, validos.min() - 0.02) * 50) / 50)
        hi = float(np.ceil(min(0.6, validos.max() + 0.02) * 50) / 50)
        norm = matplotlib.colors.Normalize(lo, max(hi, lo + 0.04))
        ax3 = bk._caixa_mapa(fig, ctx, [0.09, 0.105, 0.27, 0.185], _limites(diag.area, 0.22))
        h = g["res"] / 2
        for i in range(len(g["y"])):
            for j in range(len(g["x"])):
                xs_, ys_ = t.transform([g["x"][j] - h, g["x"][j] + h, g["x"][j] + h, g["x"][j] - h],
                                       [g["y"][i] - h, g["y"][i] - h, g["y"][i] + h, g["y"][i] + h])
                ok = g["valido"][i, j]
                ax3.add_artist(plt.Polygon(list(zip(xs_, ys_)), fc=CMAP_UMIDADE(norm(g["sm"][i, j])) if ok else "#E3E3E3",
                                           ec="white", lw=0.5, alpha=1.0 if g["dentro"][i, j] else 0.38,
                                           hatch=None if ok else "///", zorder=2))
        ax3.add_artist(bk._patch(diag.area.poligono, fc="none", ec="#111111", lw=1.1, zorder=5))
        axc = fig.add_axes([0.105, 0.097, 0.21, 0.007])
        axc.imshow(np.linspace(0, 1, 128)[None, :], cmap=CMAP_UMIDADE, aspect="auto")
        axc.set_xticks([0, 127], [br(norm.vmin, 2), f"{br(norm.vmax, 2)} m³/m³"], fontsize=5.6)
        axc.set_yticks([])
        axc.tick_params(length=0, pad=1.5)
        for sp in axc.spines.values():
            sp.set_visible(False)
        x2 = 0.39
        fig.text(x2, 0.283, "Observação válida mais recente", fontsize=6.6, color=CINZA, va="center")
        fig.text(x2, 0.262, f"{br(u.mediana, 2)} m³/m³", fontsize=15, family=F_XB, color=cor, va="center")
        fig.text(x2, 0.243, f"= {br(100 * u.mediana, 0)}% do volume do solo · mediana dos pixels válidos "
                            f"(P10–P90: {br(u.p10, 2)}–{br(u.p90, 2)})", fontsize=6.2, color="#444444", va="center")
        linhas = [("Data", f"{u.data:%d/%m/%Y %H:%M} UTC · {u.passagem}"),
                  ("Resolução", "200 m (grade EASE-2, nativa do produto)"),
                  ("Cobertura válida", f"{u.n_validos} de {u.n_pixels} pixels do talhão ({br(u.cobertura, 0)}%) · "
                                       f"{u.n_recomendados} com recuperação recomendada"),
                  ("Pixels de borda", f"{u.n_pixels - u.n_inteiros} de {u.n_pixels} (incluem área vizinha) · "
                                      f"{u.n_inteiros} inteiro(s) no talhão"),
                  ("Incerteza do produto", f"± {br(u.incerteza, 2)} m³/m³" if np.isfinite(u.incerteza) else "—")]
        yy = 0.224
        for a, b in linhas:
            fig.text(x2, yy, a, fontsize=6.2, color=CINZA, va="center")
            fig.text(x2 + 0.125, yy, b, fontsize=6.2, color="#222222", va="center")
            yy -= 0.0128
        obs = [o for o in N.observacoes]
        if len(obs) >= 2:
            ax4 = fig.add_axes([0.445, 0.098, 0.445, 0.052])
            xs_ = [o.data for o in obs]
            ys_ = [o.mediana if o.valida else np.nan for o in obs]
            ax4.plot(xs_, ys_, color="#08519C", lw=1.1, marker="o", ms=3.2)
            inval = [o.data for o in obs if not o.valida]
            if inval:
                ax4.scatter(inval, [np.nanmin(ys_)] * len(inval), marker="x", s=14, color="#999999",
                            label="passagem sem pixel válido")
                ax4.legend(fontsize=5.4, frameon=False, loc="upper right", handletextpad=0.2)
            ax4.set_xticks(xs_, [f"{d:%d/%m}" for d in xs_])
            ax4.set_ylabel("m³/m³", fontsize=6)
            ax4.margins(x=0.06, y=0.25)
            _eixos_limpos(ax4, 5.8)
            fig.text(x2, 0.16, f"Evolução · {len(obs)} passagens em {r['dias']} dias", fontsize=6.2, color=CINZA,
                     va="center")
    _texto(fig, 0.09, 0.083, f"NISAR {ns.NOME_CURTO} ({ns.MATURIDADE}), ASF DAAC/NASA. Umidade SUPERFICIAL: não "
                             "representa o perfil explorado pelas raízes, não foi usada para corrigir o balanço acima "
                             "e não substitui medida em campo. Cinza hachurado = pixel sem recuperação válida.",
           larg=150, fs=5.7, cor="#777777", entre=0.0098)
    _fechar(pdf, fig)


def pagina_metodos(pdf, ctx, num, diag: Diagnostico):
    fig = _pagina(pdf, ctx, num, "FONTES E MÉTODOS", "o que foi usado e os limites de cada análise")
    blocos = [
        ("Histórico de uso", [
            f"{diag.historico.colecao if diag.historico else 'MapBiomas'}: mapas anuais de cobertura e uso (30 m), "
            "1985 em diante; classes agrupadas e interpretadas em linha do tempo",
            "PRODES/INPE (TerraBrasilis): desmatamento anual por bioma, quando disponível"]),
        ("Comportamento da lavoura", [
            "Sentinel-2 L2A (Copernicus/ESA), 10–20 m; uma imagem por mês; nuvens e sombras removidas (SCL)",
            "NDVI, NDRE, NDMI e MSAVI2; vigor relativo = posição do pixel em relação à mediana do talhão",
            "Mapas dos índices: média entre anos do pico de cada ano agrícola; cores do percentil 2 ao 98 do talhão",
            "Estabilidade: média e variação do vigor relativo entre anos agrícolas (jul–jun)",
            "Anomalia recente: última imagem × mediana da mesma época em anos anteriores"]),
        ("Relevo e água", [
            "Copernicus DEM GLO-30 (reserva: SRTM); declividade em classes Embrapa; posição pelo TPI (Weiss, 2001)",
            "Escoamento D8 com preenchimento de depressões, considerando o entorno do talhão; perfis nos eixos",
            "Erosão: triagem pelo fator LS (Moore & Burch, 1986) e cobertura média observada"]),
        ("Clima e vento", [
            "Chuva diária CHIRPS 2.0 (UCSB/CHC, ~5 km) desde 1981; reserva NASA POWER",
            "Temperaturas diárias NASA POWER (~50 km); estatísticas históricas, não previsão",
            "Vento, temperatura e umidade horários NASA POWER (últimos 5 anos, hora local); janelas de pulverização "
            "com vento 3–10 km/h, < 30 °C e UR > 55%"]),
        ("Solo e temperatura de superfície", [
            "SoilGrids 2.0 (ISRIC), 250 m: apenas argila, silte e areia (0–60 cm), com intervalo de 90%",
            "Landsat 8/9 Coleção 2 (USGS/NASA): temperatura de superfície (ST_B10, 100 m) com máscara de nuvens "
            "(QA_PIXEL); média das passagens suavizada (~45 m) para o mapa"]),
    ]
    if getattr(diag, "agua", None) is not None:
        blocos.append(("Água no solo", [
            "Balanço de 90 dias: chuva CHIRPS (dias recentes: NASA POWER) e ET₀ Penman-Monteith FAO-56 (reserva: "
            "Hargreaves-Samani); dias sem dado não viram zero. Armazenamento só com cultura, semeadura, raiz e CAD "
            "informados (FAO-56, Kc único; 90 dias de aquecimento com três condições iniciais)",
            "NISAR L3 SME2 provisório (ASF DAAC/NASA), 200 m: mediana dos pixels válidos pela máscara de qualidade do "
            "produto; umidade superficial, não extrapolada para o perfil"]))
    compacto = len(blocos) > 5
    y = 0.835 if compacto else 0.83
    cor = ctx.marca["cor"]
    for tit, linhas in blocos:
        fig.add_artist(FancyBboxPatch((0.09, y - 0.024), 0.82, 0.024, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc=cor, ec="none"))
        fig.text(0.5, y - 0.012, tit, ha="center", va="center", fontsize=9.5, family=F_BOLD, weight="bold",
                 color="white")
        y -= 0.034 if compacto else 0.036
        for l in linhas:
            fig.text(0.1, y, "›", fontsize=9, color=ctx.marca["acento"], family=F_BOLD, va="top")
            y = _texto(fig, 0.12, y, l, larg=118 if compacto else 112, fs=7.0 if compacto else 7.4,
                       entre=0.0138 if compacto else 0.0155) - 0.003
        y -= 0.010 if compacto else 0.014
    if diag.avisos:
        _subtitulo(fig, ctx, 0.09, y, "Avisos desta execução")
        y -= 0.02
        for a in diag.avisos[:10]:
            y = _texto(fig, 0.1, y, "• " + a, larg=115, fs=6.8, cor=CINZA) - 0.002
    _fechar(pdf, fig)


def fundo_satelite(area, fonte="Esri World Imagery", chave_google=None):
    """Imagem de satélite da área para o mapa do resumo: (imagem, extent UTM, crédito) ou None."""
    from pyproj import Transformer
    from .. import externos
    lon0, lat0, lon1, lat1 = area.bbox_ll(max(150.0, 0.35 * np.sqrt(area.poligono.area)))
    try:
        r = externos.imagem_satelite(lon0, lat0, lon1, lat1, fonte, max_px=1400, chave_google=chave_google)
    except Exception:  # noqa: BLE001
        r = None
    if r is None:
        return None
    img, (xa, xb, yb, ya), cred = r
    t = Transformer.from_crs(3857, area.epsg, always_xy=True)
    x0, y0 = t.transform(xa, yb)
    x1, y1 = t.transform(xb, ya)
    return img, (x0, x1, y0, y1), cred


# ------------------------------------------------------------------ montagem
def montar(diag: Diagnostico, dados: DadosRelatorio, satelite=None) -> bytes:
    ctx = _ctx(dados)
    roteiro = [("Raio-X do talhão", lambda pdf, k: pagina_resumo(pdf, ctx, k, diag, satelite))]
    if diag.historico is not None:
        roteiro.append(("Histórico da área", lambda pdf, k: pagina_historico(pdf, ctx, k, diag)))
    if diag.lavoura is not None:
        roteiro.append(("Comportamento da lavoura", lambda pdf, k: pagina_lavoura_serie(pdf, ctx, k, diag)))
        if getattr(diag.lavoura, "hist", None):
            roteiro.append(("Índices de vegetação", lambda pdf, k: pagina_lavoura_indices(pdf, ctx, k, diag)))
        roteiro.append(("Estabilidade e anomalias", lambda pdf, k: pagina_lavoura_mapas(pdf, ctx, k, diag)))
        roteiro.append(("Uniformidade e pontos de visita", lambda pdf, k: pagina_lavoura_pontos(pdf, ctx, k, diag)))
    if diag.relevo is not None:
        roteiro.append(("Relevo", lambda pdf, k: pagina_relevo_1(pdf, ctx, k, diag)))
        roteiro.append(("Paisagem e água", lambda pdf, k: pagina_relevo_2(pdf, ctx, k, diag)))
    if diag.clima is not None:
        roteiro.append(("Perfil climático", lambda pdf, k: pagina_clima_1(pdf, ctx, k, diag)))
        roteiro.append(("Riscos climáticos", lambda pdf, k: pagina_clima_2(pdf, ctx, k, diag)))
    if diag.solo is not None:
        roteiro.append(("Contexto do solo", lambda pdf, k: pagina_solo(pdf, ctx, k, diag)))
    if diag.termico is not None or (diag.clima is not None and getattr(diag.clima, "vento", None) is not None):
        roteiro.append(("Temperatura e vento" if diag.termico is not None else "Vento e pulverização",
                        lambda pdf, k: pagina_temp_vento(pdf, ctx, k, diag)))
    if getattr(diag, "agua", None) is not None:
        roteiro.append(("Água no solo", lambda pdf, k: pagina_agua(pdf, ctx, k, diag)))
    roteiro.append(("Fontes e métodos", lambda pdf, k: pagina_metodos(pdf, ctx, k, diag)))
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"Diagnóstico por satélite – {dados.talhao or diag.area.nome}",
                                 "Author": ctx.marca["nome"]}) as pdf:
        pagina_capa(pdf, ctx, dados, diag)
        bk.pagina_sumario(pdf, ctx, [(t, i + 1, 0) for i, (t, _) in enumerate(roteiro)])
        for i, (_, f) in enumerate(roteiro):
            f(pdf, i + 1)
    return buf.getvalue()
