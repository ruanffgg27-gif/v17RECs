"""Montagem do book (PDF A4) com mapas de fertilidade e prescrições.

Estrutura:
  Capa (marca) → Sumário → Informações técnicas (+ clima) → Metodologias →
  Mapa de pontos → Altitude → Argila/Areia/Silte → [Capa Fertilidade] →
  Diagnóstico geral → mapas de fertilidade → Camada 20-40 → [Capa Prescrição] →
  prescrições por talhão → Volumes de produto → Qualidade dos mapas
"""
from __future__ import annotations

import functools
import io
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.patheffects  # noqa: E402,F401
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

from . import externos  # noqa: E402
from . import interpretacao as it  # noqa: E402
from .geo import Projeto, xy_de  # noqa: E402
from .krigagem import interpolar_grade  # noqa: E402
from .prescricao import MapaTalhao  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
A4 = (8.27, 11.69)

# ------------------------------------------------------------------ fontes
for _f in (ASSETS / "fontes").glob("*.ttf"):
    fm.fontManager.addfont(str(_f))
F_REG, F_MED, F_SEMI, F_BOLD, F_XB = (["Montserrat", "DejaVu Sans"], ["Montserrat Medium", "DejaVu Sans"],
                                      ["Montserrat SemiBold", "DejaVu Sans"], ["Montserrat", "DejaVu Sans"],
                                      ["Montserrat ExtraBold", "DejaVu Sans"])
plt.rcParams.update({"font.family": ["Montserrat", "DejaVu Sans"], "pdf.fonttype": 42, "mathtext.fontset": "custom",
                     "mathtext.rm": "Montserrat", "mathtext.bf": "Montserrat:bold"})

MARCAS = {
    "Protecplan": {"nome": "PROTECPLAN", "cor": "#276E2F", "acento": "#EF7622",
                   "logo": "book/logo_protecplan.png", "icone": "book/icone_protecplan.png"},
    "Atria": {"nome": "ATRIA", "cor": "#104A2A", "acento": "#E25A10",
              "logo": "logo.png", "icone": "book/icone_atria.png", "logo_capa": "book/logo_atria_vertical.png"},
    "Ativa": {"nome": "ATIVA AGROFOREST", "cor": "#12703E", "acento": "#8CC63F",
              "logo": "book/logo_ativa.png", "icone": "book/icone_ativa.png"},
}

# paletas medidas no book de referência
COR_CLASSE = {"Muito alto": "#121D5E", "Excelente": "#0A6FF2", "Muito bom": "#008056", "Bom": "#1FD12A",
              "Regular": "#F2EF05", "Baixo": "#FF8F01", "Muito baixo": "#F20000", "Crítico": "#850000"}
RAMPAS = {   # rampas com passos bem distintos (mapas em faixas sólidas, pensando na impressão)
    "argila": ["#FBE3D4", "#F4BE9E", "#EA9669", "#D8703F", "#B65222", "#8A3913", "#5A2208"],
    "areia": ["#FFF8C2", "#FFEE70", "#F9D72A", "#E0BC00", "#B99A00", "#8A7400", "#5C4E00"],
    "silte": ["#FBE2F0", "#F6B8DB", "#EE88C2", "#DD57A7", "#BD308A", "#8E196C", "#5B0B46"],
    "alt": ["#D7191C", "#F46D43", "#FDAE61", "#FEE08B", "#D9EF8B", "#A6D96A", "#66BD63", "#1A9850"],
}
COR_PRESC = ["#008000", "#00FF00", "#FFFF00", "#FF8000", "#FF0000", "#800000"]
# 7 ou 8 classes (ex.: sementes): mesma sequência, com um verde e um laranja intermediários
COR_PRESC_8 = ["#006400", "#00A000", "#7CFC00", "#FFFF00", "#FFC000", "#FF8000", "#FF0000", "#800000"]


def cores_doses(doses) -> dict:
    """Cor de cada dose PRESENTE no mapa, sempre a partir do verde escuro (menor dose): 2 doses → verde escuro e
    verde claro; 3 → + amarelo; 4 → + laranja… Com mais de 6 doses usa a rampa de 8 cores (espalhada se > 8)."""
    ds = sorted(set(float(d) for d in doses))
    n = len(ds)
    if n <= len(COR_PRESC):
        cores = COR_PRESC[:n]
    elif n <= len(COR_PRESC_8):
        cores = COR_PRESC_8[:n]
    else:
        cores = [COR_PRESC_8[int(round(i * (len(COR_PRESC_8) - 1) / (n - 1)))] for i in range(n)]
    return dict(zip(ds, cores))

TITULOS = {
    "ph": ("pH", "", "adimensional"), "mo": ("MATÉRIA ORGÂNICA", "(M.O.)", "g/dm³"),
    "ctc": ("CTC", "(capacidade de troca de cátions)", "mmolc/dm³"),
    "v": ("SATURAÇÃO POR BASES", "(V%)", "%"), "hal": ("ACIDEZ POTENCIAL", "(H+Al)", "mmolc/dm³"),
    "al": ("ACIDEZ TROCÁVEL", "(Al³⁺)", "mmolc/dm³"), "m": ("SATURAÇÃO POR ALUMÍNIO", "(m%)", "%"),
    "ca": ("CÁLCIO TROCÁVEL", "(Ca²⁺)", "mmolc/dm³"), "sat_ca": ("SATURAÇÃO POR CÁLCIO", "(%Ca)", "%"),
    "mg": ("MAGNÉSIO TROCÁVEL", "(Mg²⁺)", "mmolc/dm³"), "sat_mg": ("SATURAÇÃO POR MAGNÉSIO", "(%Mg)", "%"),
    "k": ("POTÁSSIO TROCÁVEL", "(K⁺)", "mmolc/dm³"), "sat_k": ("SATURAÇÃO POR POTÁSSIO", "(%K)", "%"),
    "p": ("FÓSFORO DISPONÍVEL", "(P resina)", "mg/dm³"), "s": ("ENXOFRE DISPONÍVEL", "(S-SO₄)", "mg/dm³"),
    "b": ("BORO", "(B)", "mg/dm³"), "zn": ("ZINCO", "(Zn²⁺)", "mg/dm³"), "mn": ("MANGANÊS", "(Mn²⁺)", "mg/dm³"),
    "cu": ("COBRE", "(Cu²⁺)", "mg/dm³"), "fe": ("FERRO", "(Fe²⁺)", "mg/dm³"),
    "argila": ("ARGILA", "", "%"), "areia": ("AREIA", "", "%"), "silte": ("SILTE", "", "%"),
    "alt": ("MAPA DE ALTITUDE", "", "metros"),
}
NOMES = {"ph": "pH", "mo": "Matéria orgânica", "ctc": "Capacidade de troca de cátions", "v": "Saturação por bases",
         "hal": "Acidez potencial", "al": "Acidez trocável", "m": "Saturação por Al³⁺", "ca": "Cálcio trocável",
         "sat_ca": "Saturação por Ca²⁺", "mg": "Magnésio trocável", "sat_mg": "Saturação por Mg²⁺",
         "k": "Potássio trocável", "sat_k": "Saturação por K⁺", "p": "Fósforo disponível",
         "s": "Enxofre disponível", "b": "Boro disponível", "zn": "Zinco disponível", "mn": "Manganês disponível",
         "cu": "Cobre disponível", "fe": "Ferro disponível", "argila": "Argila", "areia": "Areia", "silte": "Silte",
         "alt": "Mapa de altitude"}
CURTOS = {"ph": "pH", "mo": "M.O.", "ctc": "CTC", "v": "V%", "hal": "H+Al", "al": "Al³⁺", "m": "m%", "ca": "Ca²⁺",
          "sat_ca": "Ca/CTC", "mg": "Mg²⁺", "sat_mg": "Mg/CTC", "k": "K⁺", "sat_k": "K/CTC", "p": "P",
          "s": "S-SO₄", "b": "B", "zn": "Zn", "mn": "Mn", "cu": "Cu", "fe": "Fe", "argila": "Argila %"}
ORDEM_FERT = ["ph", "mo", "ctc", "v", "hal", "al", "m", "ca", "sat_ca", "mg", "sat_mg", "k", "sat_k",
              "p", "s", "b", "zn", "mn", "cu", "fe"]
SUBSCR = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def curto(nome: str) -> str:
    """'Talhão 1' → 'TH 01'."""
    import re
    m = re.fullmatch(r"Talh[aã]o\s*0*(\d+)", str(nome).strip())
    return f"TH {int(m.group(1)):02d}" if m else str(nome)


def br(v, casas=1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# granulometria: guardada em g/kg (é o que entra nas contas), exibida em % (g/kg ÷ 10)
FATOR_EXIBICAO = {"argila": 0.1, "areia": 0.1, "silte": 0.1}


def exibir(chave: str, valores):
    """Valores na unidade de exibição do book (granulometria em %)."""
    return valores * FATOR_EXIBICAO.get(chave, 1.0)


def casas_de(chave: str, vmax: float) -> int:
    if chave in ("b", "k", "al", "cu", "zn") or vmax < 5:
        return 2
    return 1 if vmax < 200 else 0


# --------------------------------------------------------------- dados
@dataclass
class DadosBook:
    marca: str = "Protecplan"
    produtor: str = ""
    propriedade: str = ""
    municipio: str = ""
    data_coleta: str = ""
    ano_safra: str = ""
    mes_ano: str = ""
    subamostras: int = 10
    produto_calcario: str = ""
    produto_gesso: str = "Gesso agrícola · 19% Ca²⁺; 15% S"
    produto_p: str = "P₂O₅"
    produto_k: str = "KCl (60% K₂O)"
    fonte_satelite: str | None = "Esri World Imagery"
    usar_internet: bool = True
    laudo_numero: str = ""
    chave_google: str | None = None
    cores_discretas: bool = True     # mapas de fertilidade em faixas sólidas (1 cor por classe)
    incluir_ndvi: bool = True        # página de NDVI (Sentinel-2) — precisa de internet
    titulo_capa: tuple[str, str] | None = None     # duas linhas do título da capa (padrão: avaliação + prescrições)
    metodos: dict = field(default_factory=dict)    # métodos reconhecidos no laudo: {"p": …, "ph": …}
    interpolacao: dict = field(default_factory=dict)  # {"limitar_extremos": bool, "sem_estrutura": str}

    def linhas_titulo(self) -> tuple[str, str]:
        return self.titulo_capa or ("AVALIAÇÃO DA FERTILIDADE DO SOLO", f"E PRESCRIÇÕES {self.ano_safra}".strip())


@dataclass
class Contexto:
    projeto: Projeto
    mapas: list[MapaTalhao]
    dados: DadosBook
    marca: dict
    avisos: list[str] = field(default_factory=list)
    clima: dict | None = None
    satelite: tuple | None = None
    tem_alt: bool = False
    subs: pd.DataFrame | None = None
    fonte_altitude: str = ""
    ndvi: dict | None = None
    argila_est: float = 0.0          # fração das amostras 0-20 com argila estimada pela CTC
    tem_areia_silte: bool = True


# ------------------------------------------------------------- elementos
def cor_texto(fundo: str) -> str:
    r, g, b = to_rgb(fundo)
    return "#1A1A1A" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "white"


@functools.lru_cache(maxsize=32)
def _img(nome):
    return np.asarray(Image.open(ASSETS / nome).convert("RGBA"))


def _imagem(fig, img, rect):
    """Desenha uma imagem na posição `rect` (fração da página) numa camada única por página.

    Criar um Axes por ícone custava ~12 ms cada (centenas por book); uma camada só é bem mais rápido."""
    ax = getattr(fig, "_camada_img", None)
    if ax is None:
        ax = fig.add_axes([0, 0, 1, 1], zorder=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_autoscale_on(False)
        ax.axis("off")
        fig._camada_img = ax
    L, B, W, H = rect
    ax.imshow(img, extent=[L, L + W, B, B + H], aspect="auto", zorder=1)


MOLD_Y0, MOLD_Y1 = 0.062, 0.968      # moldura (fração da página)


def base(fig, ctx: Contexto, num: int | None, moldura=True):
    cor = ctx.marca["cor"]
    fig.patch.set_facecolor("white")
    fig._cor_marca = cor                      # usada pela barra de escala (mesma cor da moldura)
    if not moldura:
        return
    fig.add_artist(FancyBboxPatch((0.07, MOLD_Y0), 0.86, MOLD_Y1 - MOLD_Y0,
                                  boxstyle="round,pad=0,rounding_size=0.045", transform=fig.transFigure,
                                  fc="none", ec=cor, lw=2.6, mutation_aspect=A4[0] / A4[1]))
    fig.text(0.5, MOLD_Y1, ctx.marca["nome"], ha="center", va="center", fontsize=12, family=F_XB,
             color="#111111", bbox=dict(fc="white", ec="none", pad=6))
    ic = _img(ctx.marca["icone"])
    h = 0.032
    w = h * ic.shape[1] / ic.shape[0] * A4[1] / A4[0]
    _imagem(fig, ic, [0.5 - w / 2, (MOLD_Y0 - h) / 2 - 0.002, w, h])
    if num is not None:
        fig.text(0.93, MOLD_Y0 / 2, str(num), ha="right", va="center", color=cor, fontsize=10, family=F_SEMI)


LARG_TITULO, LARG_SUB = 0.66, 0.78       # largura máxima (fração da página) do título e do subtítulo


def _largura(fig, t) -> float:
    r = fig.canvas.get_renderer()
    return t.get_window_extent(r).width / (fig.get_figwidth() * fig.dpi)


def caber(fig, t, larg_max: float, fs_min: float) -> None:
    """Faz um texto caber na largura dada: reduz a fonte até `fs_min` e, se ainda não couber, encurta pelo meio
    (mantém o começo e o fim — numa lista de talhões, o primeiro e o último continuam visíveis)."""
    while _largura(fig, t) > larg_max and t.get_fontsize() > fs_min:
        t.set_fontsize(max(fs_min, t.get_fontsize() - 0.5))
    txt = t.get_text()
    corte = len(txt)
    while _largura(fig, t) > larg_max and corte > 12:
        corte -= 2
        a = corte * 3 // 5
        t.set_text(txt[:a].rstrip(" ,·-") + " … " + txt[len(txt) - (corte - a):].lstrip(" ,·-"))


def titulo(fig, ctx: Contexto, texto: str, sub: str = "", y=0.905, tam=24):
    """Título centralizado com os ícones da marca dos lados e subtítulo. Títulos e subtítulos longos (vários
    talhões prescritos juntos, por exemplo) são ajustados para nunca passar da moldura."""
    cor = ctx.marca["cor"]
    linhas = texto.split("\n")
    objs = [fig.text(0.5, y - i * 0.036, l, ha="center", va="center", fontsize=tam, family=F_XB, color=cor)
            for i, l in enumerate(linhas)]
    for t in objs:
        caber(fig, t, LARG_TITULO, 12)
    fs = min(t.get_fontsize() for t in objs)
    for t in objs:
        t.set_fontsize(fs)
    larg = max(_largura(fig, t) for t in objs) / 2 + 0.035
    yb = y - (len(linhas) - 1) * 0.018
    ic = _img(ctx.marca["icone"])
    h = 0.026
    w = h * ic.shape[1] / ic.shape[0] * A4[1] / A4[0]
    for xc in (0.5 - larg, 0.5 + larg):
        _imagem(fig, ic, [xc - w / 2, yb - h / 2, w, h])
    if sub:
        ts = fig.text(0.5, y - len(linhas) * 0.036 + 0.006, sub, ha="center", va="center", fontsize=12.5,
                      family=F_BOLD, weight="bold", color=cor)
        caber(fig, ts, LARG_SUB, 8)


def texto_corrido(fig, x, y, txt: str, larg: float, y_min: float, fs: float = 8.3, fs_min: float = 6.0,
                  cor="#333333", entre: float = 1.38) -> None:
    """Parágrafo quebrado em linhas dentro de `larg` (fração da página), sem descer abaixo de `y_min`: se não
    couber, a fonte diminui até `fs_min`; no limite, o texto termina com reticências."""
    import textwrap
    alt_pag = fig.get_figheight() * 72                       # pontos
    larg_pt = larg * fig.get_figwidth() * 72
    while True:
        por_linha = max(20, int(larg_pt / (fs * 0.52)))      # ~0,52 em por caractere (Montserrat)
        passo = fs * entre / alt_pag
        linhas = textwrap.wrap(txt, por_linha)
        cabem = max(1, int((y - y_min) / passo) + 1)
        if len(linhas) <= cabem or fs <= fs_min:
            break
        fs -= 0.3
    if len(linhas) > cabem:
        linhas = linhas[:cabem]
        linhas[-1] = linhas[-1][:max(0, por_linha - 2)].rstrip(" ;,") + " …"
    for i, l in enumerate(linhas):
        t = fig.text(x, y - i * passo, l, fontsize=fs, color=cor, va="center")
        caber(fig, t, larg, fs_min - 1)


def norte(fig, x=0.845, y=0.085, h=0.042):
    n = _img("book/norte.png")
    w = h * n.shape[1] / n.shape[0] * A4[1] / A4[0]
    _imagem(fig, n, [x, y, w, h])


def escala(fig, ax, x, y, cor=None, alinhar="esq", frac=0.14, fs=7.5):
    """Barra de escala desenhada na página (fora do mapa), na mesma escala do mapa, na cor da moldura da página.

    `frac`: comprimento aproximado da barra em fração da largura da página."""
    cor = cor or getattr(fig, "_cor_marca", "#222222")
    x0, x1 = ax.get_xlim()
    bb = ax.get_position()
    m_por_fig = (x1 - x0) / bb.width
    L = _redondo(m_por_fig * frac)
    w = L / m_por_fig
    xa = x if alinhar == "esq" else x - w
    kw = dict(transform=fig.transFigure, color=cor, lw=1.4, solid_capstyle="butt")
    fig.add_artist(plt.Line2D([xa, xa + w], [y, y], **kw))
    for xx in (xa, xa + w / 2, xa + w):
        fig.add_artist(plt.Line2D([xx, xx], [y - 0.004, y + 0.004], **{**kw, "lw": 1.0}))
    fig.text(xa + w / 2, y + 0.008, f"{L:,.0f} m".replace(",", "."), ha="center", va="bottom", fontsize=fs,
             family=F_SEMI, color=cor)
    return w


def _redondo(v):
    e = 10 ** math.floor(math.log10(v))
    return min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * e - v)) * e


def _caixa_mapa(fig, ctx, rect_livre, xs=None):
    """Axes do mapa ajustado à caixa (fig coords) mantendo a proporção da fazenda."""
    x0, y0, x1, y1 = xs if xs is not None else _limites(ctx)
    dx, dy = x1 - x0, y1 - y0
    L, B, W, H = rect_livre
    asp_caixa = (H * A4[1]) / (W * A4[0])
    if dy / dx > asp_caixa:
        w = H * A4[1] / A4[0] * dx / dy
        rect = [L + (W - w) / 2, B, w, H]
    else:
        h = W * A4[0] / A4[1] * dy / dx
        rect = [L, B + (H - h) / 2, W, h]
    ax = fig.add_axes(rect)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _limites(ctx_ou_mapas, margem=0.04):
    mapas = ctx_ou_mapas.mapas if hasattr(ctx_ou_mapas, "mapas") else ctx_ou_mapas
    b = np.array([m.talhao.perimetro.bounds for m in mapas])
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    mx, my = (x1 - x0) * margem, (y1 - y0) * margem
    return x0 - mx, y0 - my, x1 + mx, y1 + my


def agrupar_por_proximidade(mapas, distancia=2500.0):
    """Talhões a menos de `distancia` m uns dos outros ficam no mesmo grupo (mesma página)."""
    n = len(mapas)
    pai = list(range(n))

    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if mapas[i].talhao.perimetro.distance(mapas[j].talhao.perimetro) <= distancia:
                pai[raiz(i)] = raiz(j)
    grupos = {}
    for i in range(n):
        grupos.setdefault(raiz(i), []).append(mapas[i])
    return list(grupos.values())


def nome_curto(nome: str, maximo: int = 20) -> str:
    """Nome do talhão para tabelas e legendas estreitas: abreviado pelo meio, mantendo o começo e o número final."""
    n = curto(nome)
    if len(n) <= maximo:
        return n
    fim = max(3, len(n) - len(n.rstrip("0123456789")) + 1)
    return n[:maximo - 1 - fim].rstrip() + "…" + n[-fim:].lstrip()


def lista_talhoes(mapas, maximo=6, max_car=58) -> str:
    """Nomes dos talhões; com muitos talhões (ou nomes longos) vira 'primeiro a último (n talhões)' para caber."""
    nomes = [curto(m.talhao.nome) for m in mapas]
    txt = ", ".join(nomes)
    if len(nomes) <= maximo and len(txt) <= max_car:
        return txt
    return f"{nomes[0]} a {nomes[-1]} ({len(nomes)} talhões)"


def _patch(geom, **kw):
    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    verts, codes = [], []
    for p in polys:
        for anel in [p.exterior, *p.interiors]:
            c = np.asarray(anel.coords)
            verts += c.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw)


def _raster_suave(m: MapaTalhao, Z, fator=3):
    """Aumenta a resolução (bilinear) para bordas suaves; valores fora do talhão estendidos."""
    mask = m.grade.mascara & np.isfinite(Z)
    if not mask.any():
        return Z, None
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    Zf = Z[tuple(idx)]
    Zs = ndimage.zoom(Zf, fator, order=1)
    r = m.grade.res
    ext = (m.grade.x[0] - r / 2, m.grade.x[-1] + r / 2, m.grade.y[-1] - r / 2, m.grade.y[0] + r / 2)
    return Zs, ext


def faixas_solidas(ax, m: MapaTalhao, Z, limites, cores, recorte=None, zorder=2):
    """Desenha as classes como CONTORNOS VETORIAIS preenchidos (não como pixels): as divisas entre cores ficam
    lisas em qualquer ampliação, sem "escadinha" nem pontos soltos de pixel. Devolve False se não há dado."""
    mask = m.grade.mascara & np.isfinite(Z)
    if not mask.any():
        return False
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    Zf = Z[tuple(idx)]                                  # estende os valores para fora do talhão (recorte depois)
    X, Y = m.grade.x, m.grade.y
    fator = int(np.clip(round(600 / max(Zf.shape)), 1, 4))     # talhões pequenos ganham curvas mais finas
    if fator > 1 and min(Zf.shape) >= 4:
        lo, hi = float(Zf.min()), float(Zf.max())
        Zf = np.clip(ndimage.zoom(Zf, fator, order=3, mode="nearest"), lo, hi)
        X = np.linspace(X[0], X[-1], Zf.shape[1])
        Y = np.linspace(Y[0], Y[-1], Zf.shape[0])
    niveis = np.array(limites, float).copy()
    niveis[0], niveis[-1] = -1e12, 1e12                 # abaixo da 1ª e acima da última faixa = classes extremas
    cs = ax.contourf(X, Y, Zf, levels=niveis, colors=list(cores), antialiased=True, zorder=zorder)
    cs.set_edgecolor("face")                            # fecha a fresta clara entre faixas vizinhas
    cs.set_linewidth(0.25)
    if recorte is not None:
        cs.set_clip_path(recorte)
    return True


# ----------------------------------------------------------- escalas de cor
def escala_classes(chave):
    """Limites (crescentes) e cores de cada classe para atributos com tabela de legendas."""
    faixas = sorted(it.ATRIBUTOS[chave][2], key=lambda t: t[1])
    limites = [faixas[0][1]] + [f[2] for f in faixas]
    cores = [COR_CLASSE[f[0]] for f in faixas]
    # H+Al, Al e m%: os nomes das classes (pensados para "maior é melhor") não se aplicam — rotula-se pela faixa
    nomes = None if chave in it.INVERTIDOS else [f[0] for f in faixas]
    return np.array(limites, float), cores, nomes


def nomes_condicao(chave) -> list[str]:
    """Condição de cada classe (na ordem dos limites). Em H+Al, Al e m% a faixa mais baixa é "Excelente"."""
    nomes = [f[0] for f in sorted(it.ATRIBUTOS[chave][2], key=lambda t: t[1])]
    return ["Excelente" if chave in it.INVERTIDOS and n == "Muito alto" else n for n in nomes]


def somar_por_condicao(chave, quantidades) -> dict:
    cont: dict = {}
    for nome, q in zip(nomes_condicao(chave), quantidades):
        cont[nome] = cont.get(nome, 0) + q
    return cont


def escala_intervalos(valores, chave, n=None):
    v = valores[np.isfinite(valores)]
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-9:
        hi = lo + 1
    cores = RAMPAS.get(chave, RAMPAS["alt"])
    n = n or len(cores)
    return np.linspace(lo, hi, n + 1), cores, None


def _cmap_posicional(cores):
    n = len(cores)
    nos = [(i + 0.5) / n for i in range(n)]
    pontos = [(0.0, to_rgb(cores[0]))] + [(p, to_rgb(c)) for p, c in zip(nos, cores)] + [(1.0, to_rgb(cores[-1]))]
    return LinearSegmentedColormap.from_list("pos", pontos)


def _posicao(Z, limites):
    return np.interp(Z, limites, np.linspace(0, 1, len(limites)))


# ------------------------------------------------------------- página de mapa
def _bloco_medias(fig, ctx, mapas, chave, casas, x, y, colunas=3, larg=0.13):
    """Médias por talhão em colunas (para muitos talhões)."""
    for i, m in enumerate(mapas):
        cx = x + (i % colunas) * larg
        cy = y - (i // colunas) * 0.016
        fig.text(cx, cy, f"{nome_curto(m.talhao.nome, 16)}: {br(np.nanmean(exibir(chave, m.atributos[chave])), casas)}",
                 fontsize=7.4,
                 color="#444444")


def pagina_atributo(pdf, ctx: Contexto, chave: str, num: int, titulo_txt=None, sub=None, mapas=None,
                    rotulo_grupo: str = ""):
    mapas = [m for m in (mapas or ctx.mapas) if chave in m.atributos]
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    t, s_, unid = TITULOS.get(chave, (chave.upper(), "", ""))
    sub_txt = sub if sub is not None else s_
    if rotulo_grupo:
        sub_txt = f"{sub_txt}  ·  {rotulo_grupo}".strip(" ·")
    titulo(fig, ctx, titulo_txt or t, sub_txt)
    todos = np.concatenate([exibir(chave, m.atributos[chave][m.grade.mascara]) for m in mapas])
    if chave in it.ATRIBUTOS:
        limites, cores, nomes = escala_classes(chave)
    else:
        limites, cores, nomes = escala_intervalos(todos, chave)
    discreto = ctx.dados.cores_discretas
    cmap = ListedColormap(cores) if discreto else _cmap_posicional(cores)
    lim = _limites(mapas)
    largo = (lim[3] - lim[1]) / (lim[2] - lim[0]) < 0.95          # fazenda "deitada"
    caixa = [0.09, 0.485, 0.82, 0.335] if largo else [0.30, 0.13, 0.61, 0.68]
    ax = _caixa_mapa(fig, ctx, caixa, lim)
    larg_fig = ax.get_position().width * A4[0]                     # polegadas
    for m in mapas:
        px_talhao = larg_fig * 200 * (m.grade.x[-1] - m.grade.x[0]) / (lim[2] - lim[0])   # ~200 dpi
        fator = int(np.clip(round(px_talhao / max(m.grade.mascara.shape[1], 1)), 1, 5 if discreto else 3))
        pp = _patch(m.talhao.perimetro, fc="none", ec="none")
        ax.add_artist(pp)
        if discreto:
            if not faixas_solidas(ax, m, exibir(chave, m.atributos[chave]), limites, cores, pp):
                continue
        else:
            Zs, ext = _raster_suave(m, exibir(chave, m.atributos[chave]), fator=fator)
            if ext is None:
                continue
            im = ax.imshow(_posicao(Zs, limites), extent=ext, cmap=cmap, vmin=0, vmax=1,
                           interpolation="bilinear", zorder=2)
            im.set_clip_path(pp)
        ax.add_artist(_patch(m.talhao.perimetro, fc="none", ec="#111111", lw=0.8, zorder=5))
        if len(mapas) > 1:
            c = m.talhao.perimetro.representative_point()
            ax.text(c.x, c.y, curto(m.talhao.nome), ha="center", va="center", fontsize=6.5 if len(mapas) > 6 else 8,
                    family=F_BOLD, weight="bold", color="#111111", zorder=6,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="white")])
    escala(fig, ax, 0.115, 0.088)
    pix = np.concatenate([np.full(m.grade.mascara.sum(), m.grade.area_pixel_ha) for m in mapas])
    cls = np.clip(np.searchsorted(limites, todos, side="right") - 1, 0, len(cores) - 1)
    areas = [pix[cls == i].sum() for i in range(len(cores))]
    _legenda_gradiente(fig, ctx, limites, cores, areas, unid, casas_de(chave, limites[-1]), cmap, discreto,
                       altura=0.29 if largo else 0.33)
    media = float(np.nanmean(todos))
    cs = casas_de(chave, media)
    u = "" if unid == "adimensional" else f" {unid}"
    x_media = 0.66 if largo else 0.5
    fig.text(x_media, 0.40 if largo else 0.098, f"Média: {br(media, cs)}{u}", ha="center", fontsize=10.5,
             family=F_BOLD, weight="bold", color=ctx.marca["cor"])
    cab_medias = "Média por talhão" + (f" ({unid})" if u else "")
    if len(mapas) > 1:
        if largo:
            fig.text(0.47, 0.381, cab_medias, fontsize=7.6, family=F_SEMI, color=ctx.marca["cor"])
            _bloco_medias(fig, ctx, mapas, chave, cs, 0.47, 0.364, colunas=3, larg=0.14)
        elif len(mapas) <= 3:
            fig.text(0.5, 0.080, "  ·  ".join(
                f"{curto(m.talhao.nome)}: {br(np.nanmean(exibir(chave, m.atributos[chave])), cs)}{u}" for m in mapas), ha="center", fontsize=8, color="#444444")
        else:
            fig.text(0.115, 0.79, cab_medias, fontsize=8, family=F_SEMI, color=ctx.marca["cor"])
            _bloco_medias(fig, ctx, mapas, chave, cs, 0.115, 0.77, colunas=1, larg=0)
    if chave in it.ATRIBUTOS:
        cl = it.classificar(chave, media)
        fig.text(0.5, 0.838 if not rotulo_grupo else 0.834, f"Condição média: {cl}", ha="center", fontsize=9,
                 family=F_SEMI, color=cor_texto(COR_CLASSE[cl]),
                 bbox=dict(boxstyle="round,pad=0.35", fc=COR_CLASSE[cl], ec="none"))
    norte(fig)
    pdf.savefig(fig)
    plt.close(fig)


def _legenda_gradiente(fig, ctx, limites, cores, areas, unid, casas, cmap, discreto=False, altura=0.33):
    n = len(cores)
    L, B, W, H = 0.115, 0.135, 0.028, altura
    ax = fig.add_axes([L, B, W, H])
    if discreto:                       # blocos sólidos separados por um filete branco
        for i, c in enumerate(cores):
            ax.add_artist(Rectangle((0, i / n), 1, 1 / n, color=c, lw=0))
            ax.axhline(i / n, color="white", lw=1.2)
    else:
        grad = np.linspace(0, 1, 256)[:, None]
        ax.imshow(grad, aspect="auto", cmap=cmap, origin="lower", extent=(0, 1, 0, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333333")
        s.set_linewidth(0.6)
    fig.text(L, B + H + 0.012, unid, fontsize=10, family=F_BOLD, weight="bold", color=ctx.marca["cor"])
    for i in range(n):
        y = B + H * (i + 0.5) / n
        lo, hi = limites[i], limites[i + 1]
        if i == n - 1:
            txt = f"> {br(lo, casas)}"
        elif i == 0:
            txt = f"< {br(hi, casas)}"
        else:
            txt = f"{br(lo, casas)} – {br(hi, casas)}"
        fig.text(L + W + 0.012, y, f"{txt}  ({br(areas[i], 2)} ha)", va="center", fontsize=8.2,
                 family=F_SEMI, color=ctx.marca["cor"])


# ------------------------------------------------------------- prescrição
def pagina_prescricao(pdf, ctx: Contexto, b, produto: str, num: int, sub: str):
    """Prescrição de um bloco de aplicação (um talhão ou vários talhões prescritos juntos)."""
    if isinstance(b, MapaTalhao):
        from .prescricao import Bloco
        b = Bloco(b.talhao.nome, [b], b.zonas)
    zon = b.zonas[produto]
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    nome_tal = curto(b.nome)
    rot = {"Calcário": "CALAGEM", "Gesso": "GESSAGEM", "S elementar": "ENXOFRE ELEMENTAR", "P2O5": "FÓSFORO",
           "KCl": "POTÁSSIO", "KCl 1ª aplicação": "POTÁSSIO · 1ª APLICAÇÃO",
           "KCl 2ª aplicação": "POTÁSSIO · 2ª APLICAÇÃO", "P2O5 correção": "FÓSFORO · CORREÇÃO"}.get(
        produto, "FÓSFORO · CORREÇÃO" if produto.startswith("Correção") else produto.upper())
    tam = 24 if len(rot) + len(nome_tal) < 22 else 18
    if not b.unico:
        sub = "  ·  ".join(x for x in (sub, lista_talhoes(b.mapas)) if x)
    titulo(fig, ctx, f"{rot} - {nome_tal.upper() if not b.unico else nome_tal}", sub, tam=tam)
    per = b.perimetro
    xs = per.bounds
    mx, my = (xs[2] - xs[0]) * 0.04, (xs[3] - xs[1]) * 0.04
    ax = _caixa_mapa(fig, ctx, [0.12, 0.30, 0.76, 0.52], (xs[0] - mx, xs[1] - my, xs[2] + mx, xs[3] + my))
    cor_de = cores_doses(d for d, _ in zon.poligonos)      # só as doses que aparecem no mapa
    for d, g in zon.poligonos:
        ax.add_artist(_patch(g, fc=cor_de[float(d)], ec="none", zorder=2))
    for m in b.mapas:
        ax.add_artist(_patch(m.talhao.perimetro, fc="none", ec="#111111", lw=0.9, zorder=5))
        if not b.unico:
            c = m.talhao.perimetro.representative_point()
            ax.text(c.x, c.y, curto(m.talhao.nome), ha="center", va="center", fontsize=7.5 if len(b.mapas) > 5 else 9,
                    family=F_BOLD, weight="bold", color="#111111", zorder=6,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.4, foreground="white")])
    escala(fig, ax, 0.83, 0.095, alinhar="dir")
    tab = zon.tabela().sort_values("Dose (kg/ha)", ascending=False)
    L, y = 0.13, 0.265
    fig.text(L, y, "kg/ha", fontsize=10, family=F_BOLD, weight="bold", color=ctx.marca["cor"])
    for _, r in tab.iterrows():
        y -= 0.022
        fig.add_artist(Rectangle((L, y - 0.007), 0.022, 0.015, transform=fig.transFigure,
                                 fc=cor_de[float(r["Dose (kg/ha)"])], ec="#333333", lw=0.4))
        fig.text(L + 0.03, y, f"{br(r['Dose (kg/ha)'], 1)}  ({br(r['Área (ha)'], 2)} ha)", va="center",
                 fontsize=9, family=F_SEMI, color=ctx.marca["cor"])
    y -= 0.034
    fig.text(L, y, f"Total: {br(zon.total_kg / 1000, 2)} t", fontsize=11, family=F_BOLD, weight="bold",
             color=ctx.marca["cor"])
    fig.text(L, y - 0.02, f"Média: {br(zon.dose_media, 1)} kg/ha  ·  Área: {br(zon.area_ha, 2)} ha",
             fontsize=9.5, family=F_SEMI, color=ctx.marca["cor"])
    norte(fig)
    pdf.savefig(fig)
    plt.close(fig)


# ------------------------------------------------------------ outras páginas
def _mistura(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * t)


def pagina_capa(pdf, ctx: Contexto):
    """Capa geral: arte de fundo; logo e título no céu; produtor e propriedade num cartão."""
    d = ctx.dados
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    fig = plt.figure(figsize=A4)
    ax = _fundo_capa(fig, "geral", ctx)
    ceu = _ceu("geral")
    _veu(ax, ceu - 0.08, ceu + 0.005, alfa=0.85)
    lg = _img(ctx.marca.get("logo_capa", ctx.marca["logo"]))
    h = 0.12
    w = h * lg.shape[1] / lg.shape[0] * A4[1] / A4[0]
    if w > 0.42:
        w = 0.42
        h = w * lg.shape[0] / lg.shape[1] * A4[0] / A4[1]
    yc = 0.895
    ax.imshow(lg, extent=(0.5 - w / 2, 0.5 + w / 2, yc - h / 2, yc + h / 2), aspect="auto", zorder=3)
    ax.plot([0.36, 0.64], [0.815, 0.815], color=ac, lw=2.4, solid_capstyle="round", zorder=3)
    l1, l2 = d.linhas_titulo()
    ax.text(0.5, 0.79, l1, ha="center", va="center", fontsize=17, family=F_XB, color=cor, zorder=3)
    ax.text(0.5, 0.764, l2, ha="center", va="center", fontsize=17, family=F_XB, color=cor, zorder=3)
    # cartão com produtor, propriedade e município (canto inferior esquerdo, sobre o talhão)
    itens = [(r, v) for r, v in (("Produtor", d.produtor), ("Propriedade", d.propriedade)) if v]
    if itens or d.municipio:
        alt = 0.028 + 0.05 * len(itens) + (0.024 if d.municipio else 0)
        x0, y0, larg = 0.055, 0.045, 0.43
        ax.add_artist(FancyBboxPatch((x0, y0), larg, alt, boxstyle="round,pad=0,rounding_size=0.018",
                                    fc=CREME, alpha=0.94, ec="none", mutation_aspect=A4[0] / A4[1], zorder=4))
        ax.add_artist(Rectangle((x0, y0 + 0.012), 0.008, alt - 0.024, fc=ac, ec="none", zorder=5))
        y = y0 + alt - 0.026
        for rot, val in itens:
            ax.text(x0 + 0.03, y, rot.upper(), fontsize=7.5, family=F_SEMI, color=ac, va="center", zorder=5)
            ax.text(x0 + 0.03, y - 0.021, val, fontsize=12.5, family=F_BOLD, weight="bold", color="#222222",
                    va="center", zorder=5)
            y -= 0.05
        if d.municipio:
            ax.text(x0 + 0.03, y + 0.008, d.municipio, fontsize=9.5, color="#555555", va="center", zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    pdf.savefig(fig)
    plt.close(fig)


# imagens de fundo das capas (arte da equipe): arquivo e onde termina o céu (fração da página, de baixo p/ cima)
# terceiro valor: quanto a arte desce na página (corta um pouco do chão e amplia o céu para o título)
CAPAS = {"geral": ("book/capa_geral.jpg", 0.792, 0.04), "fertilidade": ("book/capa_fertilidade.jpg", 0.685, 0.0),
         "prescricao": ("book/capa_prescricao.jpg", 0.825, 0.11)}


def _ceu(tipo: str) -> float:
    """Altura (fração da página) onde termina o céu livre da arte, já com o deslocamento."""
    return CAPAS[tipo][1] - CAPAS[tipo][2]
CREME = "#FAF6E6"


def imagem_capa(ax, img, desce: float):
    """Arte na largura da página, com o topo `desce` abaixo da borda superior.

    A faixa acima da arte é preenchida repetindo a primeira linha da própria imagem: assim a imagem começa na
    borda da página e não aparece a "costura" (linha fina) que a reamostragem deixava na borda superior."""
    h_pag = img.shape[0] / img.shape[1] * A4[0] / A4[1]
    topo = 1 - desce
    if desce > 0:
        n = int(np.ceil(desce / h_pag * img.shape[0])) + 2
        img = np.concatenate([np.repeat(img[:1], n, axis=0), img], axis=0)
        topo += n / (img.shape[0] - n) * h_pag
        h_pag = img.shape[0] / img.shape[1] * A4[0] / A4[1]
    ax.imshow(img, extent=(0, 1, topo - h_pag, topo), aspect="auto", interpolation="antialiased", zorder=0)


def _fundo_capa(fig, tipo: str, ctx):
    """Imagem de fundo em página inteira (largura total, cortando o excesso do céu no topo).

    Se o arquivo não existir, usa a ilustração desenhada em código."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    arq = ASSETS / CAPAS[tipo][0]
    if arq.exists():
        img = _img(CAPAS[tipo][0])
        h_pag = img.shape[0] / img.shape[1] * A4[0] / A4[1]       # altura da imagem na largura da página
        fig.patch.set_facecolor(to_rgb(tuple(img[:30].reshape(-1, 4)[:, :3].mean(0) / 255)))
        imagem_capa(ax, img, CAPAS[tipo][2])
    else:
        from .ilustracoes import ilustracao
        if tipo == "geral":
            ax.imshow(ilustracao("geral", ctx.marca["cor"], ctx.marca["acento"]), extent=(0, 1, 0, 5.0 / A4[1]),
                      aspect="auto", zorder=0)
        else:
            ax.imshow(ilustracao(tipo, ctx.marca["cor"], ctx.marca["acento"]), extent=(0, 1, 0, 1), aspect="auto",
                      zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def _veu(ax, y0, y1, cor=CREME, alfa=0.9, z=1):
    """Degradê da cor do céu (transparente em y0 → opaco em y1) para destacar o título sobre a arte."""
    t = np.linspace(0, 1, 128)[:, None]
    rgba = np.zeros((128, 2, 4))
    rgba[..., :3] = to_rgb(cor)
    rgba[..., 3] = (t * alfa) * np.ones((1, 2))
    ax.imshow(rgba, extent=(0, 1, y0, y1), origin="lower", aspect="auto", zorder=z, interpolation="bilinear")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _logo_caixa(ax, ctx, y=0.035, larg=0.22, alt_max=0.11):
    lg = _img(ctx.marca["logo"])
    w = larg
    h = w * lg.shape[0] / lg.shape[1] * A4[0] / A4[1]
    if h > alt_max:
        h = alt_max
        w = h * lg.shape[1] / lg.shape[0] * A4[1] / A4[0]
    ax.add_artist(FancyBboxPatch((0.5 - w / 2 - 0.025, y), w + 0.05, h + 0.03,
                                boxstyle="round,pad=0,rounding_size=0.015", fc="white", alpha=0.92, ec="none",
                                mutation_aspect=A4[0] / A4[1], zorder=3))
    ax.imshow(lg, extent=(0.5 - w / 2, 0.5 + w / 2, y + 0.015, y + 0.015 + h), aspect="auto", zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def pagina_secao(pdf, ctx: Contexto, texto: str, tipo: str):
    """Capa de seção: arte de fundo em página inteira e título na área do céu."""
    fig = plt.figure(figsize=A4)
    ax = _fundo_capa(fig, tipo, ctx)
    ceu = _ceu(tipo)
    linhas = texto.split("\n")
    y_tit, passo, tam, y_lin = 0.875, 0.068, 40, 0.755
    y_base = y_lin - 0.03
    if ceu > y_base:                                  # a arte sobe além do título: véu suave
        _veu(ax, y_base - 0.06, ceu + 0.01)
    for i, l in enumerate(linhas):
        ax.text(0.5, y_tit - i * passo, l, ha="center", va="center", fontsize=tam, family=F_XB,
                color=ctx.marca["cor"], zorder=4)
    ax.plot([0.38, 0.62], [y_lin, y_lin], color=ctx.marca["acento"], lw=3.2, zorder=4, solid_capstyle="round")
    _logo_caixa(ax, ctx)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_sumario(pdf, ctx: Contexto, itens: list[tuple[str, int, int]]):
    fig = plt.figure(figsize=A4)
    base(fig, ctx, None)
    titulo(fig, ctx, "SUMÁRIO")
    n = len(itens)
    ncol = 1 if n <= 34 else 2
    por_col = int(np.ceil(n / ncol))
    passo = min(0.022, 0.72 / max(por_col, 1))
    fs = max(6.5, min(10, passo * A4[1] * 72 * 0.62))
    larg = 0.74 / ncol
    for idx, (texto, pag, nivel) in enumerate(itens):
        c, r = idx // por_col, idx % por_col
        x0 = 0.13 + c * (larg + 0.02)
        y = 0.83 - r * passo
        x = x0 + 0.025 * nivel
        cor = ctx.marca["cor"] if nivel == 0 else "#333333"
        fam = F_BOLD if nivel == 0 else F_REG
        t = fig.text(x, y, texto, fontsize=fs if nivel else fs + 0.5, family=fam,
                     weight="bold" if nivel == 0 else "normal", color=cor, va="center")
        fim = x0 + larg - 0.03
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        if bb.x1 + 0.01 < fim:
            fig.add_artist(plt.Line2D([bb.x1 + 0.01, fim], [y - 0.004] * 2, transform=fig.transFigure,
                                      color="#C8C8C8", lw=0.6, ls=(0, (1, 2))))
        fig.text(x0 + larg, y, str(pag), fontsize=fs, family=F_SEMI, color=cor, va="center", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def pagina_info(pdf, ctx: Contexto, num: int):
    d, P = ctx.dados, ctx.projeto
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "\n".join(d.linhas_titulo()), tam=17)
    cor = ctx.marca["cor"]
    # bloco produtor / propriedade
    for x, rot, val, sub in ((0.30, "PRODUTOR", d.produtor, d.municipio), (0.70, "PROPRIEDADE", d.propriedade, "")):
        fig.text(x, 0.815, rot, ha="center", fontsize=9, family=F_BOLD, weight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.45", fc=cor, ec="none"))
        fig.text(x, 0.782, val or "–", ha="center", fontsize=13, family=F_BOLD, weight="bold", color="#222222")
        if sub:
            fig.text(x, 0.762, sub, ha="center", fontsize=9, color="#555555")
    fig.add_artist(plt.Line2D([0.12, 0.88], [0.742, 0.742], transform=fig.transFigure, color=cor, lw=1.2))
    # informações técnicas
    n_pts = sum(len(t.amostras) for t in P.talhoes)
    area = P.area_total
    alt = ""
    if ctx.tem_alt:
        alts = np.concatenate([m.atributos["alt"][m.grade.mascara] for m in ctx.mapas])
        alt = f"{br(float(np.nanmean(alts)), 1)} m"
    info = [("Talhões", lista_talhoes(ctx.mapas, 30, 170)),
            ("Área total", f"{br(area, 2)} ha"),
            ("Data da coleta", d.data_coleta or "–"),
            ("Camada", "0 – 20 cm" + (" (+ 20 – 40 cm em pontos selecionados)" if ctx.subs is not None
                                      and len(ctx.subs) else "")),
            ("Número de pontos", f"{n_pts}  ·  1 ponto a cada {br(area / max(n_pts, 1), 1)} ha"),
            ("Subamostras por ponto", f"{d.subamostras}  ·  {n_pts * d.subamostras} coletas no total")]
    if alt:
        info.insert(2, ("Altitude média", alt))
    fig.text(0.12, 0.715, "Informações técnicas", fontsize=13, family=F_BOLD, weight="bold", color=cor)
    y = 0.685
    import textwrap
    for k, v in info:
        fig.text(0.13, y, f"{k}:", fontsize=10, family=F_SEMI, color="#333333")
        linhas_v = textwrap.wrap(str(v), 46) or [""]        # valores longos (muitos talhões) quebram em linhas
        if len(linhas_v) > 4:
            linhas_v = linhas_v[:4]
            linhas_v[-1] = linhas_v[-1][:43].rstrip(" ,") + "…"
        for j, lv in enumerate(linhas_v):
            fig.text(0.42, y - j * 0.019, lv, fontsize=10, family=F_BOLD, weight="bold", color="#111111")
        y -= 0.026 + 0.019 * (len(linhas_v) - 1)
    # tabela por talhão (em até 3 colunas quando há muitos talhões)
    y -= 0.008
    n = len(P.talhoes)
    ncol = 1 if n <= 5 else (2 if n <= 12 else 3)
    larg = 0.76 / ncol
    passo = 0.02 if n <= 12 else 0.016
    for c in range(ncol):
        x = 0.13 + c * larg
        cab = ["Talhão", "Área (ha)", "Pontos"] if ncol > 1 else ["Talhão", "Área (ha)", "Pontos", "ha/ponto"]
        dx = [0, larg * 0.40, larg * 0.72] if ncol > 1 else [0, 0.27, 0.45, 0.61]
        for d_, c_ in zip(dx, cab):
            fig.text(x + d_, y, c_, fontsize=8 if ncol > 1 else 8.5, family=F_BOLD, weight="bold", color=cor)
    linhas = int(np.ceil(n / ncol))
    for i, t in enumerate(P.talhoes):
        c, r = i // linhas, i % linhas
        x, yy = 0.13 + c * larg, y - (r + 1) * passo
        lim_nome = {1: 34, 2: 17, 3: 11}[ncol]              # o nome não invade a coluna da área
        nome_t = curto(t.nome)
        if len(nome_t) > lim_nome:                          # corta no meio: o número no fim identifica o talhão
            fim = (lim_nome - 1) // 2
            nome_t = nome_t[:lim_nome - 1 - fim].rstrip() + "…" + nome_t[-fim:].lstrip()
        vals = [nome_t, br(t.area_ha, 2), str(len(t.amostras))]
        dx = [0, larg * 0.40, larg * 0.72] if ncol > 1 else [0, 0.27, 0.45]
        if ncol == 1:
            vals.append(br(t.area_ha / max(len(t.amostras), 1), 1))
            dx.append(0.61)
        for d_, v in zip(dx, vals):
            fig.text(x + d_, yy, v, fontsize=7.8 if ncol > 1 else 8.5, color="#222222")
    y -= linhas * passo
    # clima
    y -= 0.04
    fig.add_artist(plt.Line2D([0.12, 0.88], [y + 0.018] * 2, transform=fig.transFigure, color=cor, lw=1.2))
    fig.text(0.12, y, "Informações climatológicas", fontsize=13, family=F_BOLD, weight="bold", color=cor)
    if ctx.clima and y - 0.14 >= 0.14:
        _grafico_clima(fig, ctx, [0.14, 0.09, 0.74, y - 0.14])
        fig.text(0.5, y - 0.025, f"Normais mensais de chuva e temperatura{(' — ' + d.municipio) if d.municipio else ''}. "
                                 f"Fonte: {ctx.clima['fonte']}.", ha="center", fontsize=7.5, color="#555555")
    else:
        fig.text(0.5, y - 0.08, "Dados climáticos indisponíveis no momento da geração.", ha="center",
                 fontsize=9, color="#777777")
    pdf.savefig(fig)
    plt.close(fig)


def _grafico_clima(fig, ctx, rect):
    c = ctx.clima
    L, B, W, H = rect
    meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    x = np.arange(12)
    a1 = fig.add_axes([L, B + H * 0.52, W, H * 0.42])
    a1.fill_between(x, c["tmin"], c["tmax"], color="#F4A259", alpha=0.18, lw=0)
    for serie, cor, nome in ((c["tmax"], "#D1495B", "Máxima"), (c["tmed"], "#555555", "Média"),
                             (c["tmin"], "#2E86AB", "Mínima")):
        a1.plot(x, serie, color=cor, lw=1.8, marker="o", ms=3.5)
        a1.text(11.3, serie[-1], f"{nome}", va="center", fontsize=7, color=cor)
    for i, v in enumerate(c["tmax"]):
        a1.text(i, v + 0.9, f"{v:.0f}°", ha="center", fontsize=6.5, color="#D1495B")
    for i, v in enumerate(c["tmin"]):
        a1.text(i, v - 2.2, f"{v:.0f}°", ha="center", fontsize=6.5, color="#2E86AB")
    a1.set_xlim(-0.6, 12.2)
    a1.set_ylim(min(c["tmin"]) - 4, max(c["tmax"]) + 4)
    a1.set_xticks([])
    a1.set_ylabel("°C", fontsize=8)
    a2 = fig.add_axes([L, B, W, H * 0.46])
    a2.bar(x, c["chuva"], color="#7FA7E8", width=0.7)
    for i, v in enumerate(c["chuva"]):
        a2.text(i, v + max(c["chuva"]) * 0.02, f"{v:.0f}", ha="center", fontsize=6.5, color="#35598F")
    a2.set_xlim(-0.6, 12.2)
    a2.set_xticks(x, meses, fontsize=7.5)
    a2.set_ylabel("Chuva (mm)", fontsize=8)
    a2.set_ylim(0, max(c["chuva"]) * 1.22)
    a2.text(5.5, max(c["chuva"]) * 1.2, f"Total anual: {br(sum(c['chuva']), 0)} mm", ha="center", va="top",
            fontsize=7.5, color="#35598F", family=F_SEMI)
    for a in (a1, a2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=7)


def _linha_textura(ctx) -> list[str]:
    if ctx.argila_est <= 0:
        return ["Areia, silte e argila pelo método da pipeta (Camargo et al., 1996)"]
    txt = ["Argila estimada pela CTC a pH 7 (laudo sem granulometria): Argila (%) ≈ −18,6 + 0,657 × CTC,",
           "   ou regressão local com as amostras que têm argila; usada só no cálculo do gesso"]
    if ctx.argila_est < 1:
        txt.insert(0, "Argila pelo método da pipeta (Camargo et al., 1996) onde disponível")
    return txt


def _linha_metodo_p(ctx) -> str:
    m = (ctx.dados.metodos or {}).get("p", "resina")
    if m == "Mehlich-1":
        return "Fósforo disponível (P) por Mehlich-1 (informado no laudo; regra de P calibrada para P resina)"
    if m in ("não informado", "ausente"):
        return "Fósforo disponível (P): método não informado no laudo (considerado P resina)"
    return "Fósforo disponível (P) por resina de troca aniônica (van Raij et al., 1996)"


def _linha_metodo_ph(ctx) -> str:
    m = (ctx.dados.metodos or {}).get("ph", "CaCl₂")
    if m == "água":
        return "pH em água (informado no laudo; classes da equipe são para pH em CaCl₂)"
    if m in ("não informado", "ausente"):
        return "pH: solução não informada no laudo (considerado CaCl₂ 0,01 M)"
    return "pH em CaCl₂ 0,01 M (relação 1:2,5 v/v)"


def _linhas_interpolacao(ctx) -> list[str]:
    op = ctx.dados.interpolacao
    fiel = op.get("modo") == "fiel"
    doses_estat = op.get("modo_doses", op.get("modo", "estatistico")) == "estatistico"
    estat = ["Semivariograma ajustado automaticamente (esférico, exponencial ou gaussiano; validação cruzada);",
             "   quando a malha amostral não detecta a estrutura espacial, usa-se um semivariograma padrão",
             ("   atributos sem estrutura espacial comprovada: valor médio uniforme"
              if op.get("sem_estrutura") == "uniforme" else
              "   atributos sem estrutura espacial comprovada: mapa exploratório (ver avisos)")]
    extremos = ("Pontos atípicos isolados são limitados à faixa dos vizinhos antes da interpolação (evita \"alvos\")"
                if op.get("limitar_extremos", True) else
                "Valores extremos mantidos como no laudo (sem limitação de atípicos)")
    if not fiel:
        return estat + [extremos]
    linhas = ["Mapas de fertilidade: krigagem local fiel às amostras — modelo esférico, alcance de 4× a distância",
              "   média entre pontos vizinhos e efeito pepita de 5%; cada amostra representa a sua célula da grade"]
    if doses_estat:
        linhas += ["Doses das prescrições: krigagem com semivariograma ajustado na área (zonas mais estáveis);",
                   "   " + extremos[0].lower() + extremos[1:]]
    else:
        linhas += ["Doses das prescrições calculadas sobre as mesmas superfícies dos mapas"]
    return linhas


def pagina_metodos(pdf, ctx: Contexto, num: int, par_txt: list[str]):
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "METODOLOGIAS\nE EQUAÇÕES", tam=22)
    cor = ctx.marca["cor"]
    blocos = [
        ("Atributos investigados (métodos usuais do laboratório — conferir no laudo)", [
            _linha_metodo_ph(ctx),
            "Matéria orgânica (M.O.) por oxidação com K₂Cr₂O₇ (Heanes, 1984)",
            "Acidez potencial (H+Al) pelo método SMP (Shoemaker et al., 1961)",
            "Acidez trocável (Al³⁺) extraída com KCl 1 M (Bertsch e Bloom, 1996)",
            "Ca²⁺, Mg²⁺ e K⁺ trocáveis por resina de troca iônica (van Raij et al., 2001)",
            _linha_metodo_p(ctx),
            "Sulfato (SO₄²⁻) extraído com Ca(H₂PO₄)₂ (Fox et al., 1964)",
        ] + _linha_textura(ctx)),
        ("Mapeamento", [
            "Pontos georreferenciados ligados às amostras pelo número (identificação × nome do ponto)",
            "Interpolação de todos os atributos por krigagem ordinária"] + _linhas_interpolacao(ctx) + [
            f"Grade de {br(ctx.mapas[0].grade.res, 0)} × {br(ctx.mapas[0].grade.res, 0)} m recortada pelo perímetro "
            "de cada talhão",
            "Classes dos mapas de fertilidade: tabela de interpretação da equipe técnica",
            "Zonas de prescrição com largura mínima de 30 m e área mínima de 0,5 ha (aplicáveis a campo)"]
            + ([ctx.fonte_altitude] if ctx.fonte_altitude else [])
            + (["NDVI: Copernicus Sentinel-2 L2A, 20 m, últimos 12 meses"] if ctx.ndvi is not None else [])),
        ("Parâmetros de recomendação", par_txt),
    ]
    y = 0.80
    for tit, linhas in blocos:
        fig.add_artist(FancyBboxPatch((0.11, y - 0.004), 0.78, 0.026, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc=cor, ec="none", alpha=0.95))
        fig.text(0.5, y + 0.009, tit, ha="center", va="center", fontsize=10.5, family=F_BOLD, weight="bold",
                 color="white")
        y -= 0.03
        for l in linhas:
            fig.text(0.13, y, "›", fontsize=10, color=ctx.marca["acento"], family=F_BOLD, va="center")
            fig.text(0.15, y, l, fontsize=8.6, color="#222222", va="center")
            y -= 0.021
        y -= 0.02
    pdf.savefig(fig)
    plt.close(fig)


def pagina_pontos(pdf, ctx: Contexto, num: int, mapas=None, primeiro=1, rotulo_grupo=""):
    mapas = mapas or ctx.mapas
    talhoes = [m.talhao for m in mapas]
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "MAPA DE PONTOS", rotulo_grupo)
    P = ctx.projeto
    from pyproj import Transformer
    t3857 = Transformer.from_crs(P.epsg, 3857, always_xy=True)
    from shapely.ops import transform as sh_tr
    geoms = [sh_tr(t3857.transform, t.perimetro) for t in talhoes]
    b = np.array([g.bounds for g in geoms])
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    mx, my = (x1 - x0) * 0.12, (y1 - y0) * 0.12
    lim = (x0 - mx, y0 - my, x1 + mx, y1 + my)
    ax = _caixa_mapa(fig, ctx, [0.10, 0.12, 0.80, 0.72], lim)
    if ctx.satelite is not None:
        img, ext, cred = ctx.satelite
        ax.imshow(img, extent=ext, zorder=1, interpolation="bilinear")
        ax.set_xlim(lim[0], lim[2])
        ax.set_ylim(lim[1], lim[3])
        import textwrap
        fig.text(0.82, 0.071, "\n".join(textwrap.wrap(cred, 62)), ha="right", va="bottom", fontsize=5.5,
                 color="#666666", linespacing=1.15)
    else:
        ax.add_artist(Rectangle((lim[0], lim[1]), lim[2] - lim[0], lim[3] - lim[1], color="#EEF2EE", zorder=0))
    k = primeiro - 1
    muitos = sum(len(t.amostras) for t in talhoes) > 150
    for t, g in zip(talhoes, geoms):
        ax.add_artist(_patch(g, fc="#8C8C8C", alpha=0.72, ec="black", lw=1.2, zorder=3))
        px, py = t3857.transform(t.amostras["x"].values, t.amostras["y"].values)
        ax.scatter(px, py, s=5 if muitos else 9, c="#D7263D", zorder=5, lw=0)
        for xx, yy in zip(px, py):
            k += 1
            ax.text(xx, yy + (lim[3] - lim[1]) * 0.008, str(k), ha="center", va="bottom",
                    fontsize=4.2 if muitos else 6.3, color="white", zorder=6, family=F_SEMI)
        c = g.representative_point()
        ax.text(c.x, c.y, curto(t.nome), ha="center", va="center", fontsize=9 if len(talhoes) > 6 else 15,
                family=F_XB, color="white", zorder=7,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground="#333333")])
    escala(fig, ax, 0.12, 0.088)
    norte(fig)
    pdf.savefig(fig)
    plt.close(fig)
    return k


def pagina_diagnostico(pdf, ctx: Contexto, num: int):
    """Resumo por talhão: classes médias de todos os atributos (cores do book)."""
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "DIAGNÓSTICO GERAL", "média de cada talhão · camada 0–20 cm")
    cor = ctx.marca["cor"]
    chaves = [k for k in ORDEM_FERT if all(k in m.atributos for m in ctx.mapas)]
    tal = [curto(m.talhao.nome) for m in ctx.mapas]
    medias = {k: [float(np.nanmean(m.atributos[k])) for m in ctx.mapas] for k in chaves}
    area = [m.talhao.area_ha for m in ctx.mapas]
    geral = {k: float(np.average(v, weights=area)) for k, v in medias.items()}
    cols = tal + (["Fazenda"] if len(tal) > 1 else [])
    # coluna de rótulos com largura fixa na página (0,25), qualquer que seja o nº de talhões
    larg_ax, larg_rot = 0.78, 0.25
    L = larg_rot * len(cols) / (larg_ax - larg_rot)
    ax = fig.add_axes([0.11, 0.2, larg_ax, 0.6])
    ax.set_xlim(0, len(cols) + L)
    ax.set_ylim(len(chaves), -1.1)
    ax.axis("off")
    fs_cel = 8.3 if len(cols) <= 5 else (7 if len(cols) <= 9 else 5.8)
    for j, c in enumerate(cols):
        ax.text(L + j + 0.5, -0.45, c, ha="center", va="center", fontsize=fs_cel + 0.4, family=F_BOLD,
                weight="bold", color=cor, rotation=0 if len(cols) <= 9 else 90)
    for i, k in enumerate(chaves):
        t, s, u = TITULOS[k]
        nome = NOMES[k].replace(" disponível", "")
        if len(nome) > 18:
            nome = {"ctc": "CTC", "v": "Saturação bases", "sat_ca": "Saturação Ca", "sat_mg": "Saturação Mg", "sat_k": "Saturação K",
                    "m": "Saturação Al"}.get(k, CURTOS.get(k, nome))
        ax.text(0, i + 0.5, nome, va="center", fontsize=8, family=F_SEMI, color="#222222")
        ax.text(L - 0.06 * L, i + 0.5, u.replace("adimensional", ""), va="center", ha="right", fontsize=6.2,
                color="#777777")
        vals = medias[k] + ([geral[k]] if len(tal) > 1 else [])
        for j, v in enumerate(vals):
            cl = it.classificar(k, v)
            fc = COR_CLASSE.get(cl, "#DDDDDD")
            ax.add_artist(Rectangle((L + j + 0.04, i + 0.07), 0.92, 0.86, color=fc, lw=0))
            ax.text(L + j + 0.5, i + 0.5, br(v, casas_de(k, v)), ha="center", va="center", fontsize=fs_cel,
                    family=F_BOLD, weight="bold", color=cor_texto(fc))
    # legenda
    for i, cl in enumerate(it.CLASSES[::-1]):
        x = 0.12 + i * 0.095
        fig.add_artist(Rectangle((x, 0.165), 0.018, 0.012, transform=fig.transFigure, color=COR_CLASSE[cl]))
        fig.text(x + 0.022, 0.171, cl, fontsize=6.6, va="center", color="#333333")
    # pontos de atenção: texto corrido, itens separados por ";", sempre acima da moldura
    y = 0.14
    fig.text(0.12, y, "Pontos de atenção", fontsize=10.5, family=F_BOLD, weight="bold", color=cor)
    if len(tal) <= 4:
        itens = []
        for nome in tal:
            ruins = [CURTOS[k] for k in chaves if it.classificar(k, medias[k][tal.index(nome)]) in it.ATENCAO]
            itens.append(f"{nome}: " + (", ".join(ruins) if ruins else "nenhum atributo em classe baixa"))
    else:                                   # muitos talhões: agrupa por atributo
        itens = []
        for k in chaves:
            ruins = [nome for nome, v in zip(tal, medias[k]) if it.classificar(k, v) in it.ATENCAO]
            if ruins:
                itens.append(f"{CURTOS[k]}: " + ("todos os talhões" if len(ruins) == len(tal) else ", ".join(ruins)))
    texto_corrido(fig, 0.13, y - 0.019, "; ".join(itens) + "." if itens else "Nenhum atributo em classe baixa.",
                  larg=0.75, y_min=MOLD_Y0 + 0.018)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_subsuperficial(pdf, ctx: Contexto, num: int):
    s = ctx.subs
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "CAMADA 20–40 cm", "amostras subsuperficiais (informativo)")
    cor = ctx.marca["cor"]
    chaves = [k for k in ["ph", "ca", "mg", "k", "al", "m", "v", "ctc", "s", "p", "argila"]
              if k in s and s[k].notna().any()]
    x0, larg_id = 0.12, 0.13
    wc = (0.88 - x0 - larg_id) / len(chaves)
    y = 0.80
    fig.text(x0, y, "Talhão · ponto", fontsize=7.5, family=F_BOLD, weight="bold", color=cor, va="center")
    for j, k in enumerate(chaves):
        fig.text(x0 + larg_id + wc * (j + 0.5), y, CURTOS[k], ha="center", va="center", fontsize=7.5,
                 family=F_BOLD, weight="bold", color=cor)
    hl = min(0.028, 0.45 / max(len(s), 1))
    for _, r in s.iterrows():
        y -= hl
        pt = "–" if pd.isna(r.get("ponto")) else f"{int(r['ponto'])}"
        fig.text(x0, y, f"{curto(r['_talhao'])} · {pt}", va="center", fontsize=7.5)
        for j, k in enumerate(chaves):
            v = exibir(k, r[k])
            cl = it.classificar(k, v) if k in it.ATRIBUTOS else None
            if cl:
                fig.add_artist(Rectangle((x0 + larg_id + wc * j + 0.002, y - hl * 0.42), wc - 0.004, hl * 0.84,
                                         transform=fig.transFigure, color=COR_CLASSE[cl], lw=0))
            fig.text(x0 + larg_id + wc * (j + 0.5), y, br(v, casas_de(k, v if pd.notna(v) else 0)), ha="center",
                     va="center", fontsize=7.3, family=F_BOLD, weight="bold",
                     color=cor_texto(COR_CLASSE[cl]) if cl else "#1a1a1a")
    txt = ("A camada de 20–40 cm não entra no cálculo das doses nem nos mapas. Ela indica se há\n"
           "impedimento químico em profundidade (Ca baixo, Al ou m% elevados), o que reforça a\n"
           "necessidade de gessagem para aprofundar o sistema radicular.")
    fig.text(x0, y - 0.05, txt, fontsize=9, color="#333333", va="top", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_volumes(pdf, ctx: Contexto, num: int, tabela: pd.DataFrame, parte: int = 0, por_pagina: int = 18):
    """Dose média e total por talhão (linhas) e produto (colunas); totais da propriedade no fim."""
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "VOLUMES DE PRODUTO", "por talhão e total da propriedade" + (f" · parte {parte + 1}" if parte else ""))
    cor = ctx.marca["cor"]
    prods = list(dict.fromkeys(tabela["Produto"]))
    talhoes = list(dict.fromkeys(tabela["Talhão"]))
    pagina_tal = talhoes[parte * por_pagina:(parte + 1) * por_pagina]
    ultima = (parte + 1) * por_pagina >= len(talhoes)
    rot = lambda p: p.replace("P2O5", "P₂O₅").replace(" aplicação", " apl.").replace("Produto ", "")  # noqa: E731
    npc = len(prods)
    x0, x_area = 0.12, 0.27
    larg = (0.88 - 0.34) / npc
    y = 0.80
    for bloco, (campo, casas, titulo_b) in enumerate((("Dose média (kg/ha)", 0, "Dose média (kg/ha)"),
                                                      ("Total (t)", 2, "Total de produto (t)"))):
        fig.add_artist(FancyBboxPatch((0.11, y - 0.008), 0.78, 0.026, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc=cor, ec="none"))
        fig.text(0.13, y + 0.005, titulo_b, va="center", fontsize=9.5, family=F_BOLD, weight="bold", color="white")
        y -= 0.026
        fig.text(x0 + 0.01, y, "Talhão", fontsize=7.8, family=F_BOLD, weight="bold", color=cor)
        fig.text(x_area + 0.03, y, "Área (ha)", fontsize=7.8, family=F_BOLD, weight="bold", color=cor, ha="right")
        for j, p in enumerate(prods):
            fig.text(0.34 + larg * (j + 0.5), y, rot(p), fontsize=7.2 if npc > 4 else 7.8, family=F_BOLD,
                     weight="bold", color=cor, ha="center")
        passo = min(0.019, 0.26 / max(len(pagina_tal) + 1, 1))
        for tal in pagina_tal:
            y -= passo
            t = tabela[tabela["Talhão"] == tal]
            fig.text(x0 + 0.01, y, nome_curto(tal, 19), fontsize=8, color="#222222")
            fig.text(x_area + 0.03, y, br(t["Área (ha)"].iloc[0], 2), fontsize=8, color="#222222", ha="right")
            for j, p in enumerate(prods):
                v = t.loc[t["Produto"] == p, campo]
                fig.text(0.34 + larg * (j + 0.5), y, br(float(v.iloc[0]), casas) if len(v) else "–", fontsize=8,
                         color="#222222", ha="center")
        if ultima:
            y -= passo + 0.004
            fig.add_artist(plt.Line2D([0.12, 0.88], [y + passo * 0.7] * 2, transform=fig.transFigure,
                                      color="#BBBBBB", lw=0.6))
            area_tot = tabela.drop_duplicates("Talhão")["Área (ha)"].sum()
            fig.text(x0 + 0.01, y, "Propriedade", fontsize=8, family=F_BOLD, weight="bold", color=cor)
            fig.text(x_area + 0.03, y, br(area_tot, 2), fontsize=8, family=F_BOLD, weight="bold", color=cor, ha="right")
            for j, p in enumerate(prods):
                tp = tabela[tabela["Produto"] == p]
                v = tp["Total (t)"].sum() if campo == "Total (t)" else tp["Total (t)"].sum() * 1000 / tp["Área (ha)"].sum()
                fig.text(0.34 + larg * (j + 0.5), y, br(v, casas), fontsize=8, family=F_BOLD, weight="bold",
                         color=cor, ha="center")
        y -= 0.05
    if ultima and y > 0.26:
        tot = tabela.groupby("Produto", sort=False)["Total (t)"].sum()
        axb = fig.add_axes([0.32, 0.12, 0.52, min(y - 0.16, 0.03 * len(tot) + 0.02)])
        axb.barh(range(len(tot)), tot.values, color=cor, height=0.6)
        for i, v in enumerate(tot.values):
            axb.text(v, i, f"  {br(v, 1)} t", va="center", fontsize=8, color="#222222")
        axb.set_yticks(range(len(tot)), [rot(t) for t in tot.index], fontsize=8)
        axb.invert_yaxis()
        axb.set_xlim(0, tot.max() * 1.25)
        axb.set_xticks([])
        for sp in ("top", "right", "bottom"):
            axb.spines[sp].set_visible(False)
        axb.tick_params(length=0)
    fig.text(0.12, 0.085, "Volumes calculados pela soma (dose da zona × área da zona) dos mapas de prescrição.",
             fontsize=7, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def _pos_condicao(chave, v):
    """Posição 0–8 na escala de condição (0 = crítico … 8 = muito alto), contínua dentro da classe."""
    faixas = sorted(it.ATRIBUTOS[chave][2], key=lambda t: t[1])
    lim = [faixas[0][1]] + [f[2] for f in faixas]
    p = float(np.interp(v, lim, np.arange(len(lim))))          # 0..8 no eixo de valores
    return it.posicao_invertida(p) if chave in it.INVERTIDOS else p


def pagina_panorama(pdf, ctx: Contexto, num: int):
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "PANORAMA DA FERTILIDADE", "posição da média de cada talhão na escala de interpretação", tam=20)
    cor = ctx.marca["cor"]
    chaves = [k for k in ORDEM_FERT if k in it.ATRIBUTOS and all(k in m.atributos for m in ctx.mapas)]
    tal = [curto(m.talhao.nome) for m in ctx.mapas]
    marc = ["o", "^", "s", "D", "v", "P"]
    ax = fig.add_axes([0.30, 0.15, 0.56, 0.66])
    n = len(chaves)
    ax.set_xlim(0, 8)
    ax.set_ylim(n, -0.8)
    ax.axis("off")
    ordem = it.CLASSES[::-1]                                    # crítico → muito alto
    for j, cl in enumerate(ordem):
        ax.text(j + 0.5, -0.35, cl.replace("Muito ", "M. "), ha="center", va="center", fontsize=6.3,
                color="#555555", rotation=0)
    for i, k in enumerate(chaves):
        for j, cl in enumerate(ordem):
            ax.add_artist(Rectangle((j + 0.02, i + 0.3), 0.96, 0.4, color=COR_CLASSE[cl], alpha=0.85, lw=0))
        ax.text(-0.15, i + 0.5, NOMES[k].replace(" disponível", ""), ha="right", va="center", fontsize=7.8,
                color="#222222", clip_on=False)
        if len(tal) <= 5:
            for t, (m, mk) in enumerate(zip(ctx.mapas, marc)):
                v = float(np.nanmean(m.atributos[k]))
                p = np.clip(_pos_condicao(k, v), 0.08, 7.92)
                ax.scatter([p], [i + 0.5 + (t - (len(tal) - 1) / 2) * 0.16], marker=mk, s=34, color="white",
                           edgecolor="#111111", lw=1.1, zorder=5)
        else:                               # muitos talhões: faixa entre talhões + média da fazenda
            ps = [np.clip(_pos_condicao(k, float(np.nanmean(m.atributos[k]))), 0.08, 7.92) for m in ctx.mapas]
            area = [m.talhao.area_ha for m in ctx.mapas]
            pm = np.clip(_pos_condicao(k, float(np.average([np.nanmean(m.atributos[k]) for m in ctx.mapas],
                                                             weights=area))), 0.08, 7.92)
            ax.plot([min(ps), max(ps)], [i + 0.5] * 2, color="#111111", lw=1.6, solid_capstyle="round", zorder=4)
            ax.scatter([pm], [i + 0.5], marker="D", s=30, color="white", edgecolor="#111111", lw=1.1, zorder=5)
    if len(tal) <= 5:
        for t, (nome, mk) in enumerate(zip(tal, marc)):
            x = 0.30 + t * 0.12
            fig.add_artist(plt.Line2D([x], [0.125], marker=mk, color="white", markeredgecolor="#111111",
                                      markersize=6.5, transform=fig.transFigure))
            fig.text(x + 0.015, 0.125, nome, va="center", fontsize=8, color="#222222")
    else:
        fig.add_artist(plt.Line2D([0.30, 0.34], [0.125] * 2, color="#111111", lw=1.6, transform=fig.transFigure))
        fig.text(0.35, 0.125, "variação entre talhões", va="center", fontsize=8, color="#222222")
        fig.add_artist(plt.Line2D([0.60], [0.125], marker="D", color="white", markeredgecolor="#111111",
                                  markersize=6, transform=fig.transFigure))
        fig.text(0.615, 0.125, "média da fazenda (ponderada pela área)", va="center", fontsize=8, color="#222222")
    fig.text(0.12, 0.105, "Cada marcador mostra a média do talhão; mais à direita = teor mais alto na escala da equipe "
                          "(H+Al, Al e m% já invertidos).\nPara pH, V%, Ca na CTC e B, a classe 'Muito alto' pode indicar "
                          "excesso — não necessariamente condição melhor.",
             fontsize=7.2, color="#666666", va="top", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def pagina_area_classes(pdf, ctx: Contexto, num: int):
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "ÁREA POR CLASSE", "participação da área total em cada condição", tam=22)
    chaves = [k for k in ORDEM_FERT if k in it.ATRIBUTOS and all(k in m.atributos for m in ctx.mapas)]
    ordem = it.CLASSES[::-1]
    ax = fig.add_axes([0.32, 0.17, 0.56, 0.64])
    n = len(chaves)
    ax.set_xlim(0, 100)
    ax.set_ylim(n, 0)
    ax.axis("off")
    for i, k in enumerate(chaves):
        vals = np.concatenate([m.atributos[k][m.grade.mascara] for m in ctx.mapas])
        vals = vals[np.isfinite(vals)]
        limites, cores_k, _ = escala_classes(k)
        idx = np.clip(np.searchsorted(limites, vals, side="right") - 1, 0, len(cores_k) - 1)
        cont = somar_por_condicao(k, np.bincount(idx, minlength=len(cores_k)))
        tot = len(vals)
        x = 0.0
        for cl in ordem:
            pct = 100 * cont.get(cl, 0) / tot if tot else 0
            if pct <= 0:
                continue
            ax.add_artist(Rectangle((x, i + 0.18), pct, 0.64, color=COR_CLASSE[cl], lw=0))
            if pct >= 9:
                ax.text(x + pct / 2, i + 0.5, f"{pct:.0f}%", ha="center", va="center", fontsize=6.8,
                        color=cor_texto(COR_CLASSE[cl]), family=F_SEMI)
            x += pct
        ax.text(-1.5, i + 0.5, NOMES[k].replace(" disponível", ""), ha="right", va="center", fontsize=7.8,
                color="#222222", clip_on=False)
    for i, cl in enumerate(ordem):
        x = 0.12 + i * 0.097
        fig.add_artist(Rectangle((x, 0.135), 0.016, 0.011, transform=fig.transFigure, color=COR_CLASSE[cl]))
        fig.text(x + 0.02, 0.1405, cl, fontsize=6.5, va="center", color="#333333")
    fig.text(0.5, 0.105, f"Área mapeada: {br(ctx.projeto.area_total, 2)} ha (todos os talhões)", ha="center",
             fontsize=7.5, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def pagina_bases(pdf, ctx: Contexto, num: int, alvo_ca=55, alvo_mg=19.4):
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    titulo(fig, ctx, "EQUILÍBRIO DE BASES", "composição da CTC de cada talhão", tam=22)
    cor, ac = ctx.marca["cor"], ctx.marca["acento"]
    tal = [curto(m.talhao.nome) for m in ctx.mapas]
    comp = []
    for m in ctx.mapas:
        A = m.atributos
        ctc = np.nanmean(A["ctc"])
        ca, mg, k, hal = (np.nanmean(A[c]) for c in ("ca", "mg", "k", "hal"))
        tot = ca + mg + k + hal
        comp.append({"Ca²⁺": 100 * ca / tot, "Mg²⁺": 100 * mg / tot, "K⁺": 100 * k / tot, "H+Al": 100 * hal / tot,
                     "ctc": ctc, "ca": ca, "mg": mg, "k": k})
    cores = {"Ca²⁺": cor, "Mg²⁺": _mistura(cor, "white", 0.45), "K⁺": ac, "H+Al": "#BDBDBD"}
    ax = fig.add_axes([0.2, 0.52, 0.68, 0.26])
    for i, c in enumerate(comp):
        x = 0
        for nome in ("Ca²⁺", "Mg²⁺", "K⁺", "H+Al"):
            v = c[nome]
            ax.barh(i, v, left=x, color=cores[nome], height=0.58, edgecolor="white", lw=1.2)
            if v >= 4:
                ax.text(x + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color=cor_texto(matplotlib.colors.to_hex(cores[nome])), family=F_SEMI)
            x += v
    for alvo, rot, yy in ((alvo_ca, "meta Ca 55%", -0.62), (alvo_ca + alvo_mg, "meta Ca+Mg 74%", -0.62)):
        ax.axvline(alvo, color="#111111", lw=0.9, ls=(0, (3, 2)))
        ax.text(alvo, yy, rot, ha="center", va="bottom", fontsize=7, color="#111111")
    ax.set_yticks(range(len(tal)), tal, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(tal) - 0.5, -0.8)
    ax.set_xlabel("% da CTC", fontsize=8)
    ax.tick_params(labelsize=8, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    for j, nome in enumerate(cores):
        x = 0.2 + j * 0.13
        fig.add_artist(Rectangle((x, 0.47), 0.018, 0.012, transform=fig.transFigure, color=cores[nome]))
        fig.text(x + 0.024, 0.476, nome, fontsize=8, va="center", color="#333333")
    # relações entre bases
    y = 0.41
    fig.text(0.12, y, "Relações entre bases (médias)", fontsize=11, family=F_BOLD, weight="bold", color=cor)
    y -= 0.03
    cab = ["Talhão", "CTC (mmolc/dm³)", "Ca/Mg", "Ca/K", "Mg/K"]
    xs = [0.13, 0.33, 0.53, 0.67, 0.81]
    for x, c in zip(xs, cab):
        fig.text(x, y, c, fontsize=8.5, family=F_BOLD, weight="bold", color=cor)
    passo = min(0.024, 0.28 / max(len(tal), 1))
    for nome, c in zip(tal, comp):
        y -= passo
        for x, v in zip(xs, [nome, br(c["ctc"], 1), br(c["ca"] / c["mg"], 1), br(c["ca"] / c["k"], 1),
                             br(c["mg"] / c["k"], 1)]):
            fig.text(x, y, v, fontsize=9 if len(tal) <= 8 else 7.5, color="#222222")
    fig.text(0.12, y - 0.05, "A calagem recomendada busca elevar o Ca a cerca de 55% e o Mg a cerca de 19% da CTC.\n"
                             "Barras à esquerda das metas indicam onde a correção tem maior efeito.",
             fontsize=8, color="#555555", va="top", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


# ------------------------------------------------------------------ montagem

# ------------------------------------------------------------------- NDVI
MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def pagina_ndvi(pdf, ctx: Contexto, num: int, mapas=None, rotulo_grupo: str = ""):
    """Pico de vigor (NDVI máximo em 12 meses), série temporal e média por talhão."""
    from matplotlib.colors import BoundaryNorm
    from . import ndvi as nd
    R = ctx.ndvi
    mapas = mapas or ctx.mapas
    fig = plt.figure(figsize=A4)
    base(fig, ctx, num)
    sub = "NDVI · pico de vigor da vegetação nos últimos 12 meses"
    if rotulo_grupo:
        sub += f"  ·  {rotulo_grupo}"
    titulo(fig, ctx, "ÍNDICE DE VEGETAÇÃO", sub)

    lim = _limites(mapas)
    largo = (lim[3] - lim[1]) / (lim[2] - lim[0]) < 0.95
    ax = _caixa_mapa(fig, ctx, [0.09, 0.50, 0.82, 0.33] if largo else [0.17, 0.455, 0.66, 0.375], lim)
    # classes ajustadas à variação da própria área (percentis 2–98): com faixas fixas, uma lavoura fechada cai toda
    # na classe mais alta e não se vê diferença dentro do talhão
    from shapely import contains_xy as _cxy
    from shapely.ops import unary_union as _uni
    _xs = R.extent[0] + (np.arange(R.pico.shape[1]) + 0.5) * R.res
    _ys = R.extent[3] - (np.arange(R.pico.shape[0]) + 0.5) * R.res
    _XX, _YY = np.meshgrid(_xs, _ys)
    _v = R.pico[_cxy(_uni([m.talhao.perimetro for m in mapas]), _XX, _YY) & np.isfinite(R.pico)]
    n_cl = len(nd.CORES_NDVI)
    if len(_v):
        p_lo, p_hi = float(np.percentile(_v, 2)), float(np.percentile(_v, 98))
        if p_hi - p_lo < 0.01 * n_cl:
            c_ = (p_lo + p_hi) / 2
            p_lo, p_hi = c_ - 0.005 * n_cl, c_ + 0.005 * n_cl
        internos = np.round(np.linspace(p_lo, p_hi, n_cl - 1), 2)
        internos = np.maximum.accumulate(internos + np.arange(n_cl - 1) * 1e-6)
        lims = [-1.0] + internos.tolist() + [1.0]
        rotulos = ([f"< {br(internos[0], 2)}"] + [f"{br(a, 2)}–{br(b, 2)}" for a, b in zip(internos[:-1], internos[1:])]
                   + [f"> {br(internos[-1], 2)}"])
    else:
        lims = [lo for lo, _ in nd.FAIXAS_NDVI] + [1.0]
        rotulos = nd.ROTULOS_NDVI
    cmap = ListedColormap(nd.CORES_NDVI)
    norm = BoundaryNorm(lims, cmap.N)
    from scipy import ndimage as ndi
    from shapely.ops import unary_union as _uu
    Z = R.pico
    faltam = ~np.isfinite(Z)
    if faltam.any() and (~faltam).any():                  # estende 2 px para o recorte pelo perímetro
        dist, (ii, jj) = ndi.distance_transform_edt(faltam, return_indices=True)
        Z = np.where(faltam & (dist <= 2.5), Z[ii, jj], Z)
    im = ax.imshow(Z, extent=R.extent, cmap=cmap, norm=norm, interpolation="nearest", zorder=2)
    recorte = _patch(_uu([m.talhao.perimetro for m in mapas]), fc="none", ec="none")
    ax.add_artist(recorte)
    im.set_clip_path(recorte)
    for m in mapas:
        ax.add_artist(_patch(m.talhao.perimetro, fc="none", ec="#111111", lw=0.8, zorder=5))
        if len(mapas) > 1:
            c = m.talhao.perimetro.representative_point()
            ax.text(c.x, c.y, curto(m.talhao.nome), ha="center", va="center",
                    fontsize=6.5 if len(mapas) > 6 else 8, family=F_BOLD, weight="bold", color="#111111", zorder=6,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="white")])
    escala(fig, ax, 0.115, 0.455 if largo else 0.425)

    # legenda horizontal com % da área em cada classe
    sel = np.zeros(R.pico.shape, bool)
    xs = R.extent[0] + (np.arange(R.pico.shape[1]) + 0.5) * R.res
    ys = R.extent[3] - (np.arange(R.pico.shape[0]) + 0.5) * R.res
    from shapely import contains_xy
    from shapely.ops import unary_union
    XX, YY = np.meshgrid(xs, ys)
    sel = contains_xy(unary_union([m.talhao.perimetro for m in mapas]), XX, YY) & np.isfinite(R.pico)
    v = R.pico[sel]
    cls = np.clip(np.searchsorted(lims, v, side="right") - 1, 0, len(nd.CORES_NDVI) - 1)
    pct = np.bincount(cls, minlength=len(nd.CORES_NDVI)) / max(len(v), 1) * 100
    yl = 0.405 if largo else 0.385
    w = 0.82 / len(nd.CORES_NDVI)
    fig.text(0.09, yl + 0.028, "NDVI (pico)", fontsize=8, family=F_SEMI, color=ctx.marca["cor"])
    fig.text(0.19, yl + 0.028, "classes ajustadas à variação da área (percentis 2 a 98)", fontsize=6.6, color="#777777")
    for i, (cor, rot) in enumerate(zip(nd.CORES_NDVI, rotulos)):
        x = 0.09 + i * w
        fig.patches.append(matplotlib.patches.Rectangle((x, yl), w * 0.97, 0.018, transform=fig.transFigure,
                                                        fc=cor, ec="none"))
        fig.text(x + w * 0.485, yl + 0.009, f"{pct[i]:.0f}%", ha="center", va="center", fontsize=7,
                 family=F_SEMI, color=cor_texto(cor))
        fig.text(x + w * 0.485, yl - 0.011, rot, ha="center", va="center", fontsize=6.8, color="#333333")
    media = float(np.nanmean(v)) if len(v) else float("nan")
    fig.text(0.91, yl + 0.028, f"Média do pico: {br(media, 2)}", ha="right", fontsize=9.5, family=F_BOLD,
             weight="bold", color=ctx.marca["cor"])

    # série temporal
    nomes = [m.talhao.nome for m in mapas]
    datas = R.datas
    axs = fig.add_axes([0.12, 0.158, 0.50, 0.172])
    series = np.array([R.serie.get(n, [np.nan] * len(datas)) for n in nomes], float)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            media_s = np.nanmean(series, axis=0)
            lo_s, hi_s = np.nanmin(series, axis=0), np.nanmax(series, axis=0)
    x = np.arange(len(datas))
    if len(nomes) > 1:
        axs.fill_between(x, lo_s, hi_s, color=ctx.marca["cor"], alpha=0.13, lw=0, label="Faixa entre talhões")
        if len(nomes) <= 6:
            for n, ser in zip(nomes, series):
                axs.plot(x, ser, color="#9AA59E", lw=0.8)
                ok = np.where(np.isfinite(ser))[0]
                if len(ok):
                    axs.text(ok[-1] + 0.15, ser[ok[-1]], curto(n), fontsize=5.8, color="#6B756E", va="center")
    axs.plot(x, media_s, color=ctx.marca["cor"], lw=2, marker="o", ms=3.5,
             label="Média" + (" dos talhões" if len(nomes) > 1 else ""))
    for i, vv in enumerate(media_s):
        if np.isfinite(vv):
            axs.text(i, vv + 0.035, br(vv, 2), ha="center", fontsize=5.8, color=ctx.marca["cor"])
    axs.set_xticks(x, [f"{MESES_PT[d.month - 1]}/{d.year % 100:02d}\n{d.day:02d}" for d in datas], fontsize=6.2)
    axs.set_ylim(0, 1)
    axs.set_xlim(-0.5, len(datas) - 0.5 + (0.9 if 1 < len(nomes) <= 6 else 0))
    axs.set_ylabel("NDVI médio", fontsize=7.5)
    axs.tick_params(labelsize=6.5)
    axs.grid(axis="y", color="#E3E3E3", lw=0.6)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    axs.legend(fontsize=6.2, frameon=False, loc="lower left", ncol=2)
    fig.text(0.12, 0.343, "Evolução do NDVI (datas sem nuvens)", fontsize=8.5, family=F_SEMI,
             color=ctx.marca["cor"])

    # tabela: pico médio e variabilidade por talhão
    fig.text(0.665, 0.343, "Pico por talhão  (CV%)", fontsize=8.5, family=F_SEMI, color=ctx.marca["cor"])
    n = len(nomes)
    colunas = 1 if n <= 13 else 2
    por_col = int(np.ceil(n / colunas))
    passo = min(0.0145, 0.19 / max(por_col, 1))
    for i, nome in enumerate(nomes):
        cx = 0.665 + (i // por_col) * 0.125
        cy = 0.323 - (i % por_col) * passo
        pv = R.pico_talhao.get(nome, np.nan)
        cor = nd.CORES_NDVI[nd.classe_ndvi(pv)] if np.isfinite(pv) else "#DDDDDD"
        fig.patches.append(matplotlib.patches.Rectangle((cx, cy - 0.0045), 0.012, 0.009, transform=fig.transFigure,
                                                        fc=cor, ec="none"))
        cv = R.cv_talhao.get(nome, np.nan)
        fig.text(cx + 0.016, cy, f"{curto(nome)}: {br(pv, 2)}" + (f" ({br(cv, 0)})" if np.isfinite(cv) else ""),
                 va="center", fontsize=6.6 if colunas == 1 else 6.0, color="#333333")

    ini, fim = R.periodo
    fig.text(0.10, 0.104, f"{len(datas)} imagens sem nuvens entre {ini:%d/%m/%Y} e {fim:%d/%m/%Y}. "
                          "Pico = maior NDVI de cada pixel no período.", fontsize=6.3, color="#555555")
    fig.text(0.10, 0.092, "Nuvens e sombras removidas pela máscara SCL. CV = variação do NDVI dentro do talhão.",
             fontsize=6.3, color="#555555")
    fig.text(0.10, 0.080, nd.CREDITO, fontsize=5.6, color="#777777")
    norte(fig)
    pdf.savefig(fig)
    plt.close(fig)

def preparar_externos(ctx: Contexto, progresso=None):
    """Satélite, altitude e clima (se houver internet)."""
    if not ctx.dados.usar_internet:
        ctx.avisos.append("Internet desativada: sem imagem de satélite, altitude e clima")
        return
    P = ctx.projeto
    from shapely.ops import unary_union
    todos = unary_union([t.perimetro_ll for t in P.talhoes])
    lon0, lat0, lon1, lat1 = todos.bounds
    c = todos.centroid
    if progresso:
        progresso("Buscando dados climáticos (NASA POWER)…")
    ctx.clima = externos.clima(c.x, c.y)
    if ctx.clima is None:
        ctx.avisos.append("Clima: não foi possível consultar a NASA POWER")
    if ctx.dados.fonte_satelite:
        if progresso:
            progresso("Baixando imagem de satélite…")
        dl, dt = (lon1 - lon0) * 0.15, (lat1 - lat0) * 0.15
        caixa = (lon0 - dl, lat0 - dt, lon1 + dl, lat1 + dt)
        try:
            ctx.satelite = externos.imagem_satelite(*caixa, ctx.dados.fonte_satelite, max_px=1800,
                                                    chave_google=ctx.dados.chave_google)
        except Exception:  # noqa: BLE001
            ctx.satelite = None
        if ctx.satelite is None and ctx.dados.fonte_satelite != "Esri World Imagery":
            ctx.avisos.append(f"Satélite: {ctx.dados.fonte_satelite} indisponível"
                              + (" (sem chave google_maps_key nos Secrets)" if "Google" in ctx.dados.fonte_satelite
                                 and not ctx.dados.chave_google else "") + "; tentando Esri")
            try:
                ctx.satelite = externos.imagem_satelite(*caixa, "Esri World Imagery", max_px=1800)
            except Exception:  # noqa: BLE001
                ctx.satelite = None
        if ctx.satelite is None:
            ctx.avisos.append("Satélite: imagem indisponível; mapa de pontos sem fundo")
    if ctx.dados.incluir_ndvi:
        if progresso:
            progresso("Calculando NDVI (Sentinel-2, últimos 12 meses)…")
        from . import ndvi as nd
        try:
            ctx.ndvi = nd.calcular({m.talhao.nome: m.talhao.perimetro for m in ctx.mapas}, P.epsg,
                                   (lon0, lat0, lon1, lat1))
        except Exception:  # noqa: BLE001
            ctx.ndvi = None
        if ctx.ndvi is None:
            ctx.avisos.append("NDVI: imagens Sentinel-2 indisponíveis ou encobertas; página de NDVI omitida")
    if progresso:
        progresso("Consultando altitude (SRTM)…")
    altitude_por_tiles(ctx) or altitude_por_pontos(ctx)
    ctx.tem_alt = all("alt" in m.atributos for m in ctx.mapas)
    if not any("alt" in m.atributos for m in ctx.mapas):
        ctx.avisos.append("Altitude: serviço de elevação indisponível; página de altitude omitida")


def altitude_por_tiles(ctx: Contexto) -> bool:
    """Altitude em cada pixel da grade, lida direto dos tiles de terreno (rápido)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(ctx.projeto.epsg, 4326, always_xy=True)
    pts = []
    for m in ctx.mapas:
        g = m.grade
        pts.append(tr.transform(g.xx[g.mascara], g.yy[g.mascara]))
    lons = np.concatenate([p[0] for p in pts])
    lats = np.concatenate([p[1] for p in pts])
    try:
        alt = externos.elevacao_tiles(lons, lats)
    except Exception:  # noqa: BLE001
        alt = None
    if alt is None or np.isfinite(alt).mean() < 0.8:
        return False
    from .prescricao import _suavizar_nan
    k = 0
    for m in ctx.mapas:
        g = m.grade
        n = int(g.mascara.sum())
        A = np.full(g.mascara.shape, np.nan)
        A[g.mascara] = alt[k:k + n]
        k += n
        ok = g.mascara & np.isfinite(A)
        m.atributos["alt"] = _suavizar_nan(A, ok, sigma=max(15 / g.res, 0.8))   # tira o serrilhado do SRTM
    ctx.fonte_altitude = externos.CREDITO_ALTITUDE
    return True


def altitude_por_pontos(ctx: Contexto, max_pontos: int = 800) -> bool:
    """Reserva: consulta pontos (OpenTopoData/Open-Meteo, 100 por requisição) e interpola por krigagem.

    O espaçamento é ajustado para no máximo ~800 pontos no total (≈ 8 requisições)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(ctx.projeto.epsg, 4326, always_xy=True)
    area_px = sum(int(m.grade.mascara.sum()) for m in ctx.mapas)
    lons, lats, dono = [], [], []
    for i, m in enumerate(ctx.mapas):
        g = m.grade
        passo = max(1, int(round(60 / g.res)), int(np.ceil(np.sqrt(area_px / max_pontos))))
        X, Y = g.xx[::passo, ::passo], g.yy[::passo, ::passo]
        sel = g.mascara[::passo, ::passo]
        if sel.sum() < 12:                       # talhão pequeno: garante alguns pontos
            passo = max(1, int(np.sqrt(g.mascara.sum() / 12)))
            X, Y = g.xx[::passo, ::passo], g.yy[::passo, ::passo]
            sel = g.mascara[::passo, ::passo]
        lo, la = tr.transform(X[sel], Y[sel])
        lons += list(lo)
        lats += list(la)
        dono += [(i, x, y) for x, y in zip(X[sel], Y[sel])]
    alt = externos.elevacao(lons, lats) if lons else None
    if alt is None:
        return False
    alt = np.asarray(alt)
    for i, m in enumerate(ctx.mapas):
        sel = [(k, x, y) for k, (j, x, y) in enumerate(dono) if j == i]
        if not sel:
            continue
        k, xs, ys = zip(*sel)
        vals = alt[list(k)]
        m.atributos["alt"], _ = interpolar_grade(m.grade, np.array(xs), np.array(ys), vals)
    ctx.fonte_altitude = "Altitude: SRTM 30 m (OpenTopoData) / Copernicus (Open-Meteo)"
    return True


def texto_parametros(par, aj) -> list[str]:
    c = par.calcario
    linhas = [
        "Calagem: equilíbrio de bases — elevação de Ca²⁺ e Mg²⁺ a 55% e 19% da CTC",
        f"   calcário de {c.cao:g}% CaO, {c.mgo:g}% MgO, PRNT {c.prnt:g}; dose entre {c.dose_min:.0f} e {c.dose_max:.0f} kg/ha",
        "Gessagem: pelos teores de argila e de SO₄²⁻ (adaptado de Sousa e Lobato, 2004)",
        f"   S almejado: {par.gesso.s_alvo_arenoso:g} / {par.gesso.s_alvo_medio:g} / {par.gesso.s_alvo_argiloso:g} mg/dm³ "
        f"(argila < {par.gesso.limite_arenoso / 10:g}% / {par.gesso.limite_arenoso / 10:g}–{par.gesso.limite_medio / 10:g}% "
        f"/ > {par.gesso.limite_medio / 10:g}%); dose mínima {par.gesso.dose_min:.0f} kg/ha",
        f"Fósforo: P₂O₅ = {br(par.fosforo.coef, 2)} × P^{br(par.fosforo.expoente, 3)}, entre "
        f"{par.fosforo.dose_min:.0f} e {par.fosforo.dose_max:.0f} kg/ha"
        + (f"; aplicado via formulação {aj.formula_p.strip()}" if aj.formula_p.strip() else ""),
        f"Potássio: KCl = {br(par.potassio.coef, 2)} × K^{br(par.potassio.expoente, 3)}, mínimo "
        f"{par.potassio.dose_min:.0f} kg/ha" + (f"; parcelado {aj.kcl_pct_1:.0f}% + {100 - aj.kcl_pct_1:.0f}%"
                                                  if aj.kcl_parcelado else ""),
        "Mapas de prescrição: doses por pixel pelas mesmas regras, agrupadas em até 6 zonas",
    ]
    from .regras import ler_formula
    f_ = ler_formula(aj.formula_p)
    if f_ and f_[1] > 0 and f_[2] > 0:
        linhas.insert(-1, f"   K₂O da formulação ({f_[2]:g}%) descontado do KCl (KCl com {par.potassio.k2o_kcl:g}% de K₂O)")
    return linhas


ROTULO_SUMARIO = {"Calcário": "Calagem", "Gesso": "Gessagem", "S elementar": "Enxofre elementar", "P2O5": "Fósforo",
                  "KCl": "Potássio", "KCl 1ª aplicação": "Potássio – 1ª aplicação",
                  "KCl 2ª aplicação": "Potássio – 2ª aplicação"}


def ordem_prescricoes(blocos, aj, sub_p: str, subs_prod: dict) -> list[tuple]:
    """Páginas de prescrição agrupadas POR PRODUTO: calagem de todos os talhões, depois gessagem (ou S elementar),
    fósforo e potássio. Devolve [(rótulo no sumário ou None, bloco, produto, subtítulo)]."""
    def classe(prod):
        if prod == "Calcário":
            return 0
        if prod in ("Gesso", "S elementar"):
            return 1
        if prod == "P2O5" or prod.startswith("Produto"):
            return 2
        if prod == "P2O5 correção" or prod.startswith("Correção"):
            return 2.5
        if prod == "KCl":
            return 3
        if prod.startswith("KCl 1"):
            return 4
        if prod.startswith("KCl 2"):
            return 5
        return 6
    def mostra(b, prod):
        if prod == "KCl" and aj.kcl_parcelado and any(p.startswith("KCl ") for p in b.zonas):
            return False                       # parcelado: saem as duas parcelas, não a dose total
        if prod == "P2O5" and any(p.startswith("Produto") for p in b.zonas):
            return False                       # com formulação, sai a dose do produto
        return True
    presentes = list(dict.fromkeys(prod for b in blocos for prod in b.zonas if mostra(b, prod)))
    presentes.sort(key=classe)
    saida = []
    for prod in presentes:
        primeiro = True
        for b in blocos:
            if prod not in b.zonas or not mostra(b, prod):
                continue
            if prod.startswith("Correção"):
                sub = f"Formulação {prod.replace('Correção ', '')} · dose do produto comercial (além da manutenção)"
                rot = f"Fósforo – correção ({prod.replace('Correção ', '')})"
            elif prod == "P2O5 correção":
                sub, rot = "Correção de P · kg/ha de P₂O₅ (além da manutenção)", "Fósforo – correção"
            else:
                sub = sub_p if prod.startswith("Produto") else subs_prod.get(prod, prod)
                rot = ROTULO_SUMARIO.get(prod, f"Fósforo – {prod.replace('Produto ', '')}"
                                         if prod.startswith("Produto") else prod)
            saida.append((rot if primeiro else None, b, prod, sub))
            primeiro = False
    return saida


def montar_book(projeto: Projeto, mapas: list[MapaTalhao], dados: DadosBook, par, aj, laudo_dados=None,
                progresso=None, blocos=None) -> tuple[bytes, list[str]]:
    from .prescricao import tabela_volumes
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

    # roteiro: (título no sumário ou None, nível, função)
    roteiro = []
    add = lambda t, n, f: roteiro.append((t, n, f))  # noqa: E731
    grupos = agrupar_por_proximidade(mapas)
    varios = len(grupos) > 1
    rot_g = [lista_talhoes(g) if varios else "" for g in grupos]
    add("Informações técnicas", 0, lambda pdf, k: pagina_info(pdf, ctx, k))
    add("Metodologias e equações", 0, lambda pdf, k: pagina_metodos(pdf, ctx, k, texto_parametros(par, aj)))
    inicio = [1]
    for gi, g in enumerate(grupos):
        def _pts(pdf, k, g=g, gi=gi):
            if gi == 0:
                inicio[0] = 1
            inicio[0] = pagina_pontos(pdf, ctx, k, g, inicio[0], rot_g[gi]) + 1
        add("Mapa de pontos" if gi == 0 else None, 0, _pts)
    atributos_base = (["alt"] if ctx.tem_alt else []) + (["argila"] if ctx.argila_est <= 0.5 else []) \
        + (["areia", "silte"] if ctx.tem_areia_silte else [])
    for ch in atributos_base:
        if all(ch in m.atributos for m in mapas):
            sub_b = "parte das amostras com argila estimada pela CTC" if ch == "argila" and ctx.argila_est > 0 else None
            for gi, g in enumerate(grupos):
                add(NOMES[ch] if gi == 0 else None, 0,
                    lambda pdf, k, ch=ch, g=g, gi=gi, sb=sub_b: pagina_atributo(pdf, ctx, ch, k, sub=sb, mapas=g,
                                                                               rotulo_grupo=rot_g[gi]))
    if ctx.ndvi is not None:
        for gi, g in enumerate(grupos):
            add("Índice de vegetação (NDVI)" if gi == 0 else None, 0,
                lambda pdf, k, g=g, gi=gi: pagina_ndvi(pdf, ctx, k, g, rot_g[gi]))
    add("Mapas de fertilidade", 0, None)
    add("Diagnóstico geral", 1, lambda pdf, k: pagina_diagnostico(pdf, ctx, k))
    add("Panorama da fertilidade", 1, lambda pdf, k: pagina_panorama(pdf, ctx, k))
    add("Área por classe", 1, lambda pdf, k: pagina_area_classes(pdf, ctx, k))
    if all(all(c in m.atributos for c in ("ca", "mg", "k", "hal", "ctc")) for m in mapas):
        add("Equilíbrio de bases", 1, lambda pdf, k: pagina_bases(pdf, ctx, k, 100 * par.calcario.alvo_ca,
                                                                   100 * par.calcario.alvo_mg))
    for ch in ORDEM_FERT:
        if all(ch in m.atributos for m in mapas):
            for gi, g in enumerate(grupos):
                add(NOMES[ch] if gi == 0 else None, 1,
                    lambda pdf, k, ch=ch, g=g, gi=gi: pagina_atributo(pdf, ctx, ch, k, mapas=g, rotulo_grupo=rot_g[gi]))
    if ctx.subs is not None and len(ctx.subs):
        add("Camada 20–40 cm", 1, lambda pdf, k: pagina_subsuperficial(pdf, ctx, k))
    add("Mapas de prescrição", 0, None)
    form = None
    from .regras import ler_formula
    f_ = ler_formula(aj.formula_p)
    if f_ and f_[1] > 0:
        form = f"{int(f_[0]):02d}-{int(f_[1]):02d}-{int(f_[2]):02d}"
    sub_p = (f"Formulação {form} · dose do produto comercial" if form else
             "P₂O₅ (referência — converter para o fertilizante utilizado)")
    subs_prod = {"Calcário": dados.produto_calcario, "Gesso": dados.produto_gesso,
                 "S elementar": "Enxofre elementar", "P2O5": sub_p, "KCl": dados.produto_k,
                 "KCl 1ª aplicação": f"{dados.produto_k} · {aj.kcl_pct_1:.0f}% da dose total",
                 "KCl 2ª aplicação": f"{dados.produto_k} · {100 - aj.kcl_pct_1:.0f}% da dose total"}
    nomes_prod = {"Calcário": "calagem", "Gesso": "gessagem", "S elementar": "enxofre elementar",
                  "P2O5": "fósforo", "KCl": "potássio", "KCl 1ª aplicação": "potássio 1ª apl.",
                  "KCl 2ª aplicação": "potássio 2ª apl."}
    from .prescricao import Bloco
    blocos = blocos or [Bloco(m.talhao.nome, [m], m.zonas) for m in mapas]
    for rot_sum, b, prod, sub in ordem_prescricoes(blocos, aj, sub_p, subs_prod):
        add(rot_sum, 1, lambda pdf, k, b=b, prod=prod, sub=sub: pagina_prescricao(pdf, ctx, b, prod, k, sub))
    vol = tabela_volumes(mapas)
    vol = vol[~vol["Produto"].str.startswith("KCl ")]          # parcelas: só nas páginas de prescrição
    if form:
        vol = vol[vol["Produto"] != "P2O5"]
    n_tal = vol["Talhão"].nunique()
    for parte in range(int(np.ceil(n_tal / 18))):
        add("Volumes de produto" if parte == 0 else None, 0,
            lambda pdf, k, parte=parte: pagina_volumes(pdf, ctx, k, vol, parte))

    itens, num = [], 1
    for t, n, f in roteiro:
        if t:
            itens.append((t, num, n))
        num += 1
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"Book – {dados.propriedade}", "Author": MARCAS[dados.marca]["nome"]}) as pdf:
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
