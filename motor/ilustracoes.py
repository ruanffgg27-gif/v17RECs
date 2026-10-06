"""Ilustrações autorais (vetoriais, desenhadas em código) das capas do book.

- capa geral: paisagem agrícola com talhões em perspectiva, sol, pivô, pontos de amostragem,
  drone e satélite (agricultura de precisão);
- capa dos mapas de fertilidade: perfil do solo, trado e amostra, partícula de argila com cátions
  trocáveis (química) e agregados/poros (física), vidraria de laboratório;
- capa dos mapas de prescrição: talhão pintado em zonas de dose, trator com distribuidor a lanço de
  taxa variável, rastros e sinal de GNSS.

Cada ilustração é renderizada uma vez por combinação de cores (cache) e entra no PDF como imagem.
"""
from __future__ import annotations

import functools
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402
from PIL import Image  # noqa: E402

SOL = ("#FFD27A", "#FDB44B", "#F68B2C")          # centro → borda
CEU = "#FFF4E3"
TERRAS = ["#5B3A24", "#7A4E2F", "#9A6538", "#B87D45", "#CF9A5C"]
CULTURAS = ["#6FA83E", "#8DBB4C", "#4F8A36", "#A9C95E", "#D8C07A", "#A87549", "#7CAF4A", "#C7B36A"]
COR_PRESC = ["#008000", "#00FF00", "#FFFF00", "#FF8000", "#FF0000", "#800000"]


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(np.clip(a + (b - a) * t, 0, 1))


def _luminancia(c) -> float:
    r, g, b = to_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ------------------------------------------------------------------ utilitários
def _fig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_autoscale_on(False)
    return fig, ax


def _render(fig, dpi) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGBA"))


def gradiente(ax, x0, x1, y0, y1, c_baixo, c_cima, a_baixo=1.0, a_cima=1.0, z=0, n=256):
    t = np.linspace(0, 1, n)[:, None]
    cb, cc = np.array(to_rgb(c_baixo)), np.array(to_rgb(c_cima))
    rgb = cb + (cc - cb) * t[..., None]
    a = (a_baixo + (a_cima - a_baixo) * t)[..., None]
    img = np.concatenate([rgb * np.ones((n, 2, 3)), a * np.ones((n, 2, 1))], axis=2)
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="lower", aspect="auto", zorder=z, interpolation="bicubic")


def brilho(ax, cx, cy, r, cor, a_max=0.5, z=1, n=18):
    """Halo radial suave (círculos concêntricos translúcidos)."""
    for i in range(n):
        f = 1 - i / n
        ax.add_patch(Circle((cx, cy), r * f, color=cor, alpha=a_max / n * (1.2 - f) * 2.2, lw=0, zorder=z))


def ondulado(x, base, amp, sem, harm=((1, 1.0), (2.3, 0.45), (5.1, 0.18))):
    r = np.random.default_rng(sem)
    y = np.full_like(x, base, dtype=float)
    for f, a in harm:
        y += amp * a * np.sin(2 * np.pi * f * x / x.max() + r.uniform(0, 6.28))
    return y


# --------------------------------------------------------------- perspectiva
class Perspectiva:
    """Plano do chão (x lateral, z profundidade ≥ 1) projetado na tela: X = cx + f·x/z, Y = yh − H/z."""

    def __init__(self, cx, yh, H, f):
        self.cx, self.yh, self.H, self.f = cx, yh, H, f

    def __call__(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        return self.cx + self.f * x / z, self.yh - self.H / z

    def inversa(self, X, Y):
        z = self.H / np.maximum(self.yh - Y, 1e-6)
        return (X - self.cx) * z / self.f, z

    def poligono(self, pts, n=12):
        """Projeta um polígono do chão, subdividindo as arestas (bordas levemente curvas)."""
        out = []
        for (x0, z0), (x1, z1) in zip(pts, pts[1:] + pts[:1]):
            t = np.linspace(0, 1, n, endpoint=False)
            X, Y = self(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t)
            out += list(zip(X, Y))
        return out


def _campo_ruido(sem, xr=(-16, 16), zr=(1, 45), n=320, sigma=7):
    """Campo aleatório suave no chão (ruído gaussiano filtrado): manchas com transições graduais."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    r = np.random.default_rng(sem)
    g = gaussian_filter(r.normal(size=(n, n)), sigma)
    g = (g - g.mean()) / g.std()

    def f(x, z):
        ix = (np.asarray(x) - xr[0]) / (xr[1] - xr[0]) * (n - 1)
        iz = (np.asarray(z) - zr[0]) / (zr[1] - zr[0]) * (n - 1)
        return map_coordinates(g, [np.ravel(iz), np.ravel(ix)], order=1, mode="nearest").reshape(np.shape(x))
    return f


def _campo_suave(sem, n=6, escala=1.0):
    r = np.random.default_rng(sem)
    cx, cz = r.uniform(-8, 8, n), r.uniform(1, 12, n)
    ph = r.uniform(0, 6.28, n)
    fx = r.uniform(0.25, 0.6, n) / escala

    def f(x, z):
        v = 0
        for a, b, p, q in zip(cx, cz, ph, fx):
            v = v + np.sin((x - a) * q + p) * np.cos((z - b) * q * 1.3 + p / 2)
        return v / n
    return f


# ----------------------------------------------------------------- elementos
def sol(ax, cx, cy, r, z=2):
    brilho(ax, cx, cy, r * 3.2, SOL[0], a_max=0.55, z=z)
    ang = np.linspace(0, 2 * np.pi, 28, endpoint=False) + 0.06
    for i, a in enumerate(ang):                                   # raios finos, alternados
        comp = r * (2.05 if i % 2 else 1.6)
        ax.add_patch(Wedge((cx, cy), comp, np.degrees(a) - 1.4, np.degrees(a) + 1.4, width=comp - r * 1.18,
                           color=SOL[1], alpha=0.22, lw=0, zorder=z))
    for k, (c, f) in enumerate(zip(SOL[::-1], (1.0, 0.86, 0.64))):
        ax.add_patch(Circle((cx, cy), r * f, color=c, lw=0, zorder=z + 1 + k * 0.01))
    ax.add_patch(Circle((cx - r * 0.25, cy + r * 0.25), r * 0.28, color="white", alpha=0.25, lw=0, zorder=z + 1.1))


def arvores(ax, xs, ys, escala, cor, z, sem=0):
    r = np.random.default_rng(sem)
    for x, y in zip(xs, ys):
        s = escala * r.uniform(0.75, 1.25)
        ax.add_patch(Rectangle((x - s * 0.06, y), s * 0.12, s * 0.45, color="#5A3B22", lw=0, zorder=z))
        for dx, dy, rr, t in ((0, 0.62, 0.36, 0.0), (-0.2, 0.5, 0.26, 0.12), (0.2, 0.52, 0.27, -0.08)):
            ax.add_patch(Circle((x + dx * s, y + dy * s), rr * s, color=mix(cor, "white", 0.1 + t), lw=0, zorder=z + 0.1))


def satelite(ax, x, y, s, acento, z=20, ang=-20):
    t = matplotlib.transforms.Affine2D().rotate_deg_around(x, y, ang) + ax.transData
    corpo = FancyBboxPatch((x - 0.18 * s, y - 0.13 * s), 0.36 * s, 0.26 * s, boxstyle="round,pad=0,rounding_size=0.04",
                           fc="#E9EEF2", ec="#58636B", lw=0.8, zorder=z, transform=t)
    ax.add_patch(corpo)
    for lado in (-1, 1):
        x0 = x + lado * 0.24 * s - (0.62 * s if lado < 0 else 0)
        ax.add_patch(Rectangle((x0, y - 0.1 * s), 0.62 * s, 0.2 * s, fc="#27456B", ec="#8FA9C7", lw=0.6, zorder=z,
                               transform=t))
        for j in range(1, 4):
            xx = x0 + j * 0.155 * s
            ax.plot([xx, xx], [y - 0.1 * s, y + 0.1 * s], color="#8FA9C7", lw=0.4, zorder=z + 0.1, transform=t)
        ax.plot([x + lado * 0.18 * s, x + lado * 0.24 * s], [y, y], color="#58636B", lw=0.8, zorder=z, transform=t)
    ax.add_patch(Circle((x, y - 0.2 * s), 0.07 * s, fc=acento, ec="none", zorder=z + 0.2, transform=t))


def ondas(ax, x, y, r0, n, ang0, ang1, cor, z=19, lw=1.0, alpha=0.6):
    for i in range(n):
        ax.add_patch(matplotlib.patches.Arc((x, y), 2 * (r0 + i * r0 * 0.9), 2 * (r0 + i * r0 * 0.9), theta1=ang0,
                                            theta2=ang1, color=cor, lw=lw, alpha=alpha * (1 - i / (n + 1)), zorder=z,
                                            ls=(0, (3, 2))))


def drone(ax, x, y, s, acento, z=22):
    escuro = "#2F3A40"
    ax.add_patch(FancyBboxPatch((x - 0.22 * s, y - 0.06 * s), 0.44 * s, 0.12 * s,
                                boxstyle="round,pad=0,rounding_size=0.05", fc=escuro, ec="none", zorder=z))
    for lado in (-1, 1):
        ax.plot([x + lado * 0.2 * s, x + lado * 0.52 * s], [y + 0.02 * s, y + 0.1 * s], color=escuro, lw=1.4, zorder=z)
        ax.add_patch(Ellipse((x + lado * 0.52 * s, y + 0.13 * s), 0.42 * s, 0.045 * s, fc="#9AA7AE", alpha=0.8,
                             ec="none", zorder=z + 0.1))
        ax.plot([x + lado * 0.52 * s] * 2, [y + 0.08 * s, y + 0.13 * s], color=escuro, lw=1.1, zorder=z)
    ax.add_patch(Circle((x, y - 0.075 * s), 0.045 * s, fc=acento, ec="none", zorder=z + 0.2))


def sede(ax, x, y, s, cor, acento, z=6):
    """Silos e galpão da fazenda no horizonte."""
    metal, sombra = "#D7DDE1", "#AEB8BF"
    for k, (dx, h, w) in enumerate(((0.0, 0.95, 0.3), (0.34, 1.15, 0.34), (0.72, 0.85, 0.28))):
        x0 = x + dx * s
        ax.add_patch(Rectangle((x0, y), w * s, h * s, fc=metal, ec=sombra, lw=0.5, zorder=z))
        for yy in np.linspace(y + 0.12 * s, y + h * s - 0.05 * s, 5):
            ax.plot([x0, x0 + w * s], [yy, yy], color=sombra, lw=0.4, zorder=z + 0.01)
        ax.add_patch(Polygon([(x0 - 0.02 * s, y + h * s), (x0 + w * s + 0.02 * s, y + h * s),
                              (x0 + w * s / 2, y + (h + 0.18) * s)], closed=True, fc=sombra, ec="none", zorder=z))
    xg = x + 1.1 * s
    ax.add_patch(Rectangle((xg, y), 0.95 * s, 0.42 * s, fc=mix(cor, "white", 0.25), ec="none", zorder=z))
    ax.add_patch(Polygon([(xg - 0.05 * s, y + 0.42 * s), (xg + 1.0 * s, y + 0.42 * s), (xg + 0.475 * s, y + 0.66 * s)],
                         closed=True, fc=mix(acento, "black", 0.1), ec="none", zorder=z))
    ax.add_patch(Rectangle((xg + 0.36 * s, y), 0.23 * s, 0.3 * s, fc=mix(cor, "black", 0.3), zorder=z + 0.01))


def pneu(ax, cx, cy, r, z, aro="#9AA7AE", garras=18):
    """Pneu agrícola com garras (vista lateral)."""
    ax.add_patch(Circle((cx, cy), r, fc="#232628", ec="none", zorder=z))
    for a in np.linspace(0, 2 * np.pi, garras, endpoint=False):
        t = matplotlib.transforms.Affine2D().rotate_around(cx, cy, a) + ax.transData
        ax.add_patch(Rectangle((cx + r * 0.9, cy - r * 0.07), r * 0.14, r * 0.14, fc="#232628", ec="none", zorder=z,
                               transform=t))
    ax.add_patch(Circle((cx, cy), r * 0.78, fc="#34393C", ec="none", zorder=z + 0.01))
    ax.add_patch(Circle((cx, cy), r * 0.52, fc=aro, ec=mix(aro, "black", 0.25), lw=0.8, zorder=z + 0.02))
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.add_patch(Circle((cx + r * 0.36 * np.cos(a), cy + r * 0.36 * np.sin(a)), r * 0.04,
                            fc=mix(aro, "black", 0.3), ec="none", zorder=z + 0.03))
    ax.add_patch(Circle((cx, cy), r * 0.16, fc=mix(aro, "black", 0.35), ec="none", zorder=z + 0.03))


def quadriciclo(ax, x, y, s, cor, acento, z=10):
    """Quadriciclo (vista lateral, voltado à direita) com amostrador hidráulico de solo na traseira."""
    escuro, metal = "#2F3A40", "#8C979E"
    corpo = acento
    # bloco do motor e chassi
    ax.add_patch(Rectangle((x - 0.35 * s, y + 0.3 * s), 0.7 * s, 0.28 * s, fc=escuro, zorder=z))
    pneu(ax, x - 0.52 * s, y + 0.3 * s, 0.3 * s, z + 0.1, garras=14)
    pneu(ax, x + 0.52 * s, y + 0.3 * s, 0.3 * s, z + 0.1, garras=14)
    # carenagem com para-lamas
    for cxr in (x - 0.52 * s, x + 0.52 * s):
        ax.add_patch(Wedge((cxr, y + 0.3 * s), 0.44 * s, 5, 175, width=0.11 * s, fc=corpo, ec="none", zorder=z + 0.2))
    ax.add_patch(Polygon([(x - 0.4 * s, y + 0.58 * s), (x + 0.42 * s, y + 0.58 * s), (x + 0.36 * s, y + 0.84 * s),
                          (x + 0.05 * s, y + 0.9 * s), (x - 0.2 * s, y + 0.74 * s)], closed=True, fc=corpo, ec="none",
                         zorder=z + 0.2))
    ax.add_patch(Polygon([(x + 0.05 * s, y + 0.9 * s), (x + 0.36 * s, y + 0.84 * s), (x + 0.33 * s, y + 0.8 * s),
                          (x + 0.06 * s, y + 0.85 * s)], closed=True, fc=mix(corpo, "white", 0.35), zorder=z + 0.21))
    ax.add_patch(FancyBboxPatch((x - 0.62 * s, y + 0.72 * s), 0.62 * s, 0.11 * s,
                                boxstyle="round,pad=0,rounding_size=0.05", fc=escuro, ec="none", zorder=z + 0.25))  # banco
    ax.plot([x + 0.3 * s, x + 0.4 * s], [y + 0.86 * s, y + 1.08 * s], color=escuro, lw=2.4 * s,
            solid_capstyle="round", zorder=z + 0.26)
    ax.plot([x + 0.32 * s, x + 0.48 * s], [y + 1.08 * s, y + 1.1 * s], color=escuro, lw=3 * s,
            solid_capstyle="round", zorder=z + 0.26)                                                   # guidão
    ax.add_patch(Circle((x + 0.9 * s, y + 0.64 * s), 0.05 * s, fc="#FFF1C2", ec="none", zorder=z + 0.3))  # farol
    # bagageiros
    ax.plot([x + 0.55 * s, x + 0.95 * s], [y + 0.8 * s, y + 0.8 * s], color=escuro, lw=2 * s, zorder=z + 0.15)
    ax.plot([x - 1.0 * s, x - 0.55 * s], [y + 0.82 * s, y + 0.82 * s], color=escuro, lw=2 * s, zorder=z + 0.15)
    for k in range(3):                                                                            # sacos de amostra
        ax.add_patch(FancyBboxPatch((x - 0.97 * s + k * 0.14 * s, y + 0.85 * s), 0.11 * s, 0.15 * s,
                                    boxstyle="round,pad=0,rounding_size=0.02", fc="white", ec="#B8C0C5", lw=0.5,
                                    zorder=z + 0.16))
    # amostrador hidráulico: torre, cilindro e haste cravada no solo
    xa = x - 1.15 * s
    ax.add_patch(Rectangle((xa - 0.045 * s, y + 0.02 * s), 0.09 * s, 1.15 * s, fc=metal, ec="none", zorder=z + 0.05))
    ax.plot([xa, x - 0.9 * s], [y + 0.55 * s, y + 0.55 * s], color=metal, lw=2.2 * s, zorder=z + 0.05)
    ax.add_patch(FancyBboxPatch((xa - 0.09 * s, y + 0.66 * s), 0.18 * s, 0.38 * s,
                                boxstyle="round,pad=0,rounding_size=0.03", fc=cor, ec="none", zorder=z + 0.06))
    ax.plot([xa, xa], [y + 0.1 * s, y - 0.1 * s], color="#5E676D", lw=2.6 * s, zorder=z + 0.04)
    ax.add_patch(Ellipse((xa, y - 0.02 * s), 0.32 * s, 0.07 * s, fc="#7A4E2F", alpha=0.8, lw=0, zorder=z + 0.03))


# --------------------------------------------------------------- capa geral
def _capa_geral(cor, acento, W=8.27, H=5.0):
    fig, ax = _fig(W, H)
    yh = 2.65                                                    # linha do horizonte
    gradiente(ax, 0, W, yh - 0.1, H, mix(SOL[0], CEU, 0.55), "white", 1, 0, z=0)
    sol(ax, 6.55, yh + 0.55, 0.42, z=1)
    # satélite + sinal
    satelite(ax, 7.3, 4.55, 0.55, acento, ang=-12)
    ondas(ax, 7.3, 4.55, 0.26, 4, 205, 260, mix(cor, "white", 0.2), lw=0.9)
    # morros ao fundo
    x = np.linspace(0, W, 500)
    for k, (b, a, t) in enumerate(((yh + 0.33, 0.22, 0.72), (yh + 0.18, 0.18, 0.55), (yh + 0.06, 0.12, 0.38))):
        y = ondulado(x, b, a, 10 + k)
        ax.fill_between(x, yh - 0.05, y, color=mix(cor, CEU, t), lw=0, zorder=3 + k)
    sede(ax, 1.05, yh - 0.03, 0.55, cor, acento, z=6.4)
    arvores(ax, np.linspace(0.3, 7.9, 34) + np.random.default_rng(2).uniform(-0.1, 0.1, 34),
            np.full(34, yh - 0.02), 0.22, mix(cor, "white", 0.15), z=6.5, sem=3)
    # talhões em perspectiva (grade de talhões girada no terreno, recortada à frente da câmera)
    from shapely import box as sbox, segmentize
    from shapely.affinity import rotate
    from shapely.geometry import LineString, Polygon as SPoly
    P = Perspectiva(cx=W * 0.5, yh=yh, H=2.0, f=2.9)
    visivel = sbox(-60, 1.0, 60, 60)
    r = np.random.default_rng(5)

    def desenha(geom, **kw):
        for g in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            if g.is_empty or not isinstance(g, SPoly):
                continue
            xs, zs = np.array(segmentize(g, 0.25).exterior.coords).T
            X, Y = P(xs, zs)
            pol = Polygon(np.column_stack([X, Y]), closed=True, **kw)
            ax.add_patch(pol)
            yield g, pol

    us = np.cumsum(r.uniform(2.2, 3.6, 16)) - 24
    vs = np.cumsum(r.uniform(1.6, 3.0, 16)) - 4
    campos = []
    for a in range(len(us) - 1):
        for b in range(len(vs) - 1):
            q = SPoly([(us[a], vs[b]), (us[a + 1], vs[b]), (us[a + 1], vs[b + 1]), (us[a], vs[b + 1])])
            q = rotate(q, 28, origin=(0, 0)).intersection(visivel)
            if not q.is_empty and q.area > 0.01:
                campos.append((q, a, b))
    paleta = CULTURAS + ["#6FA83E", "#8DBB4C"]
    for q, a, b in campos:
        c = mix(paleta[(a * 3 + b * 5) % len(paleta)], cor, 0.1)
        preparo = paleta[(a * 3 + b * 5) % len(paleta)] == "#A87549"
        dist = q.centroid.y
        c = mix(c, mix(cor, "#B9C8B0", 0.7), float(np.clip((dist - 6) / 30, 0, 0.35)))   # névoa ao longe
        for g, pol in desenha(q, fc=c, ec=mix(c, "#F3EBD8", 0.6), lw=float(np.clip(1.0 - dist / 18, 0.15, 1.0)),
                              zorder=8):
            # linhas de plantio (paralelas a um lado do talhão → convergem a um ponto de fuga)
            passo = 0.16 if not preparo else 0.3
            if dist > 25:
                continue
            base_rot = rotate(g, -28, origin=(0, 0))
            u0, v0, u1, v1 = base_rot.bounds
            for uu in np.arange(u0 + passo / 2, u1, passo):
                ln = rotate(LineString([(uu, v0 - 1), (uu, v1 + 1)]), 28, origin=(0, 0)).intersection(g)
                for seg in (ln.geoms if hasattr(ln, "geoms") else [ln]):
                    if seg.is_empty or seg.length < 0.05:
                        continue
                    xs, zs = np.array(segmentize(seg, 0.5).coords).T
                    X, Y = P(xs, zs)
                    alpha = float(np.clip(0.75 - dist / 30, 0.15, 0.75))
                    ax.plot(X, Y, color=mix(c, "black", 0.22 if not preparo else 0.3), lw=0.6 if dist < 4 else 0.4,
                            alpha=alpha, zorder=8.1, solid_capstyle="butt")
    # pivô central
    t = np.linspace(0, 2 * np.pi, 160)
    cxp, czp, rp = 3.1, 4.4, 1.25
    X, Y = P(cxp + rp * np.cos(t), czp + rp * np.sin(t))
    cp = mix("#4F9A3A", cor, 0.1)
    ax.fill(X, Y, color=cp, zorder=8.3, lw=0)
    ax.plot(X, Y, color=mix(cp, "white", 0.6), lw=1, zorder=8.31)
    for f in np.linspace(0.12, 0.95, 9):
        X2, Y2 = P(cxp + rp * f * np.cos(t), czp + rp * f * np.sin(t))
        ax.plot(X2, Y2, color=mix(cp, "black", 0.2), lw=0.45, alpha=0.6, zorder=8.35)
    ang_b = np.radians(200)
    xa, za = cxp + rp * np.cos(ang_b), czp + rp * np.sin(ang_b)
    Xc, Yc = P(cxp, czp)
    Xa, Ya = P(xa, za)
    for k in range(6):                                             # lance do pivô com torres
        f0 = k / 6
        Xk, Yk = P(cxp + (xa - cxp) * f0, czp + (za - czp) * f0)
        ax.plot([Xk, Xk], [Yk, Yk + 0.07], color="#8C979E", lw=0.8, zorder=8.45)
    ax.plot([Xc, Xa], [Yc + 0.07, Ya + 0.07], color="#E9EEF2", lw=1.4, zorder=8.5)
    ax.plot([Xc, Xc], [Yc, Yc + 0.12], color="#8C979E", lw=1.4, zorder=8.5)
    # pontos de amostragem georreferenciados (malha regular num talhão da frente)
    for ug in np.linspace(-1.1, 0.5, 4):
        for vg in (1.6, 2.2):
            xg, zg = ug * np.cos(np.radians(28)) - vg * np.sin(np.radians(28)) + 0.6, \
                ug * np.sin(np.radians(28)) + vg * np.cos(np.radians(28))
            if zg < 1.05:
                continue
            X, Y = P(xg, zg)
            if 1.0 < X < 3.0 and Y < 1.9:                       # espaço do quadriciclo
                continue
            ax.plot([X, X], [Y, Y + 0.16], color="#333333", lw=0.9, zorder=9)
            ax.add_patch(Circle((X, Y + 0.18), 0.05, fc=acento, ec="white", lw=0.9, zorder=9.1))
            ax.add_patch(Ellipse((X, Y), 0.08, 0.025, fc="black", alpha=0.25, lw=0, zorder=8.9))
    # quadriciclo com amostrador coletando solo num talhão da frente
    xq, yq = 2.05, 0.84
    ax.add_patch(Ellipse((xq - 0.1, yq + 0.01), 1.6, 0.1, fc="black", alpha=0.18, lw=0, zorder=9.4))
    quadriciclo(ax, xq, yq, 0.72, cor, acento, z=9.5)
    # perfil de solo na base
    xb = np.linspace(0, W, 400)
    topo = 0.62
    for k, (c, yb) in enumerate(zip(TERRAS, (0.62, 0.49, 0.36, 0.24, 0.11))):
        ax.fill_between(xb, 0, yb + 0.012 * np.sin(xb * (3 + k) + k), color=c, lw=0, zorder=10 + k * 0.01)
    rr = np.random.default_rng(9)
    ax.scatter(rr.uniform(0, W, 380), rr.uniform(0.02, 0.58, 380), s=rr.uniform(0.3, 4, 380), color="#2E1D12",
               alpha=0.3, lw=0, zorder=10.2)
    for xr in rr.uniform(0.2, W - 0.2, 26):                     # raízes
        yy = np.linspace(0.62, 0.62 - rr.uniform(0.15, 0.45), 20)
        ax.plot(xr + 0.03 * np.sin(np.linspace(0, 6, 20) + xr), yy, color="#E8D3B0", lw=0.5, alpha=0.6, zorder=10.3)
    ax.fill_between(xb, 0.6, 0.66, color=mix(cor, "black", 0.1), lw=0, zorder=10.4)
    return fig


# ---------------------------------------------------------- capa fertilidade
def soja(ax, x, ysup, h, cor, z=6, sem=0):
    """Planta de soja estilizada (folhas trifolioladas) com raiz pivotante, laterais e nódulos."""
    r = np.random.default_rng(sem)
    verde, verde2 = mix(cor, "#6FA83E", 0.55), mix(cor, "#9CC75B", 0.5)
    t = np.linspace(0, 1, 30)
    xs = x + 0.05 * np.sin(t * 3 + sem)
    ax.plot(xs, ysup + t * h, color=verde, lw=2.2, solid_capstyle="round", zorder=z)
    for k, (f, lado) in enumerate(((0.45, -1), (0.72, 1), (1.0, 0))):
        xn, yn = x + 0.05 * np.sin(f * 3 + sem), ysup + f * h
        comp = 0.32 * h if lado else 0.12 * h
        xp, yp = xn + lado * comp, yn + (0.1 * h if lado else 0.05 * h)
        ax.plot([xn, xp], [yn, yp], color=verde, lw=1.4, zorder=z)
        for a in (-50, 0, 50):                                       # trifólio
            ang = np.radians(90 + a - (30 * lado))
            L = 0.2 * h
            ax.add_patch(Ellipse((xp + 0.5 * L * np.cos(ang), yp + 0.5 * L * np.sin(ang)), L, L * 0.52,
                                 angle=np.degrees(ang), fc=verde2 if a else verde, ec="none", zorder=z + 0.1))
    # raízes
    prof = h * 1.9
    yr = np.linspace(ysup, ysup - prof, 40)
    xr = x + 0.04 * np.sin(np.linspace(0, 4, 40) + sem)
    cr = "#F1E3C8"
    ax.plot(xr, yr, color=cr, lw=1.8, alpha=0.95, zorder=z)
    for k in range(9):
        y0 = ysup - prof * (0.08 + 0.09 * k)
        for lado in (-1, 1):
            comp = prof * r.uniform(0.18, 0.34) * (1 - k / 12)
            xx = np.linspace(0, lado * comp, 14)
            yy = y0 - np.abs(xx) * r.uniform(0.35, 0.8) - 0.03 * np.sin(np.abs(xx) * 9)
            ax.plot(x + xx, yy, color=cr, lw=0.9, alpha=0.85, zorder=z)
            if k < 5 and r.random() < 0.7:                            # nódulos
                q = r.integers(4, 12)
                ax.add_patch(Circle((x + xx[q], yy[q]), 0.035, fc="#E8B7A0", ec="none", zorder=z + 0.05))


def _capa_fertilidade(cor, acento, W=8.27, H=11.69):
    """Capa dos mapas de fertilidade — minimalista: nutrientes, planta e perfil do solo."""
    fig, ax = _fig(W, H)
    ysup = 5.4
    gradiente(ax, 0, W, ysup, H, mix(cor, "white", 0.9), "white", z=0)
    # nutrientes analisados (faixa discreta abaixo do título)
    elems = ["Ca", "Mg", "K", "P", "S"]
    lado, gap = 0.78, 0.22
    x0 = (W - (len(elems) * lado + (len(elems) - 1) * gap)) / 2
    for k, sim in enumerate(elems):
        xe = x0 + k * (lado + gap)
        ax.add_patch(FancyBboxPatch((xe, 7.25), lado, lado, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="white", ec=mix(cor, "white", 0.55), lw=1.3, zorder=1))
        ax.text(xe + lado / 2, 7.25 + lado / 2, sim, fontsize=19, ha="center", va="center",
                color=(acento if _luminancia(acento) < 0.55 else mix(acento, "black", 0.35)) if sim == "P" else cor,
                fontweight="bold", zorder=1.1)
    # perfil do solo: horizontes lisos
    x = np.linspace(0, W, 400)
    cores_h = ["#5E3C24", "#7A4E2F", "#96643D", "#B27E4E"]
    topos = [ysup + 0.03 * np.sin(x * 1.7), 4.05 + 0.12 * np.sin(x * 0.9 + 1), 2.7 + 0.14 * np.sin(x * 0.7 + 2.5),
             1.35 + 0.12 * np.sin(x * 0.8 + 4)]
    for c, tp in zip(cores_h, topos):
        ax.fill_between(x, 0, tp, color=c, lw=0, zorder=2)
    ax.fill_between(x, ysup - 0.02, topos[0] + 0.05, color=mix(cor, "#5E3C24", 0.35), lw=0, zorder=2.1)
    r = np.random.default_rng(3)                                   # poucos agregados, discretos
    for _ in range(70):
        xx, yy = r.uniform(0.1, W - 0.1), r.uniform(0.2, ysup - 0.25)
        k = int(np.searchsorted(-np.array([t.mean() for t in topos]), -yy))
        base = cores_h[min(max(k - 1, 0), 3)]
        ax.add_patch(Ellipse((xx, yy), r.uniform(0.07, 0.18), r.uniform(0.05, 0.11), angle=r.uniform(0, 180),
                             fc=mix(base, "black", 0.18), ec="none", alpha=0.8, zorder=2.2))
    # plantas de soja com raízes
    for k, (xp, h) in enumerate(((2.6, 1.15), (4.15, 1.35), (5.7, 1.1))):
        soja(ax, xp, ysup, h, cor, z=4, sem=k)
    # trado de amostragem cravado (0–20 cm)
    xt = 1.05
    ax.add_patch(Rectangle((xt - 0.045, ysup - 1.25), 0.09, 2.3, fc="#9AA4AA", ec="none", zorder=5))
    ax.add_patch(FancyBboxPatch((xt - 0.42, ysup + 1.0), 0.84, 0.11, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#6B757B", ec="none", zorder=5))
    for yy in np.arange(ysup - 1.2, ysup - 0.55, 0.14):
        ax.plot([xt - 0.12, xt + 0.12], [yy, yy + 0.07], color="#6B757B", lw=1.3, zorder=5.1)
    return fig


def _trator(ax, x, y, s, cor, acento, z=20):
    """Trator moderno (vista lateral, voltado à esquerda) com distribuidor a lanço de discos duplos.

    (x, y) = ponto do chão sob a roda traseira do trator. Devolve o centro dos discos do distribuidor."""
    escuro, vidro, cinza = "#2F3A40", "#BFE0EE", "#9AA7AE"
    claro = mix(cor, "white", 0.22)
    R, r = 0.62 * s, 0.4 * s
    xr, xf = x, x - 1.5 * s
    # ---- distribuidor (atrás, à direita)
    xd = x + 2.15 * s
    ax.plot([x + 0.55 * s, xd - 0.45 * s], [y + 0.52 * s, y + 0.42 * s], color=escuro, lw=3 * s, zorder=z - 0.2)
    pneu(ax, xd + 0.1 * s, y + 0.3 * s, 0.3 * s, z - 0.1, garras=12)
    ax.add_patch(Rectangle((xd - 0.5 * s, y + 0.5 * s), 1.2 * s, 0.1 * s, fc=escuro, zorder=z - 0.05))
    tremonha = [(xd - 0.62 * s, y + 1.55 * s), (xd + 0.82 * s, y + 1.55 * s), (xd + 0.55 * s, y + 0.72 * s),
                (xd - 0.35 * s, y + 0.72 * s)]
    ax.add_patch(Polygon(tremonha, closed=True, fc="#D9DEE1", ec="none", zorder=z))
    ax.add_patch(Polygon([tremonha[0], tremonha[1], (xd + 0.76 * s, y + 1.38 * s), (xd - 0.57 * s, y + 1.38 * s)],
                         closed=True, fc=acento, ec="none", zorder=z + 0.01))                         # faixa
    ax.add_patch(Rectangle((xd - 0.66 * s, y + 1.53 * s), 1.52 * s, 0.07 * s, fc=escuro, zorder=z + 0.02))
    ax.add_patch(Polygon([(xd - 0.35 * s, y + 0.72 * s), (xd + 0.55 * s, y + 0.72 * s), (xd + 0.25 * s, y + 0.6 * s),
                          (xd - 0.05 * s, y + 0.6 * s)], closed=True, fc=mix("#D9DEE1", "black", 0.2), zorder=z))
    for dxs in (-0.1, 0.4):                                                                    # discos
        cxd = xd + dxs * s
        ax.add_patch(Ellipse((cxd, y + 0.55 * s), 0.46 * s, 0.09 * s, fc=escuro, ec=cinza, lw=0.7, zorder=z + 0.05))
        for a in np.linspace(0, np.pi, 4, endpoint=False):
            ax.plot([cxd - 0.2 * s * np.cos(a), cxd + 0.2 * s * np.cos(a)],
                    [y + 0.55 * s - 0.035 * s * np.sin(a), y + 0.55 * s + 0.035 * s * np.sin(a)],
                    color=cinza, lw=0.7, zorder=z + 0.06)
    # ---- trator
    pneu(ax, xr, y + R, R, z + 0.3, garras=22)
    pneu(ax, xf, y + r, r, z + 0.3, garras=16)
    # capô com nariz arredondado
    capo = [(xf - 0.55 * s, y + 0.48 * s), (x - 0.35 * s, y + 0.48 * s), (x - 0.35 * s, y + 1.12 * s),
            (xf - 0.25 * s, y + 1.02 * s), (xf - 0.5 * s, y + 0.95 * s), (xf - 0.6 * s, y + 0.8 * s)]
    ax.add_patch(Polygon(capo, closed=True, fc=cor, ec="none", zorder=z + 0.2))
    ax.add_patch(Polygon([(xf - 0.25 * s, y + 1.02 * s), (x - 0.35 * s, y + 1.12 * s), (x - 0.35 * s, y + 1.04 * s),
                          (xf - 0.3 * s, y + 0.95 * s)], closed=True, fc=claro, ec="none", zorder=z + 0.21))  # brilho
    ax.add_patch(Polygon([(xf - 0.58 * s, y + 0.55 * s), (xf - 0.42 * s, y + 0.55 * s), (xf - 0.42 * s, y + 0.92 * s),
                          (xf - 0.55 * s, y + 0.85 * s)], closed=True, fc=escuro, zorder=z + 0.22))           # grade
    for k in range(5):
        yy = y + (0.6 + k * 0.06) * s
        ax.plot([xf - 0.56 * s, xf - 0.44 * s], [yy, yy], color=mix(escuro, "white", 0.3), lw=0.6, zorder=z + 0.23)
    ax.add_patch(Rectangle((xf - 0.46 * s, y + 0.92 * s), 0.28 * s, 0.045 * s, fc="#FFF1C2", zorder=z + 0.23))  # farol
    ax.add_patch(Rectangle((xf - 0.7 * s, y + 0.34 * s), 0.3 * s, 0.22 * s, fc=escuro, zorder=z + 0.19))      # lastro
    ax.add_patch(Rectangle((xf + 0.05 * s, y + 0.8 * s), 0.9 * s, 0.05 * s, fc=acento, zorder=z + 0.23))       # friso
    ax.add_patch(Rectangle((x - 0.62 * s, y + 1.12 * s), 0.06 * s, 0.62 * s, fc=escuro, zorder=z + 0.15))     # escape
    ax.add_patch(Rectangle((x - 0.64 * s, y + 1.72 * s), 0.1 * s, 0.05 * s, fc=escuro, zorder=z + 0.15))
    # cabine
    cab = [(x - 0.5 * s, y + 1.05 * s), (x + 0.45 * s, y + 1.05 * s), (x + 0.4 * s, y + 2.1 * s),
           (x - 0.38 * s, y + 2.1 * s)]
    ax.add_patch(Polygon(cab, closed=True, fc=escuro, ec="none", zorder=z + 0.25))
    ax.add_patch(Polygon([(x - 0.42 * s, y + 1.2 * s), (x - 0.02 * s, y + 1.2 * s), (x - 0.02 * s, y + 1.98 * s),
                          (x - 0.32 * s, y + 1.98 * s)], closed=True, fc=vidro, ec="none", zorder=z + 0.26))
    ax.add_patch(Polygon([(x + 0.05 * s, y + 1.2 * s), (x + 0.36 * s, y + 1.2 * s), (x + 0.32 * s, y + 1.98 * s),
                          (x + 0.05 * s, y + 1.98 * s)], closed=True, fc=vidro, ec="none", zorder=z + 0.26))
    ax.add_patch(Polygon([(x - 0.3 * s, y + 1.9 * s), (x - 0.18 * s, y + 1.98 * s), (x - 0.38 * s, y + 1.3 * s),
                          (x - 0.42 * s, y + 1.32 * s)], closed=True, fc="white", alpha=0.45, zorder=z + 0.27))
    ax.add_patch(FancyBboxPatch((x - 0.5 * s, y + 2.08 * s), 1.02 * s, 0.1 * s, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=cor, ec="none", zorder=z + 0.28))                                          # teto
    ax.add_patch(Ellipse((x - 0.05 * s, y + 2.22 * s), 0.2 * s, 0.09 * s, fc=acento, ec="none", zorder=z + 0.28))  # GNSS
    ax.add_patch(Wedge((xr, y + R), R + 0.1 * s, 10, 170, width=0.12 * s, fc=cor, ec="none", zorder=z + 0.31))  # para-lama
    ax.add_patch(Rectangle((x + 0.45 * s, y + 0.45 * s), 0.14 * s, 0.14 * s, fc=escuro, zorder=z + 0.2))       # engate
    return xd + 0.15 * s, y + 0.55 * s


def _capa_prescricao(cor, acento, W=8.27, H=11.69):
    """Capa dos mapas de prescrição. Céu calmo em cima para o título."""
    fig, ax = _fig(W, H)
    yh = 6.0
    gradiente(ax, 0, W, yh, H, mix(SOL[0], CEU, 0.5), "white", z=0)
    sol(ax, 1.3, yh + 0.55, 0.42, z=1)
    x = np.linspace(0, W, 500)
    for k, (b, a, t) in enumerate(((yh + 0.33, 0.2, 0.7), (yh + 0.16, 0.14, 0.5))):
        ax.fill_between(x, yh - 0.05, ondulado(x, b, a, 40 + k), color=mix(cor, CEU, t), lw=0, zorder=2 + k)
    arvores(ax, np.linspace(0.2, 8.1, 30), np.full(30, yh - 0.02), 0.2, mix(cor, "white", 0.1), z=4, sem=8)
    # chão pintado em zonas de dose (mapa de prescrição em perspectiva)
    P = Perspectiva(cx=W * 0.5, yh=yh, H=5.6, f=4.2)
    n_y, n_x = 800, 560
    Y, X = np.mgrid[0:yh - 0.01:complex(n_y), 0:W:complex(n_x)]
    gx, gz = P.inversa(X, Y)
    fcampo = _campo_ruido(21)
    campo = fcampo(gx, gz)
    lim = np.array([-0.97, -0.43, 0.0, 0.43, 0.97])            # sextis da normal: 6 zonas de área igual
    cls = np.searchsorted(lim, campo)
    paleta = np.array([mix(mix(c, "white", 0.5), cor, 0.1) for c in COR_PRESC])
    img = paleta[cls]
    nevoa = np.clip((Y - (yh - 1.6)) / 1.6, 0, 1)[..., None] ** 1.5 * 0.85
    img = img * (1 - nevoa) + np.array(mix(cor, CEU, 0.7)) * nevoa
    ax.imshow(img, extent=(0, W, 0, yh - 0.01), origin="lower", aspect="auto", zorder=5, interpolation="bilinear")
    cont = np.where(Y < yh - 0.7, campo, np.nan)
    ax.contour(X, Y, cont, levels=lim, colors="white", linewidths=0.7, alpha=0.5, zorder=5.1)
    # linhas de plantio convergindo ao ponto de fuga (discretas)
    for xl in np.arange(-14, 14.01, 0.2):
        Xl, Yl = P([xl, xl], [1.0, 45])
        ax.plot(Xl, Yl, color="#3B3B2E", lw=0.35, alpha=0.12, zorder=5.2)
    # trator com distribuidor a lanço
    s = 1.0
    xt, yt = 3.35, 2.35
    ax.add_patch(Ellipse((xt + 0.45, yt + 0.02), 5.8, 0.24, fc="black", alpha=0.16, lw=0, zorder=19.5))
    xd, yd = _trator(ax, xt, yt, s, cor, acento)
    r = np.random.default_rng(11)
    n = 700
    ang = r.uniform(np.radians(-35), np.radians(215), n)
    dist = (r.uniform(0.15, 1.0, n) ** 0.6) * 1.9
    gx_ = xd + dist * np.cos(ang) * 1.15 + 0.35
    gy_ = yd - 0.1 + dist * np.sin(ang) * 0.28
    ok = gx_ > xd - 0.2
    ax.scatter(gx_[ok], gy_[ok], s=r.uniform(1, 4, ok.sum()), color=mix(acento, "white", 0.25), alpha=0.85, lw=0,
               zorder=19)
    return fig


@functools.lru_cache(maxsize=12)
def ilustracao(tipo: str, cor: str, acento: str, dpi: int = 170) -> np.ndarray:
    """Imagem RGBA da ilustração ('geral', 'fertilidade' ou 'prescricao')."""
    f = {"geral": _capa_geral, "fertilidade": _capa_fertilidade, "prescricao": _capa_prescricao}[tipo]
    return _render(f(cor, acento), dpi)
