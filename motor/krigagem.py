"""Krigagem ordinária com semivariograma ajustado automaticamente.

Para cada atributo:
1. Semivariograma experimental (8 classes de distância até metade da maior distância).
2. Ajuste dos modelos esférico, exponencial e gaussiano (mínimos quadrados ponderados
   pelo nº de pares).
3. Escolha do modelo pelo menor erro na validação cruzada (deixa-um-fora).
4. Sem dependência espacial (efeito pepita ≥ 90% do patamar) ou poucos pontos →
   semivariograma padrão, com o motivo registrado.

Dois modos (parâmetro `modo` de `interpolar`):
- "fiel" (padrão dos mapas de fertilidade desde a v15): krigagem ordinária LOCAL, modelo esférico com alcance de
  4× a distância média entre pontos vizinhos e pepita de 5% (filtrada). Cada amostra composta representa a sua
  célula da grade, e o mapa mostra o que foi medido em cada ponto — é o comportamento dos mapas de amostragem em
  grade usados no mercado (calibrado contra books feitos no SMS).
- "estatistico" (v11–v14; base das prescrições): os passos 1–4 acima. Quando o semivariograma da área inteira
  não mostra estrutura, o mapa tende à média/tendência geral — mais conservador para dose, mas esconde a
  variação medida ponto a ponto.
Grau de dependência espacial (Cambardella et al., 1994): C0/(C0+C1) < 25% forte;
25–75% moderado; > 75% fraco.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist, pdist, squareform
from shapely import contains_xy

MIN_PONTOS_KRIGAGEM = 12


def _esf(h, c0, c1, a):
    h = np.asarray(h, float)
    r = np.where(h < a, c0 + c1 * (1.5 * h / a - 0.5 * (h / a) ** 3), c0 + c1)
    return np.where(h == 0, 0.0, r)


def _exp(h, c0, c1, a):
    h = np.asarray(h, float)
    return np.where(h == 0, 0.0, c0 + c1 * (1 - np.exp(-3 * h / a)))


def _gau(h, c0, c1, a):
    h = np.asarray(h, float)
    return np.where(h == 0, 0.0, c0 + c1 * (1 - np.exp(-3 * (h / a) ** 2)))


MODELOS = {"esférico": _esf, "exponencial": _exp, "gaussiano": _gau}


@dataclass
class Ajuste:
    metodo: str                 # 'krigagem' | 'IDW' | 'constante'
    modelo: str = ""
    c0: float = 0.0             # efeito pepita
    c1: float = 0.0             # contribuição
    alcance: float = 0.0        # m
    rmse_cv: float = float("nan")
    r2_cv: float = float("nan")
    n: int = 0
    motivo: str = ""
    outliers: int = 0           # valores extremos limitados antes da interpolação
    sem_estrutura: bool = False  # validação cruzada não supera a média: estrutura espacial não comprovada
    modo: str = "estatistico"   # 'fiel' (local, segue as amostras) | 'estatistico' (semivariograma ajustado)

    @property
    def gde(self) -> float:
        tot = self.c0 + self.c1
        return 100 * self.c0 / tot if tot > 0 else float("nan")

    @property
    def classe_dependencia(self) -> str:
        if self.metodo != "krigagem":
            return "—"
        g = self.gde
        return "forte" if g < 25 else ("moderada" if g <= 75 else "fraca")

    def descricao(self) -> str:
        if self.modo == "fiel" and self.metodo == "krigagem":
            r2 = f" · R² da validação cruzada {self.r2_cv:.2f}".replace(".", ",") if np.isfinite(self.r2_cv) else ""
            return (f"Krigagem ordinária local (fiel às amostras) · esférico · alcance {self.alcance:,.0f} m"
                    .replace(",", ".") + r2)
        if self.sem_estrutura and self.metodo == "krigagem":
            return ("Krigagem ordinária · ESTRUTURA ESPACIAL NÃO COMPROVADA (validação cruzada não supera a média) — "
                    "mapa exploratório")
        if self.metodo == "krigagem" and self.motivo:
            return (f"Krigagem ordinária · {self.motivo} · alcance {self.alcance:,.0f} m").replace(",", ".")
        if self.metodo == "krigagem":
            return (f"Krigagem ordinária · modelo {self.modelo} · alcance {self.alcance:,.0f} m · "
                    f"pepita {self.gde:.0f}% ({self.classe_dependencia})").replace(",", ".")
        if self.metodo == "IDW":
            return f"Inverso do quadrado da distância ({self.motivo})"
        return f"Valor constante ({self.motivo})"


def semivariograma(x, y, z, n_lags: int = 8):
    xy = np.column_stack([x, y])
    d = pdist(xy)
    g = 0.5 * pdist(z[:, None], "sqeuclidean")
    dmax = d.max() / 2
    bordas = np.linspace(0, dmax, n_lags + 1)
    lag, gam, npar = [], [], []
    for a, b in zip(bordas[:-1], bordas[1:]):
        m = (d > a) & (d <= b)
        if m.sum() >= 3:
            lag.append(d[m].mean())
            gam.append(g[m].mean())
            npar.append(m.sum())
    return np.array(lag), np.array(gam), np.array(npar), dmax


def _ajustar(modelo, lag, gam, npar, var, dmax):
    f = MODELOS[modelo]
    p0 = [0.2 * var, 0.8 * var, dmax / 2]
    lim = ([0.10 * var, 0, lag.min() * 0.5], [var * 3, var * 3, dmax * 1.5])   # pepita ≥ 10% da variância
    p, _ = curve_fit(f, lag, gam, p0=p0, bounds=lim, sigma=1 / np.sqrt(npar), maxfev=20000)
    return p


def _matriz_ok(xy, f, par):
    n = len(xy)
    C = np.ones((n + 1, n + 1))
    C[:n, :n] = f(squareform(pdist(xy)), *par)
    C[n, n] = 0
    return C


def _prever_ok(xy, z, xy_novo, f, par, blocos: int = 20000):
    n = len(xy)
    C = _matriz_ok(xy, f, par)
    Cinv = np.linalg.pinv(C)
    out = np.empty(len(xy_novo))
    for i in range(0, len(xy_novo), blocos):
        b = xy_novo[i:i + blocos]
        g = np.ones((len(b), n + 1))
        g[:, :n] = f(cdist(b, xy), *par)
        w = g @ Cinv.T                      # pesos (λ1..λn, μ)
        out[i:i + blocos] = w[:, :n] @ z
    return out


def _prever_suave(xy, z, xy_novo, f, par, blocos: int = 20000):
    """Krigagem ordinária com a pepita FILTRADA: prevê o componente contínuo do atributo.

    Na forma clássica γ(0) = 0 e o mapa "salta" para o valor exato em cima de cada amostra — o que aparece como um
    pontinho de outra cor em volta do ponto. Aqui γ(h→0) = pepita também entre alvo e amostra: a superfície fica
    contínua e passa a (quase) pepita de distância do valor medido."""
    n = len(xy)
    Cinv = np.linalg.pinv(_matriz_ok(xy, f, par))
    out = np.empty(len(xy_novo))
    for i in range(0, len(xy_novo), blocos):
        b = xy_novo[i:i + blocos]
        g = np.ones((len(b), n + 1))
        g[:, :n] = f(np.maximum(cdist(b, xy), 1e-9), *par)
        out[i:i + blocos] = (g @ Cinv.T)[:, :n] @ z
    return out


FIEL_PEPITA = 0.05      # fração da variância
FIEL_ALCANCE = 4.0      # × distância média ao vizinho mais próximo


def _interpolar_fiel(xy, z, z_orig, novo, n_out, sem_estrutura):
    n = len(z)
    var = z.var(ddof=1)
    d = squareform(pdist(xy))
    d_viz = np.sort(d, axis=1)[:, 1].mean()
    a = float(np.clip(FIEL_ALCANCE * d_viz, 1.0, max(d.max() * 1.5, 1.0)))
    par = np.array([FIEL_PEPITA * var, (1 - FIEL_PEPITA) * var, a])
    f = MODELOS["esférico"]
    rmse, r2 = _metricas(z, _loo_ok(xy, z, f, par)) if n <= 1500 else (float("nan"),) * 2
    sem = bool(np.isfinite(r2) and r2 <= 0)
    if sem and sem_estrutura == "uniforme":
        return np.full(len(novo), float(np.mean(z_orig))), Ajuste(
            "constante", n=n, rmse_cv=rmse, r2_cv=r2, outliers=n_out, sem_estrutura=True, modo="fiel",
            motivo="estrutura espacial não comprovada — valor médio uniforme")
    vals = np.clip(_prever_suave(xy, z, novo, f, par), z.min(), z.max())
    return vals, Ajuste("krigagem", "esférico", float(par[0]), float(par[1]), a, rmse, r2, n, outliers=n_out,
                        sem_estrutura=sem, modo="fiel")


def _loo_ok(xy, z, f, par):
    """Validação cruzada (deixa-um-fora) pela fórmula fechada de Dubrule (1983): uma inversão só.

    Para a krigagem ordinária, z_i − ẑ_(−i) = (A⁻¹·[z; 0])_i / (A⁻¹)_ii, com A a matriz aumentada."""
    n = len(z)
    try:
        Ainv = np.linalg.inv(_matriz_ok(xy, f, par))
        d = np.diag(Ainv)[:n]
        if np.all(np.abs(d) > 1e-12):
            return z - (Ainv[:n, :n] @ z) / d
    except np.linalg.LinAlgError:
        pass
    return _loo_ok_lento(xy, z, f, par)


def _loo_ok_lento(xy, z, f, par):
    pred = np.empty(len(z))
    idx = np.arange(len(z))
    for i in idx:
        m = idx != i
        pred[i] = _prever_ok(xy[m], z[m], xy[i:i + 1], f, par)[0]
    return pred


def idw(xy, z, xy_novo, p: float = 2.0):
    d = cdist(xy_novo, xy)
    d = np.maximum(d, 1e-6)
    w = 1 / d ** p
    return (w @ z) / w.sum(axis=1)


def _metricas(z, pred):
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))
    ss = np.sum((z - z.mean()) ** 2)
    r2 = float(1 - np.sum((z - pred) ** 2) / ss) if ss > 0 else float("nan")
    return rmse, r2


def _atipicos_locais(xy: np.ndarray, z: np.ndarray, k: int = 6, lim: float = 3.0) -> tuple[np.ndarray, int]:
    """Atípicos espaciais: ponto muito diferente dos seus k vizinhos mais próximos.

    Resíduo r = z − mediana dos vizinhos; escala robusta s = 1,4826·MAD(r). Pontos com |r| > lim·s
    e fora da faixa [mín, máx] dos vizinhos são trazidos para essa faixa. Evita os "alvos"
    (anéis concêntricos) que um único ponto isolado cria no mapa e na prescrição — zonas que
    não se consegue aplicar. O valor do laudo (e a dose da planilha) não muda; só o mapa.
    """
    n = len(z)
    if n < 10:
        return z, 0
    k = min(k, n - 1)
    d = squareform(pdist(xy))
    viz = np.argsort(d, axis=1)[:, 1:k + 1]
    zv = z[viz]
    r = z - np.median(zv, axis=1)
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    if s <= 0:
        return z, 0
    lo, hi = zv.min(axis=1), zv.max(axis=1)
    sel = (np.abs(r) > lim * s) & ((z < lo) | (z > hi))
    if not sel.any():
        return z, 0
    z = z.copy()
    z[sel] = np.clip(z[sel], lo[sel], hi[sel])
    return z, int(sel.sum())


def interpolar(x, y, z, gx, gy, forcar_idw: bool = False, limitar_extremos: bool = True,
               sem_estrutura: str = "exploratorio", modo: str = "estatistico") -> tuple[np.ndarray, Ajuste]:
    """Interpola z(x, y) nos pontos (gx, gy) por krigagem ordinária. Devolve (valores, ajuste).

    Sempre krigagem (padronização visual entre talhões e atributos). Quando o semivariograma
    experimental não permite um ajuste confiável — poucos pontos, efeito pepita puro ou alcance
    menor que a distância entre pontos —, usa-se um semivariograma padrão (esférico, pepita 15%
    da variância, alcance de 2,5× a distância média entre vizinhos), registrado no ajuste.

    Estrutura espacial não comprovada (R² da validação cruzada ≤ 0, isto é, a krigagem não prevê melhor que a
    média): `sem_estrutura="exploratorio"` mantém o mapa marcado como exploratório; `"uniforme"` usa o valor médio
    (mapa e dose uniformes). `limitar_extremos=False` desliga a limitação de valores extremos isolados.
    """
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    ok = np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
    x, y, z = x[ok], y[ok], z[ok]
    xy, novo = np.column_stack([x, y]), np.column_stack([np.ravel(gx), np.ravel(gy)])
    n = len(z)
    if n == 0:
        return np.full(len(novo), np.nan), Ajuste("constante", n=0, motivo="sem dados")
    if n < 3 or np.ptp(z) == 0:
        return np.full(len(novo), z.mean()), Ajuste("constante", n=n, motivo="valores iguais ou < 3 pontos")
    if forcar_idw:
        pred = np.array([idw(xy[np.arange(n) != i], z[np.arange(n) != i], xy[i:i + 1])[0] for i in range(n)])
        rmse, r2 = _metricas(z, pred)
        return idw(xy, z, novo), Ajuste("IDW", n=n, rmse_cv=rmse, r2_cv=r2, motivo="escolhido pelo usuário")

    # valores extremos (além de 3 intervalos interquartis) são limitados às cercas de Tukey:
    # um único ponto atípico não deve criar um "alvo" no mapa
    z_orig = z.copy()
    n_out = 0
    if modo == "fiel":
        # só a cerca global (3 intervalos interquartis), se pedida: os picos medidos ficam no mapa
        if n >= 10 and limitar_extremos:
            q1, q3 = np.percentile(z, [25, 75])
            if q3 > q1:
                lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
                n_out = int(((z < lo) | (z > hi)).sum())
                z = np.clip(z, lo, hi)
        if np.ptp(z) == 0:
            return np.full(len(novo), z.mean()), Ajuste("constante", n=n, motivo="valores iguais", modo="fiel")
        return _interpolar_fiel(xy, z, z_orig, novo, n_out, sem_estrutura)
    if n >= 10 and limitar_extremos:
        q1, q3 = np.percentile(z, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            n_out = int(((z < lo) | (z > hi)).sum())
            z = np.clip(z, lo, hi)
        z, n_loc = _atipicos_locais(xy, z)
        n_out += n_loc
    var = z.var(ddof=1)
    d_viz = np.sort(squareform(pdist(xy)), axis=1)[:, 1].mean()
    dmax_total = pdist(xy).max()
    melhor, motivo = None, ""
    if n >= MIN_PONTOS_KRIGAGEM:
        lag, gam, npar, dmax = semivariograma(x, y, z)
        if len(lag) >= 4:
            for nome, f in MODELOS.items():
                try:
                    par = _ajustar(nome, lag, gam, npar, var, dmax)
                except (RuntimeError, ValueError):
                    continue
                if n <= 600:
                    rmse, r2 = _metricas(z, _loo_ok(xy, z, f, par))
                else:                      # muitos pontos: escolhe pelo ajuste do semivariograma
                    rmse = float(np.sqrt(np.average((f(lag, *par) - gam) ** 2, weights=npar)))
                    r2 = float("nan")
                if melhor is None or rmse < melhor[2]:
                    melhor = (nome, par, rmse, r2)
            if melhor is not None:
                c0, c1, a = melhor[1]
                if c1 <= 0 or c0 / (c0 + c1) >= 0.90:
                    motivo, melhor = "efeito pepita puro", None
                elif a < 0.9 * d_viz:
                    motivo, melhor = f"alcance ajustado ({a:.0f} m) menor que a distância entre pontos", None
            else:
                motivo = "semivariograma não ajustou"
        else:
            motivo = "pares insuficientes"
    else:
        motivo = f"apenas {n} pontos"

    if melhor is not None:
        nome, par, rmse, r2 = melhor
        padrao = False
    else:                                   # semivariograma padrão
        nome = "esférico"
        a = min(max(2.5 * d_viz, 1.0), dmax_total * 1.5)
        par = np.array([0.15 * var, 0.85 * var, a])
        rmse, r2 = _metricas(z, _loo_ok(xy, z, MODELOS[nome], par)) if n <= 600 else (float("nan"),) * 2
        padrao = True
    sem = bool(np.isfinite(r2) and r2 <= 0)
    if sem and sem_estrutura == "uniforme":
        media = float(np.mean(z_orig))
        return np.full(len(novo), media), Ajuste("constante", n=n, rmse_cv=rmse, r2_cv=r2, outliers=n_out,
                                                 sem_estrutura=True,
                                                 motivo="estrutura espacial não comprovada — valor médio uniforme")
    vals = _prever_ok(xy, z, novo, MODELOS[nome], par)
    vals = np.clip(vals, z.min(), z.max())      # não extrapola além do observado
    aj = Ajuste("krigagem", nome, float(par[0]), float(par[1]), float(par[2]), rmse, r2, n, outliers=n_out,
                sem_estrutura=sem)
    if padrao:
        aj.motivo = f"semivariograma padrão ({motivo})"
    return vals, aj


# --------------------------------------------------------------------- grade
@dataclass
class Grade:
    x: np.ndarray        # centros (1D)
    y: np.ndarray
    mascara: np.ndarray  # (ny, nx) True dentro do talhão
    res: float

    @property
    def xx(self):
        return np.meshgrid(self.x, self.y)[0]

    @property
    def yy(self):
        return np.meshgrid(self.x, self.y)[1]

    @property
    def area_pixel_ha(self) -> float:
        return self.res ** 2 / 10_000


def grade_do_poligono(poligono, res: float = 10.0) -> Grade:
    x0, y0, x1, y1 = poligono.bounds
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y1 - res / 2, y0, -res)      # de cima para baixo
    X, Y = np.meshgrid(xs, ys)
    m = contains_xy(poligono, X, Y)
    return Grade(xs, ys, m, res)


def interpolar_grade(grade: Grade, x, y, z, forcar_idw=False, **kw) -> tuple[np.ndarray, Ajuste]:
    X, Y = grade.xx, grade.yy
    out = np.full(X.shape, np.nan)
    vals, aj = interpolar(x, y, z, X[grade.mascara], Y[grade.mascara], forcar_idw, **kw)
    out[grade.mascara] = vals
    return out, aj
