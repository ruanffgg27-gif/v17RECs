"""Planejamento da amostragem de solo a partir do perímetro.

Dois caminhos:

1. GRADE — o usuário escolhe a densidade (ha por ponto). O talhão é dividido em células iguais, alinhadas ao
   lado mais comprido do talhão (ou ao ângulo informado), e cada célula recebe um ponto:
   - "quadrada": linhas e colunas alinhadas (o padrão de mercado);
   - "desencontrada": linhas alternadas deslocadas meia célula (malha triangular — para a mesma densidade,
     a maior distância de qualquer lugar até um ponto é menor, o que favorece a interpolação);
   - posição do ponto: centro da célula ou sorteado dentro dela (amostragem aleatória estratificada, que evita
     coincidir sempre com a mesma linha de plantio ou de tráfego).
   Células cortadas pelo perímetro recebem ponto quando sobram com pelo menos `fracao_min` da área; o ponto vai
   para dentro do pedaço que sobrou. A origem da malha é escolhida entre várias posições para o número de pontos
   ficar o mais próximo de área ÷ densidade.

2. ZONAS DE MANEJO — o talhão é dividido em zonas pelo vigor relativo histórico da lavoura (NDVI do pico de
   cada ano agrícola, em relação à mediana do talhão no ano, média de vários anos). Dentro de cada zona os
   pontos são espalhados de forma equilibrada (k-médias espacial), com a densidade escolhida por zona.

Os pontos são numerados em "zigue-zague" (ordem de caminhamento) e exportados em KML com o nome = número,
no mesmo padrão que a aba de recomendação usa para ligar amostras e pontos (Pontos_TH_1.kml).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage
from shapely import affinity, contains_xy
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .geo import projetar

FORMATOS = {"quadrada": "Grade quadrada", "desencontrada": "Grade desencontrada (triangular)"}
POSICOES = {"centro": "Centro da célula", "aleatoria": "Aleatória dentro da célula"}
NOMES_ZONA = {2: ["Baixo vigor", "Alto vigor"], 3: ["Baixo vigor", "Médio vigor", "Alto vigor"],
              4: ["Baixo vigor", "Médio-baixo", "Médio-alto", "Alto vigor"],
              5: ["Muito baixo", "Baixo vigor", "Médio vigor", "Alto vigor", "Muito alto"]}
COR_ZONA = {2: ["#D73027", "#1A9850"], 3: ["#D73027", "#FEE08B", "#1A9850"],
            4: ["#D73027", "#FDAE61", "#A6D96A", "#1A9850"],
            5: ["#A50026", "#F46D43", "#FEE08B", "#66BD63", "#006837"]}


@dataclass
class TalhaoAmostragem:
    nome: str
    poligono: Polygon | MultiPolygon          # UTM (m)
    epsg: int

    @property
    def area_ha(self) -> float:
        return self.poligono.area / 1e4


@dataclass
class Plano:
    talhoes: list[TalhaoAmostragem]
    pontos: pd.DataFrame            # Ponto, Talhão, Zona, x, y, Longitude, Latitude, Área (ha)
    celulas: list = field(default_factory=list, repr=False)          # geometria (UTM) representada por cada ponto
    zonas: list[tuple[str, str, object]] = field(default_factory=list, repr=False)   # (talhão, zona, geometria)
    metodo: str = ""
    descricao: str = ""
    avisos: list[str] = field(default_factory=list)

    @property
    def area_ha(self) -> float:
        return sum(t.area_ha for t in self.talhoes)

    @property
    def ha_por_ponto(self) -> float:
        return self.area_ha / len(self.pontos) if len(self.pontos) else float("nan")


# ------------------------------------------------------------------ grade
def orientacao(pol) -> float:
    """Ângulo (graus, 0–180) do lado mais comprido do menor retângulo que contém o talhão."""
    r = pol.minimum_rotated_rectangle
    if not isinstance(r, Polygon):
        return 0.0
    c = np.asarray(r.exterior.coords)
    lados = np.diff(c[:4 + 1], axis=0)
    i = int(np.argmax(np.hypot(lados[:, 0], lados[:, 1])))
    return float(np.degrees(np.arctan2(lados[i, 1], lados[i, 0])) % 180)


def _ponto_dentro(pedaco, alvo: Point, margem: float):
    """Ponto para a célula: o alvo, se cai dentro do pedaço com folga da borda; senão o ponto interno mais
    "central" do pedaço (afastado da borda)."""
    nucleo = pedaco.buffer(-margem) if margem > 0 else pedaco
    if not nucleo.is_empty and nucleo.contains(alvo):
        return alvo
    base = nucleo if not nucleo.is_empty else pedaco
    c = base.centroid
    return c if base.contains(c) else base.representative_point()


def _celulas(pol_r, lado: float, formato: str, ox: float, oy: float):
    """Células (linha, coluna, geometria) cobrindo o polígono já rotacionado."""
    x0, y0, x1, y1 = pol_r.bounds
    out = []
    j0 = int(np.floor((y0 - oy) / lado))
    j1 = int(np.ceil((y1 - oy) / lado))
    for j in range(j0, j1):
        desl = lado / 2 if (formato == "desencontrada" and j % 2) else 0.0
        i0 = int(np.floor((x0 - ox - desl) / lado))
        i1 = int(np.ceil((x1 - ox - desl) / lado))
        for i in range(i0, i1):
            cx = ox + desl + i * lado
            cy = oy + j * lado
            out.append((j, i, box(cx, cy, cx + lado, cy + lado)))
    return out


def _grade_talhao(pol, ha_por_ponto: float, formato: str, angulo: float | None, posicao: str,
                  fracao_min: float, margem_borda: float, rng,
                  deslocamento: tuple[float, float] = (0.0, 0.0)) -> tuple[list[Point], list, float]:
    lado = float(np.sqrt(ha_por_ponto * 1e4))
    ang = orientacao(pol) if angulo is None else float(angulo)
    centro = pol.centroid
    pol_r = affinity.rotate(pol, -ang, origin=centro)
    alvo_n = max(1, round(pol.area / (lado * lado)))
    melhor = None
    passos = [0.0, 0.25, 0.5, 0.75]
    x0, y0 = pol_r.bounds[:2]
    for fx in passos:
        for fy in passos:
            itens = []
            for j, i, cel in _celulas(pol_r, lado, formato, x0 - fx * lado, y0 - fy * lado):
                ped = cel.intersection(pol_r)
                if ped.is_empty or ped.area < fracao_min * lado * lado:
                    continue
                itens.append((j, i, cel, ped))
            coberto = sum(p.area for *_, p in itens) / pol_r.area
            inteiras = sum(p.area > 0.97 * lado * lado for *_, p in itens)
            # 1º: número de pontos perto do alvo; 2º: mais área representada; 3º: mais células inteiras
            nota = (-abs(len(itens) - alvo_n), round(coberto, 3), inteiras)
            if melhor is None or nota > melhor[0]:
                melhor = (nota, itens, (x0 - fx * lado, y0 - fy * lado))
    itens = melhor[1]
    de, dn = deslocamento
    if de or dn:
        # malha deslocada pelo usuário (metros para leste e para norte) a partir da posição automática:
        # as células são recortadas de novo pelo perímetro, então pontos entram e saem pelas bordas
        a = np.radians(-ang)
        sx, sy = de * np.cos(a) - dn * np.sin(a), de * np.sin(a) + dn * np.cos(a)
        ox, oy = melhor[2]
        itens = []
        for j, i, cel in _celulas(pol_r, lado, formato, ox + sx, oy + sy):
            ped = cel.intersection(pol_r)
            if not ped.is_empty and ped.area >= fracao_min * lado * lado:
                itens.append((j, i, cel, ped))
    if not itens:                                   # talhão menor que a fração mínima de uma célula
        itens = [(0, 0, pol_r.envelope, pol_r)]
    pts, cels, ordem = [], [], []
    margem = min(margem_borda, lado * 0.2)
    for j, i, cel, ped in itens:
        if posicao == "aleatoria":
            nucleo = ped.buffer(-margem)
            base = nucleo if not nucleo.is_empty else ped
            bx0, by0, bx1, by1 = base.bounds
            p = None
            for _ in range(60):
                c = Point(rng.uniform(bx0, bx1), rng.uniform(by0, by1))
                if base.contains(c):
                    p = c
                    break
            p = p or _ponto_dentro(ped, cel.centroid, margem)
        else:
            p = _ponto_dentro(ped, cel.centroid, margem)
        pts.append(p)
        cels.append(ped)
        ordem.append((j, p.x))
    # caminhamento em zigue-zague: linha a linha, alternando o sentido
    linhas = sorted({j for j, _ in ordem})
    pos = {j: k for k, j in enumerate(linhas)}
    idx = sorted(range(len(pts)), key=lambda k: (pos[ordem[k][0]], ordem[k][1] if pos[ordem[k][0]] % 2 == 0
                                                 else -ordem[k][1]))
    pts = [affinity.rotate(pts[k], ang, origin=centro) for k in idx]
    cels = [affinity.rotate(cels[k], ang, origin=centro) for k in idx]
    return pts, cels, ang


def _tabela(talhoes, por_talhao, numerar_por_talhao: bool) -> tuple[pd.DataFrame, list]:
    linhas, cels, n = [], [], 0
    for t, (pts, cs, zonas) in zip(talhoes, por_talhao):
        if numerar_por_talhao:
            n = 0
        for p, c, z in zip(pts, cs, zonas):
            n += 1
            ll = projetar(p, t.epsg, inverso=True)
            linhas.append({"Ponto": n, "Talhão": t.nome, "Zona": z, "x": p.x, "y": p.y, "Longitude": ll.x,
                           "Latitude": ll.y, "Área (ha)": c.area / 1e4 if c is not None else np.nan})
            cels.append(c)
    return pd.DataFrame(linhas, columns=["Ponto", "Talhão", "Zona", "x", "y", "Longitude", "Latitude",
                                         "Área (ha)"]), cels


def plano_grade(talhoes: list[TalhaoAmostragem], ha_por_ponto: float = 5.0, formato: str = "quadrada",
                angulo: float | None = None, posicao: str = "centro", fracao_min: float = 0.35,
                margem_borda: float = 20.0, numerar_por_talhao: bool = True, semente: int = 1,
                deslocamento: tuple[float, float] = (0.0, 0.0), giro: float = 0.0) -> Plano:
    """Pontos em grade com a densidade pedida. `angulo=None` alinha a malha ao lado mais comprido de cada talhão.

    `deslocamento` = (leste, norte) em metros e `giro` em graus (anti-horário) movem a malha inteira a partir da
    posição automática — para fugir de um carreador, alinhar com as linhas de plantio etc."""
    if ha_por_ponto <= 0:
        raise ValueError("A densidade (ha por ponto) deve ser maior que zero.")
    rng = np.random.default_rng(semente)
    por_talhao, angs = [], []
    for t in talhoes:
        ang_t = None if (angulo is None and not giro) else ((orientacao(t.poligono) if angulo is None
                                                              else float(angulo)) + giro) % 180
        pts, cels, ang = _grade_talhao(t.poligono, ha_por_ponto, formato, ang_t, posicao, fracao_min,
                                       margem_borda, rng, deslocamento)
        por_talhao.append((pts, cels, [""] * len(pts)))
        angs.append(ang)
    df, cels = _tabela(talhoes, por_talhao, numerar_por_talhao)
    lado = np.sqrt(ha_por_ponto * 1e4)
    ajuste = ""
    if deslocamento[0] or deslocamento[1] or giro:
        ajuste = (f" · malha deslocada {deslocamento[0]:+.0f} m (L) e {deslocamento[1]:+.0f} m (N)"
                  + (f", girada {giro:+.0f}°" if giro else ""))
    p = Plano(talhoes, df, cels, metodo="grade",
              descricao=f"{FORMATOS[formato]} de {lado:.0f} × {lado:.0f} m ({ha_por_ponto:g} ha por ponto) · "
                        f"{POSICOES[posicao].lower()}{ajuste}")
    for t in talhoes:
        n = int((df["Talhão"] == t.nome).sum())
        if n and abs(t.area_ha / n - ha_por_ponto) > 0.25 * ha_por_ponto:
            p.avisos.append(f"{t.nome}: {n} ponto(s) em {t.area_ha:.1f} ha = 1 ponto a cada {t.area_ha / n:.1f} ha "
                            f"(pedido: {ha_por_ponto:g} ha) — talhão pequeno ou muito recortado para essa malha.")
    return p


# ------------------------------------------------------------------ zonas de manejo
def classificar_zonas(vigor: np.ndarray, mascara: np.ndarray, n_zonas: int = 3, res: float = 10.0,
                      area_min_ha: float = 1.0, suavizacao_m: float = 30.0) -> np.ndarray:
    """Raster de zonas (0 = menor vigor … n−1 = maior; −1 fora) a partir do vigor relativo histórico.

    Suaviza (~30 m), separa em classes de mesma área (quantis — toda zona tem tamanho útil para amostrar) e
    elimina manchas menores que `area_min_ha`, que não dá para amostrar nem manejar em separado."""
    ok = mascara & np.isfinite(vigor)
    if ok.sum() < n_zonas * 4:
        raise ValueError("Imagens insuficientes para separar zonas neste talhão.")
    sig = max(suavizacao_m / res, 0.5)
    num = ndimage.gaussian_filter(np.where(ok, vigor, 0.0), sig)
    den = ndimage.gaussian_filter(ok.astype(float), sig)
    V = np.where(ok, num / np.maximum(den, 1e-9), np.nan)
    cortes = np.quantile(V[ok], np.linspace(0, 1, n_zonas + 1)[1:-1])
    C = np.full(vigor.shape, -1, int)
    C[ok] = np.searchsorted(cortes, V[ok], side="right")
    # pixels do talhão sem imagem: zona do vizinho mais próximo
    falta = mascara & ~ok
    if falta.any():
        idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        C[falta] = C[tuple(idx)][falta]
    min_px = max(1, int(round(area_min_ha * 1e4 / res ** 2)))
    for _ in range(8):                               # funde manchas pequenas na zona vizinha dominante
        mudou = False
        for z in range(n_zonas):
            rot, n = ndimage.label(C == z)
            if not n:
                continue
            tam = ndimage.sum(np.ones_like(rot), rot, np.arange(1, n + 1))
            for k in np.where(tam < min_px)[0] + 1:
                m = rot == k
                borda = ndimage.binary_dilation(m, iterations=1) & ~m & mascara
                viz = C[borda]
                viz = viz[(viz >= 0) & (viz != z)]
                if len(viz):
                    C[m] = np.bincount(viz).argmax()
                    mudou = True
        if not mudou:
            break
    C[~mascara] = -1
    return C


def _poligono_da_zona(C: np.ndarray, z: int, xs, ys, res: float, recorte):
    """Polígono (UTM) dos pixels da zona z, com as bordas em "escada" amaciadas e recortado pelo perímetro."""
    lin, col = np.where(C == z)
    if not len(lin):
        return None
    faixas = []
    for i in np.unique(lin):                         # une os pixels de cada linha em retângulos (rápido)
        c = np.sort(col[lin == i])
        quebras = np.where(np.diff(c) > 1)[0]
        ini = np.r_[c[0], c[quebras + 1]]
        fim = np.r_[c[quebras], c[-1]]
        for a, b in zip(ini, fim):
            faixas.append(box(xs[a] - res / 2, ys[i] - res / 2, xs[b] + res / 2, ys[i] + res / 2))
    r = res * 1.5                                    # abre e fecha: arredonda os degraus dos pixels
    g = unary_union(faixas).buffer(-r).buffer(2 * r).buffer(-r).simplify(res * 0.7)
    return g.intersection(recorte)


def _kmedias(xy: np.ndarray, k: int, rng, iteracoes: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """k-médias simples (Lloyd) com início k-means++: centros bem espalhados nos pixels da zona."""
    n = len(xy)
    k = min(k, n)
    centros = [xy[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((xy[:, None, :] - np.array(centros)[None, :, :]) ** 2).sum(-1), axis=1)
        centros.append(xy[int(np.argmax(d2))])        # o mais distante dos já escolhidos (determinístico)
    c = np.array(centros, float)
    for _ in range(iteracoes):
        rot = ((xy[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
        novo = np.array([xy[rot == j].mean(0) if (rot == j).any() else c[j] for j in range(k)])
        if np.allclose(novo, c, atol=0.5):
            break
        c = novo
    return c, ((xy[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)


def plano_zonas(talhoes: list[TalhaoAmostragem], vigores: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                n_zonas: int = 3, ha_por_ponto: float = 5.0, min_por_zona: int = 2, area_min_ha: float = 1.0,
                margem_borda: float = 20.0, numerar_por_talhao: bool = True, semente: int = 1,
                anos: list[str] | None = None) -> Plano:
    """Pontos por zona de manejo. `vigores[i]` = (vigor relativo, máscara, xs, ys) do talhão i, em grade regular
    (xs crescente, ys decrescente, UTM)."""
    rng = np.random.default_rng(semente)
    por_talhao, zonas_out = [], []
    nomes = NOMES_ZONA[n_zonas]
    for t, (V, M, xs, ys) in zip(talhoes, vigores):
        res = float(abs(xs[1] - xs[0]))
        C = classificar_zonas(V, M, n_zonas, res, area_min_ha)
        X, Y = np.meshgrid(xs, ys)
        pts, cels, zs = [], [], []
        nucleo = t.poligono.buffer(-margem_borda)
        dentro = contains_xy(nucleo, X, Y) if not nucleo.is_empty else M
        usado = None                                  # partição exata: cada zona tira o que as anteriores ocuparam
        for z in range(n_zonas):
            if z == n_zonas - 1 and usado is not None:
                geom = t.poligono.difference(usado)
            else:
                geom = _poligono_da_zona(C, z, xs, ys, res, t.poligono)
                if geom is not None and usado is not None:
                    geom = geom.difference(usado)
            if geom is None or geom.is_empty:
                continue
            geom = geom.buffer(0)
            usado = geom if usado is None else unary_union([usado, geom])
            zonas_out.append((t.nome, nomes[z], geom))
            sel = (C == z) & dentro
            if sel.sum() < 4:
                sel = C == z
            xy = np.column_stack([X[sel], Y[sel]])
            k = max(min_por_zona, int(round(geom.area / 1e4 / ha_por_ponto)))
            centros, rot = _kmedias(xy, k, rng)
            for j in range(len(centros)):
                grupo = xy[rot == j]
                if not len(grupo):
                    continue
                p = grupo[((grupo - centros[j]) ** 2).sum(1).argmin()]     # pixel da zona mais próximo do centro
                pts.append(Point(float(p[0]), float(p[1])))
                cels.append(None)
                zs.append(nomes[z])
        # caminhamento: vizinho mais próximo a partir do ponto mais a oeste/sul
        if pts:
            xy = np.array([[p.x, p.y] for p in pts])
            falta = list(range(len(pts)))
            atual = int(np.lexsort((xy[:, 1], xy[:, 0]))[0])
            ordem = [atual]
            falta.remove(atual)
            while falta:
                d = ((xy[falta] - xy[atual]) ** 2).sum(1)
                atual = falta[int(d.argmin())]
                ordem.append(atual)
                falta.remove(atual)
            pts, cels, zs = [pts[i] for i in ordem], [cels[i] for i in ordem], [zs[i] for i in ordem]
        por_talhao.append((pts, cels, zs))
    df, cels = _tabela(talhoes, por_talhao, numerar_por_talhao)
    for nome_t, nome_z, g in zonas_out:                 # área representada por ponto = área da zona ÷ pontos
        sel = (df["Talhão"] == nome_t) & (df["Zona"] == nome_z)
        if sel.any():
            df.loc[sel, "Área (ha)"] = g.area / 1e4 / sel.sum()
    periodo = f" · anos agrícolas {anos[0]} a {anos[-1]}" if anos else ""
    return Plano(talhoes, df, cels, zonas_out, metodo="zonas",
                 descricao=f"{n_zonas} zonas de manejo pelo vigor histórico (NDVI Sentinel-2{periodo}) · "
                           f"1 ponto a cada {ha_por_ponto:g} ha em cada zona (mínimo {min_por_zona})")


def vigor_da_lavoura(L) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """(vigor relativo médio, máscara, xs, ys) a partir de motor.sat.lavoura.Lavoura."""
    if not L or not L.vigor_rel:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        V = np.nanmean(np.stack(list(L.vigor_rel.values())), axis=0)
    return V, L.grade.mascara, L.grade.xs, L.grade.ys


# ------------------------------------------------------------------ edição manual
def _talhao_de(plano: Plano, nome: str) -> TalhaoAmostragem:
    return next(t for t in plano.talhoes if t.nome == nome)


def _zona_em(plano: Plano, talhao: str, p: Point) -> str:
    for tal, zona, g in plano.zonas:
        if tal == talhao and g.buffer(0.5).contains(p):
            return zona
    return ""


def renumerar(plano: Plano, por_talhao: bool = True) -> None:
    """Numeração corrida 1…n na ordem atual da tabela (por talhão ou na fazenda)."""
    df = plano.pontos
    if por_talhao:
        df["Ponto"] = df.groupby("Talhão", sort=False).cumcount() + 1
    else:
        df["Ponto"] = np.arange(1, len(df) + 1)


def editar_pontos(plano: Plano, mover: dict | None = None, remover: list | None = None,
                  adicionar: list | None = None, por_talhao: bool = True) -> list[str]:
    """Aplica edições feitas no mapa. Índices = posição da linha em plano.pontos.

    mover: {índice: (lon, lat)} — o ponto só muda se o novo lugar ficar dentro do MESMO talhão;
    remover: [índices]; adicionar: [(lon, lat)] — entra no talhão que contém o lugar clicado.
    Depois de remover/adicionar a numeração é refeita. Devolve as edições recusadas (texto para o usuário)."""
    recusas: list[str] = []
    df = plano.pontos.reset_index(drop=True)
    cels = list(plano.celulas) if len(plano.celulas) == len(df) else [None] * len(df)
    for i, (lon, lat) in (mover or {}).items():
        i = int(i)
        if not 0 <= i < len(df):
            continue
        t = _talhao_de(plano, df.at[i, "Talhão"])
        p = projetar(Point(float(lon), float(lat)), t.epsg)
        if not t.poligono.contains(p):
            recusas.append(f"Ponto {int(df.at[i, 'Ponto'])} ({t.nome}): o novo lugar fica fora do talhão — mantido "
                           "onde estava.")
            continue
        df.loc[i, ["x", "y", "Longitude", "Latitude"]] = [p.x, p.y, float(lon), float(lat)]
        if plano.zonas:
            df.at[i, "Zona"] = _zona_em(plano, t.nome, p) or df.at[i, "Zona"]
    novos = []
    for lon, lat in adicionar or []:
        alvo = None
        for t in plano.talhoes:
            p = projetar(Point(float(lon), float(lat)), t.epsg)
            if t.poligono.contains(p):
                alvo = (t, p)
                break
        if alvo is None:
            recusas.append("Ponto novo fora de todos os talhões — ignorado.")
            continue
        t, p = alvo
        novos.append({"Ponto": 0, "Talhão": t.nome, "Zona": _zona_em(plano, t.nome, p), "x": p.x, "y": p.y,
                      "Longitude": float(lon), "Latitude": float(lat), "Área (ha)": np.nan})
    fora = sorted({int(i) for i in (remover or []) if 0 <= int(i) < len(df)})
    if fora:
        df = df.drop(index=fora)
        cels = [c for k, c in enumerate(cels) if k not in set(fora)]
    if novos:
        df = pd.concat([df, pd.DataFrame(novos)], ignore_index=True)
        cels += [None] * len(novos)
        ordem = {t.nome: k for k, t in enumerate(plano.talhoes)}          # novos ao fim do seu talhão
        pos = df["Talhão"].map(ordem).to_numpy()
        idx = np.argsort(pos, kind="stable")
        df = df.iloc[idx]
        cels = [cels[k] for k in idx]
    plano.pontos = df.reset_index(drop=True)
    plano.celulas = cels
    if fora or novos:
        renumerar(plano, por_talhao)
        if "à mão" not in plano.descricao:
            plano.descricao += " · pontos incluídos/removidos à mão"
    return recusas


def dados_do_mapa(plano: Plano) -> dict:
    """Geometrias em lon/lat para o editor interativo (perímetros, células, zonas e pontos)."""
    def aneis(g, epsg):
        ll = projetar(g, epsg, inverso=True)
        partes = ll.geoms if hasattr(ll, "geoms") else [ll]
        out = []
        for p in partes:
            if isinstance(p, Polygon) and not p.is_empty:
                out.append([[[round(y, 7), round(x, 7)] for x, y in r.coords] for r in [p.exterior, *p.interiors]])
        return out
    epsg = {t.nome: t.epsg for t in plano.talhoes}
    n_z = len({z for _, z, _ in plano.zonas})
    cores = dict(zip(NOMES_ZONA.get(n_z, []), COR_ZONA.get(n_z, [])))
    cel = []
    if not plano.zonas and len(plano.celulas) == len(plano.pontos) and len(plano.pontos) <= 1500:
        for (_, r), c in zip(plano.pontos.iterrows(), plano.celulas):
            if c is not None and not c.is_empty:
                cel += aneis(c.simplify(1.0), epsg[r["Talhão"]])
    return {
        "talhoes": [{"nome": t.nome, "poligonos": aneis(t.poligono, t.epsg)} for t in plano.talhoes],
        "zonas": [{"nome": f"{tal} · {z}", "cor": cores.get(z, "#CCCCCC"), "poligonos": aneis(g.simplify(2.0), epsg[tal])}
                  for tal, z, g in plano.zonas],
        "celulas": cel,
        "pontos": [{"i": int(i), "n": int(r["Ponto"]), "t": r["Talhão"], "z": r["Zona"] or "",
                    "lat": round(float(r["Latitude"]), 7), "lon": round(float(r["Longitude"]), 7)}
                   for i, r in plano.pontos.reset_index(drop=True).iterrows()],
    }


# ------------------------------------------------------------------ talhões a partir dos KML
def talhoes_de_kmls(kmls) -> list[TalhaoAmostragem]:
    """Cada polígono dos arquivos vira um talhão. Nome: número do talhão no nome do arquivo ('Perimetro_TH_3.kml'
    → 'TH 03'), ou do campo 'Field'; sem número, 'TH 01', 'TH 02'… na ordem dos arquivos."""
    from .geo import epsg_utm_sirgas, numero_talhao
    brutos = []
    for k in kmls:
        n_arq = numero_talhao(k.nome) or numero_talhao(str(k.atributos.get("Field", "")))
        for i, p in enumerate(k.poligonos):
            brutos.append((n_arq if len(k.poligonos) == 1 else None, p))
    usados = {int(n) for n, _ in brutos if n}
    livre = (i for i in range(1, 10_000) if i not in usados)
    out, vistos = [], set()
    for n, p in brutos:
        num = int(n) if n and int(n) not in vistos else next(livre)
        vistos.add(num)
        c = p.representative_point()
        epsg = epsg_utm_sirgas(c.x, c.y)
        out.append(TalhaoAmostragem(f"TH {num:02d}", projetar(p, epsg), epsg))
    return sorted(out, key=lambda t: t.nome)


# ------------------------------------------------------------------ exportação
def _kml(pontos: pd.DataFrame, nome: str) -> bytes:
    def desc(r):
        partes = [str(r["Talhão"])] + ([str(r["Zona"])] if r["Zona"] else [])
        return " · ".join(partes)
    pm = "".join(
        f"<Placemark><name>{int(r['Ponto'])}</name><description>{desc(r)}</description><Point><coordinates>"
        f"{r['Longitude']:.7f},{r['Latitude']:.7f},0</coordinates></Point></Placemark>" for _, r in pontos.iterrows())
    return (f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>{nome}</name><Folder><name>{nome}</name>{pm}</Folder></Document></kml>").encode("utf-8")


def _gpx(pontos: pd.DataFrame, nome: str) -> bytes:
    w = "".join(f'<wpt lat="{r["Latitude"]:.7f}" lon="{r["Longitude"]:.7f}"><name>{int(r["Ponto"])}</name>'
                f'<desc>{r["Talhão"]}{" · " + r["Zona"] if r["Zona"] else ""}</desc></wpt>'
                for _, r in pontos.iterrows())
    return (f'<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1" creator="ATRIA" '
            f'xmlns="http://www.topografix.com/GPX/1/1"><metadata><name>{nome}</name></metadata>{w}</gpx>'
            ).encode("utf-8")


def _shp_pontos(pontos: pd.DataFrame, nome: str) -> dict[str, bytes]:
    import shapefile
    from .prescricao import WGS84_PRJ
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POINT)
    w.field("Ponto", "N", 6, 0)
    w.field("Talhao", "C", 20)
    w.field("Zona", "C", 20)
    for _, r in pontos.iterrows():
        w.point(float(r["Longitude"]), float(r["Latitude"]))
        w.record(int(r["Ponto"]), str(r["Talhão"]), str(r["Zona"] or ""))
    w.close()
    return {f"{nome}.shp": shp.getvalue(), f"{nome}.shx": shx.getvalue(), f"{nome}.dbf": dbf.getvalue(),
            f"{nome}.prj": WGS84_PRJ.encode()}


def tabela_coordenadas(plano: Plano) -> pd.DataFrame:
    t = plano.pontos[["Ponto", "Talhão", "Zona", "Latitude", "Longitude", "Área (ha)"]].copy()
    if not plano.zonas:
        t = t.drop(columns="Zona")
    return t


def excel_pontos(plano: Plano) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        tabela_coordenadas(plano).round({"Latitude": 7, "Longitude": 7, "Área (ha)": 2}).to_excel(
            xw, sheet_name="Pontos", index=False)
        resumo = plano.pontos.groupby(["Talhão"] + (["Zona"] if plano.zonas else []), sort=False).agg(
            Pontos=("Ponto", "size")).reset_index()
        areas = {t.nome: t.area_ha for t in plano.talhoes}
        if plano.zonas:
            az = {(a, b): g.area / 1e4 for a, b, g in plano.zonas}
            resumo["Área (ha)"] = [az.get((a, b), np.nan) for a, b in zip(resumo["Talhão"], resumo["Zona"])]
        else:
            resumo["Área (ha)"] = resumo["Talhão"].map(areas)
        resumo["ha por ponto"] = resumo["Área (ha)"] / resumo["Pontos"]
        resumo.round(2).to_excel(xw, sheet_name="Resumo", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(12, len(str(col[0].value)) + 4)
    return buf.getvalue()


def zip_amostragem(plano: Plano) -> bytes:
    """Pontos_TH_n.kml (um por talhão, pronto para a aba de recomendação), todos os pontos em KML, GPX e
    shapefile, as zonas (se houver) em KML e a planilha de coordenadas."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for t in plano.talhoes:
            d = plano.pontos[plano.pontos["Talhão"] == t.nome]
            if not len(d):
                continue
            num = "".join(c for c in t.nome if c.isdigit()).lstrip("0") or "1"
            z.writestr(f"KML por talhao/Pontos_TH_{num}.kml", _kml(d, f"Pontos {t.nome}"))
        z.writestr("Pontos_todos.kml", _kml(plano.pontos, "Pontos de amostragem"))
        z.writestr("Pontos_todos.gpx", _gpx(plano.pontos, "Pontos de amostragem"))
        for nome, dados in _shp_pontos(plano.pontos, "PONTOS").items():
            z.writestr(f"Shapefile/{nome}", dados)
        if plano.zonas:
            pm = ""
            for tal, zona, g in plano.zonas:
                ll = projetar(g, next(t.epsg for t in plano.talhoes if t.nome == tal), inverso=True)
                partes = ll.geoms if hasattr(ll, "geoms") else [ll]
                for p in partes:
                    if not isinstance(p, Polygon) or p.is_empty:
                        continue
                    ext = " ".join(f"{x:.7f},{y:.7f},0" for x, y in p.exterior.coords)
                    furos = "".join("<innerBoundaryIs><LinearRing><coordinates>"
                                    + " ".join(f"{x:.7f},{y:.7f},0" for x, y in r.coords)
                                    + "</coordinates></LinearRing></innerBoundaryIs>" for r in p.interiors)
                    pm += (f"<Placemark><name>{tal} · {zona}</name><Polygon><outerBoundaryIs><LinearRing>"
                           f"<coordinates>{ext}</coordinates></LinearRing></outerBoundaryIs>{furos}</Polygon>"
                           "</Placemark>")
            z.writestr("Zonas_de_manejo.kml",
                       ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                        f"<Document><name>Zonas de manejo</name>{pm}</Document></kml>").encode("utf-8"))
        z.writestr("Coordenadas.xlsx", excel_pontos(plano))
        z.writestr("LEIA-ME.txt", (
            "PONTOS DE AMOSTRAGEM DE SOLO\n\n"
            f"{plano.descricao}\n{len(plano.pontos)} pontos em {plano.area_ha:.2f} ha "
            f"(1 ponto a cada {plano.ha_por_ponto:.2f} ha).\n\n"
            "KML por talhao/Pontos_TH_n.kml : um arquivo por talhão; o nome de cada ponto é o seu número.\n"
            "   Identifique as amostras no laboratório com o MESMO número (AMOSTRA 01, 02…): a plataforma liga\n"
            "   laudo e ponto pelo número ao gerar os mapas.\n"
            "Pontos_todos.kml / .gpx : todos os pontos (Google Earth, Avenza, GPS de mão, celular).\n"
            "Shapefile/PONTOS.* : pontos em WGS84 para o software do GPS/monitor.\n"
            + ("Zonas_de_manejo.kml : polígonos das zonas.\n" if plano.zonas else "")
            + "Coordenadas.xlsx : tabela de coordenadas (graus decimais, WGS84) e resumo por talhão.\n").encode("utf-8"))
    return buf.getvalue()
