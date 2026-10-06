"""Mapas de atributos e de prescrição por talhão (pixel → zonas → shapefile)."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shapefile  # pyshp
from scipy import ndimage
from shapely import box, union_all
from shapely.geometry import MultiPolygon, Polygon, mapping

from . import regras
from .geo import Projeto, Talhao, projetar
from .interpretacao import completar_derivados
from .krigagem import Ajuste, Grade, grade_do_poligono, interpolar_grade
from .parametros import Ajustes, Parametros

ATRIB_MAPA = ["ph", "mo", "ctc", "v", "hal", "al", "m", "ca", "sat_ca", "mg", "sat_mg", "k", "sat_k",
              "p", "s", "b", "zn", "mn", "cu", "fe", "argila", "silte", "areia"]
BASE_DOSE = ["ca", "mg", "hal", "al", "k", "p", "s", "argila"]
WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')


@dataclass
class MapaTalhao:
    talhao: Talhao
    grade: Grade
    atributos: dict[str, np.ndarray] = field(default_factory=dict)
    ajustes: dict[str, Ajuste] = field(default_factory=dict)
    doses: dict[str, np.ndarray] = field(default_factory=dict)      # produto → pixels (kg/ha)
    zonas: dict[str, "Zonas"] = field(default_factory=dict)
    # superfícies usadas só no cálculo das doses quando diferem das dos mapas (mapas "fiéis", doses "estatísticas")
    base: dict[str, np.ndarray] = field(default_factory=dict)
    ajustes_base: dict[str, Ajuste] = field(default_factory=dict)

    def para_doses(self) -> dict[str, np.ndarray]:
        return {**self.atributos, **self.base} if self.base else self.atributos

    def ajustes_doses(self) -> dict[str, Ajuste]:
        return {**self.ajustes, **self.ajustes_base} if self.ajustes_base else self.ajustes


@dataclass
class Zonas:
    produto: str
    niveis: list[float]
    poligonos: list[tuple[float, Polygon | MultiPolygon]]    # (dose, geometria UTM)
    classes: np.ndarray                                     # raster de classes (−1 fora)

    def tabela(self) -> pd.DataFrame:
        linhas = [{"Dose (kg/ha)": d, "Área (ha)": g.area / 1e4} for d, g in self.poligonos]
        t = pd.DataFrame(linhas).groupby("Dose (kg/ha)", as_index=False).sum()
        t["Produto (kg)"] = t["Dose (kg/ha)"] * t["Área (ha)"]
        return t

    @property
    def area_ha(self) -> float:
        return sum(g.area for _, g in self.poligonos) / 1e4

    @property
    def total_kg(self) -> float:
        return sum(d * g.area / 1e4 for d, g in self.poligonos)

    @property
    def dose_media(self) -> float:
        return self.total_kg / self.area_ha if self.area_ha else float("nan")


# ------------------------------------------------------------- superfícies
def interpolar_talhao(m: MapaTalhao, a: pd.DataFrame, atributos, modo: str = "estatistico",
                      modo_doses: str | None = None, so_se=None, **opcoes) -> None:
    """Preenche m.atributos (mapas) e, quando o modo das doses difere do dos mapas, m.base (só BASE_DOSE).

    No modo "fiel" o mapa nunca é trocado pelo valor médio: a opção `sem_estrutura="uniforme"` vale só para as
    superfícies das doses."""
    modo_doses = modo_doses or modo
    op_mapa = {**opcoes, "sem_estrutura": "exploratorio"} if modo == "fiel" else opcoes
    for k in dict.fromkeys(list(atributos) + BASE_DOSE):
        if k in a and a[k].notna().sum() > 0 and (so_se is None or k in so_se):
            m.atributos[k], m.ajustes[k] = interpolar_grade(m.grade, a["x"], a["y"], a[k], modo=modo, **op_mapa)
            if modo_doses != modo and k in BASE_DOSE:
                m.base[k], m.ajustes_base[k] = interpolar_grade(m.grade, a["x"], a["y"], a[k], modo=modo_doses,
                                                                **opcoes)


def superficies(projeto: Projeto, res: float = 10.0, atributos=None, modo: str = "estatistico",
                modo_doses: str | None = None, **opcoes) -> list[MapaTalhao]:
    """Krigagem de cada atributo em cada talhão.

    `modo`: interpolação dos MAPAS ("fiel" segue as amostras; "estatistico" usa o semivariograma ajustado).
    `modo_doses`: interpolação usada no cálculo das doses (padrão: a mesma dos mapas).
    `opcoes`: limitar_extremos, sem_estrutura (ver krigagem)."""
    atributos = atributos or ATRIB_MAPA
    mapas = []
    for t in projeto.talhoes:
        m = MapaTalhao(t, grade_do_poligono(t.perimetro, res))
        interpolar_talhao(m, completar_derivados(t.amostras), atributos, modo, modo_doses, **opcoes)
        mapas.append(m)
    return mapas


# ------------------------------------------------------------------ doses
def _vetor(fun, *arrs, **kw):
    out = np.full(arrs[0].shape, np.nan)
    ok = np.all([np.isfinite(a) for a in arrs], axis=0)
    it = zip(*[a[ok] for a in arrs])
    out[ok] = [(np.nan if (d := fun(*v, **kw).dose) is None else d) for v in it]   # 0 = não aplicar (≠ sem dado)
    return out


REQUISITOS = {"Calcário": ("ca", "mg", "hal"), "Gesso": ("s", "argila"), "P2O5": ("p",), "KCl": ("k",)}


def calcular_doses(mapas: list[MapaTalhao], par: Parametros, abertura: dict[str, bool],
                   aj: Ajustes, avisos: list[str] | None = None) -> dict[str, float]:
    """Doses por pixel com as mesmas regras e ajustes da planilha. Devolve os fatores aplicados.

    Cada produto tem seus próprios requisitos (REQUISITOS): se falta um atributo num talhão, só aquele produto
    fica de fora naquele talhão — os demais seguem. As ausências vão para `avisos`."""
    avisos = avisos if avisos is not None else []
    for m in mapas:
        A = m.para_doses()
        ab = bool(abertura.get(m.talhao.nome, abertura.get(m.talhao.chave, False)))
        m.doses = {}
        for prod, req in REQUISITOS.items():
            falta = [k for k in req if k not in A]
            if falta:
                avisos.append(f"{m.talhao.nome}: sem {', '.join(falta)} — prescrição de {prod} não gerada")
                continue
            if prod == "P2O5" and aj.bloquear_p:
                continue
            zero = np.zeros_like(A[req[0]])
            if prod == "Calcário":
                m.doses[prod] = _vetor(lambda ca, mg, hal, al, k: regras.calcario(ca, mg, hal, al, k, ab, par.calcario),
                                       A["ca"], A["mg"], A["hal"], A.get("al", zero), A.get("k", zero))
            elif prod == "Gesso":
                m.doses[prod] = _vetor(lambda s_, arg: regras.gesso(s_, arg, par.gesso), A["s"], A["argila"])
            elif prod == "P2O5":
                m.doses[prod] = _vetor(lambda p_: regras.p2o5(p_, par.fosforo), A["p"])
            else:
                m.doses[prod] = _vetor(lambda k: regras.kcl(k, par.potassio), A["k"])
        if aj.gesso_por_s_elementar and "Gesso" in m.doses:
            g = m.doses.pop("Gesso")
            m.doses["S elementar"] = np.where(np.isfinite(g), par.enxofre.coef * np.power(g, par.enxofre.expoente),
                                              np.nan)

    fatores = {}

    def escalar(nome, a):
        com = [m for m in mapas if nome in m.doses]
        if not com:
            return
        todos = np.concatenate([m.doses[nome][m.grade.mascara] for m in com])
        media = np.nanmean(todos) if np.isfinite(todos).any() else np.nan   # pixels de mesma área
        f = (a.media_alvo / media) if (a.media_alvo is not None and media > 0) else (1 + a.pct / 100)
        fatores[nome] = f
        for m in com:
            m.doses[nome] = m.doses[nome] * f

    for nome, a in (("Calcário", aj.calcario), ("S elementar" if aj.gesso_por_s_elementar else "Gesso", aj.gesso),
                    ("P2O5", aj.p2o5)):
        escalar(nome, a)

    # formulação de P: dose do produto; o K2O que ela traz é descontado do KCl (mesma regra da planilha)
    form = regras.ler_formula(aj.formula_p)
    for m in mapas:
        if form and form[1] > 0 and "P2O5" in m.doses:
            prod = m.doses["P2O5"] / (form[1] / 100)
            m.doses[nome_produto(form)] = prod
            if form[2] > 0 and "KCl" in m.doses:
                equiv = regras.kcl_equivalente(prod, form[2], par.potassio.k2o_kcl)
                k = m.doses["KCl"]
                m.doses["KCl"] = np.where(np.isfinite(k), np.clip(k - np.nan_to_num(equiv), 0, None), np.nan)
    escalar("KCl", aj.kcl)
    # correção de P (além da manutenção): dose média informada; variável = mesmo formato da curva de manutenção
    alvo_pc = aj.correcao_p2o5()
    if alvo_pc > 0:
        variavel = aj.p_correcao_variavel and not aj.bloquear_p
        com = [m for m in mapas if "p" in m.para_doses()]
        for m in com:
            P = m.para_doses()["p"]
            with np.errstate(all="ignore"):
                w = np.where(np.isfinite(P) & (P > 0), par.fosforo.coef * np.power(np.maximum(P, 1e-9),
                                                                                    par.fosforo.expoente), np.nan)
            m.doses[P_CORRECAO] = w if variavel else np.where(np.isfinite(P), 1.0, np.nan)
        if com:
            todos = np.concatenate([m.doses[P_CORRECAO][m.grade.mascara] for m in com])
            media = np.nanmean(todos) if np.isfinite(todos).any() else np.nan
            form_c = regras.ler_formula(aj.p_correcao_formula)
            for m in com:
                d = m.doses.pop(P_CORRECAO) * (alvo_pc / media if media and media > 0 else np.nan)
                if form_c and form_c[1] > 0:                # com formulação, a prescrição sai em dose do produto
                    m.doses[nome_correcao(form_c)] = d / (form_c[1] / 100)
                else:
                    m.doses[P_CORRECAO] = d
            fatores[P_CORRECAO] = alvo_pc / media if media and media > 0 else float("nan")
    for m in mapas:
        if aj.kcl_parcelado and "KCl" in m.doses:
            m.doses["KCl 1ª aplicação"] = m.doses["KCl"] * aj.kcl_pct_1 / 100
            m.doses["KCl 2ª aplicação"] = m.doses["KCl"] * (100 - aj.kcl_pct_1) / 100
    return fatores


P_CORRECAO = "P2O5 correção"


def nome_correcao(form) -> str:
    return f"Correção {int(form[0]):02d}-{int(form[1]):02d}-{int(form[2]):02d}"


FAMILIAS = ["Calagem", "Gessagem", "Fósforo", "Correção de P", "Potássio"]


def familia(produto: str) -> str:
    """Grupo de prescrição de um produto (para escolher o que entra no book e nos shapefiles)."""
    if produto == "Calcário":
        return "Calagem"
    if produto in ("Gesso", "S elementar"):
        return "Gessagem"
    if produto == P_CORRECAO or produto.startswith("Correção"):
        return "Correção de P"
    if produto == "P2O5" or produto.startswith("Produto"):
        return "Fósforo"
    if produto.startswith("KCl"):
        return "Potássio"
    return produto


def filtrar_prescricoes(mapas: list[MapaTalhao], familias) -> list[str]:
    """Mantém em m.doses só os produtos das famílias escolhidas (antes de gerar as zonas). Os mapas de
    fertilidade não mudam. Devolve as famílias retiradas que tinham dose calculada."""
    if familias is None:
        return []
    manter = set(familias)
    fora = set()
    for m in mapas:
        for prod in list(m.doses):
            if familia(prod) not in manter:
                fora.add(familia(prod))
                del m.doses[prod]
    return [f for f in FAMILIAS if f in fora]


def nome_produto(form) -> str:
    return f"Produto {int(form[0]):02d}-{int(form[1]):02d}-{int(form[2]):02d}"


def alvos_de(aj: Ajustes) -> dict[str, float]:
    """Médias-alvo (volume já comprado) por produto de prescrição, para recalibrar as zonas."""
    alvos = {}
    if aj.calcario.media_alvo is not None:
        alvos["Calcário"] = aj.calcario.media_alvo
    if aj.gesso.media_alvo is not None:
        alvos["S elementar" if aj.gesso_por_s_elementar else "Gesso"] = aj.gesso.media_alvo
    if aj.p2o5.media_alvo is not None:
        alvos["P2O5"] = aj.p2o5.media_alvo
        form = regras.ler_formula(aj.formula_p)
        if form and form[1] > 0:
            alvos[nome_produto(form)] = aj.p2o5.media_alvo / (form[1] / 100)
    if aj.kcl.media_alvo is not None:
        alvos["KCl"] = aj.kcl.media_alvo
    pc = aj.correcao_p2o5()                           # a média das zonas de correção fecha com a dose informada
    if pc > 0:
        form_c = regras.ler_formula(aj.p_correcao_formula)
        if form_c and form_c[1] > 0:
            alvos[nome_correcao(form_c)] = pc / (form_c[1] / 100)
        else:
            alvos[P_CORRECAO] = pc
    return alvos


def passo_arredondamento(produto: str, par: Parametros) -> int:
    a = par.arredondamento
    for prefixo, passo in (("Calcário", a.calcario), ("Gesso", a.gesso), ("S elementar", a.s_elementar),
                           ("P2O5", a.p2o5), ("Produto", a.produto_p), ("Correção", a.produto_p), ("KCl", a.kcl)):
        if produto.startswith(prefixo):
            return passo
    return 1


# ------------------------------------------------------------------ zonas
def _suavizar_nan(D, mascara, sigma):
    V = np.where(mascara, np.nan_to_num(D), 0.0)
    W = mascara.astype(float)
    num = ndimage.gaussian_filter(V, sigma)
    den = ndimage.gaussian_filter(W, sigma)
    out = np.where(mascara & (den > 1e-6), num / np.maximum(den, 1e-6), np.nan)
    return out


def _peneira(C, mascara, min_px):
    """Remove manchas pequenas: cada mancha < min_px assume a classe vizinha mais comum."""
    C = C.copy()
    for _ in range(3):
        mudou = False
        for k in np.unique(C[mascara]):
            lab, n = ndimage.label((C == k) & mascara)
            if n == 0:
                continue
            tam = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            for i in np.where(tam < min_px)[0] + 1:
                reg = lab == i
                borda = ndimage.binary_dilation(reg) & ~reg & mascara
                viz = C[borda]
                if viz.size:
                    C[reg] = np.bincount(viz).argmax()
                    mudou = True
        if not mudou:
            break
    return C


def _preencher_fora(D, mascara):
    """Estende os valores para fora do talhão (vizinho mais próximo) para o contorno não 'grudar' na borda."""
    idx = ndimage.distance_transform_edt(~mascara, return_distances=False, return_indices=True)
    return D[tuple(idx)]


def _regiao_acima(g: Grade, campo, limiar):
    """Polígono (UTM) da região onde campo ≥ limiar, com bordas suaves (curva de contorno)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon as P
    r = g.res
    # moldura com valor baixo para fechar as curvas na borda da grade
    pad = np.pad(campo, 1, constant_values=np.nanmin(campo) - 1)
    xs = np.concatenate([[g.x[0] - r], g.x, [g.x[-1] + r]])
    ys = np.concatenate([[g.y[0] + r], g.y, [g.y[-1] - r]])
    fig, ax = plt.subplots()
    cs = ax.contourf(xs, ys, pad, levels=[limiar, np.inf])
    reg = Polygon()
    for path in cs.get_paths():
        for anel in path.to_polygons(closed_only=True):
            if len(anel) >= 4:
                reg = reg.symmetric_difference(P(anel).buffer(0))   # regra par-ímpar (trata furos)
    plt.close(fig)
    return reg.buffer(0)


def _dose_suave(m: MapaTalhao, produto: str, suavizacao_m: float = 25.0):
    g, D = m.grade, m.doses[produto]
    mask = g.mascara & np.isfinite(D)
    return _suavizar_nan(D, mask, sigma=max(suavizacao_m / g.res, 1.0)), mask


def niveis_de_dose(lo: float, hi: float, n_zonas: int, passo: float) -> list[float]:
    """n_zonas doses igualmente espaçadas entre lo e hi, arredondadas (refina o passo se a faixa é estreita)."""
    niveis = []
    # passo fracionário (sementes/m): as doses ficam sempre no degrau pedido — não se refina além dele
    for p in ((passo, 1, 0.5, 0.1) if passo >= 1 else (passo,)):
        niveis = sorted(set(regras.arredondar(v, p) if p >= 1 else round(round(v / p) * p, 4)
                            for v in np.linspace(lo, hi, n_zonas)))
        if len(niveis) >= n_zonas:
            break
    return niveis


def gerar_zonas(m: MapaTalhao, produto: str, n_zonas: int = 6, passo: int = 1,
                area_min_ha: float = 0.5, largura_min_m: float = 30.0, suavizacao_m: float = 25.0,
                niveis: list[float] | None = None) -> Zonas:
    """Zonas de manejo aplicáveis: suavização de ~25 m, faixas com pelo menos `largura_min_m`
    (largura de aplicação) e áreas de pelo menos `area_min_ha`.

    `niveis`: doses já definidas (blocos de talhões usam as mesmas doses em todos os talhões)."""
    g, D = m.grade, m.doses[produto]
    Ds, mask = _dose_suave(m, produto, suavizacao_m)
    if niveis is None:
        niveis = niveis_de_dose(np.nanmin(Ds[mask]), np.nanmax(Ds[mask]), n_zonas, passo)
    niveis = list(niveis)
    niv = np.array(niveis, float)
    C = np.full(D.shape, -1, int)
    C[mask] = np.abs(Ds[mask][:, None] - niv[None, :]).argmin(axis=1)
    min_px = max(1, int(area_min_ha / g.area_pixel_ha))
    C[mask] = _peneira(np.where(mask, C, 0), mask, min_px)[mask]

    # regiões encaixadas R_k = {dose suavizada ≥ limiar_k} traçadas por curvas de contorno;
    # zona k = R_k − R_{k+1} (partição exata do perímetro, sem sobreposição nem buraco)
    per = m.talhao.perimetro
    campo = _preencher_fora(Ds, mask)
    limiares = [(niv[k - 1] + niv[k]) / 2 for k in range(1, len(niv))]
    regioes = [per]
    r_ab = largura_min_m / 2
    for t in limiares:
        reg = _regiao_acima(g, campo, t)
        if r_ab > 0 and not reg.is_empty:          # abertura morfológica: remove faixas/anéis estreitos
            reg = reg.buffer(-r_ab, join_style="round").buffer(r_ab, join_style="round")
        reg = reg.intersection(per)
        regioes.append(reg.intersection(regioes[-1]).buffer(0))
    polys = []
    for k, dose in enumerate(niv):
        z = regioes[k].difference(regioes[k + 1]) if k + 1 < len(regioes) else regioes[k]
        if not z.is_empty:
            partes = list(z.geoms) if hasattr(z, "geoms") else [z]
            for p in partes:
                if isinstance(p, Polygon) and p.area > 0.01:
                    polys.append((float(dose), p))
    return Zonas(produto, niveis, _fundir_pequenos(polys, area_min_ha * 1e4), C)


def _fundir_pequenos(polys, area_min):
    """Funde cada polígono menor que area_min ao vizinho com maior divisa comum.

    Polígonos pequenos sem vizinho (ilhas isoladas) são mantidos e não voltam a ser escolhidos — o laço termina
    quando não há mais o que fundir."""
    polys = [[d, p] for d, p in polys]
    isolados: set[int] = set()
    while len(polys) >= 2:
        cand = [i for i, (_, p) in enumerate(polys) if p.area < area_min and id(p) not in isolados]
        if not cand:
            break
        alvo = min(cand, key=lambda i: polys[i][1].area)
        d0, p0 = polys.pop(alvo)
        melhor, comp = None, 0.0
        for j, (_, q) in enumerate(polys):
            if p0.distance(q) < 0.5:
                c = p0.buffer(0.5).intersection(q.boundary).length
                if c > comp:
                    melhor, comp = j, c
        if melhor is None:            # isolado (não encosta em ninguém): mantém e não tenta de novo
            isolados.add(id(p0))
            polys.append([d0, p0])
            continue
        polys[melhor][1] = polys[melhor][1].union(p0).buffer(0)     # mantém todas as partes (sem perda de área)
    # junta polígonos de mesma dose que se tocam
    saida = []
    for dose in sorted({d for d, _ in polys}):
        u = union_all([p for d, p in polys if d == dose])
        for p in (u.geoms if hasattr(u, "geoms") else [u]):
            if isinstance(p, Polygon) and not p.is_empty:
                saida.append((dose, p))
    return saida


@dataclass
class Bloco:
    """Bloco de aplicação: um ou mais talhões com prescrição e shapefile únicos por produto."""
    nome: str
    mapas: list[MapaTalhao]
    zonas: dict[str, Zonas] = field(default_factory=dict)

    @property
    def unico(self) -> bool:
        return len(self.mapas) == 1

    @property
    def perimetro(self):
        return union_all([m.talhao.perimetro for m in self.mapas])

    @property
    def talhoes(self) -> list[str]:
        return [m.talhao.nome for m in self.mapas]


def _juntar_zonas(produto: str, niveis, lista: list[Zonas]) -> Zonas:
    """Zonas de vários talhões num só conjunto: um registro (multipolígono) por dose."""
    polys = []
    for dose in sorted({d for z in lista for d, _ in z.poligonos}):
        u = union_all([g for z in lista for d, g in z.poligonos if d == dose])
        if not u.is_empty:
            polys.append((dose, u))
    return Zonas(produto, list(niveis), polys, None)


PARCELAS = ("KCl 1ª aplicação", "KCl 2ª aplicação")


def _mapear_doses(z: Zonas, novo: dict[float, float]) -> Zonas:
    return Zonas(z.produto, sorted({novo.get(float(d), d) for d in z.niveis}),
                 [(novo.get(float(d), d), g) for d, g in z.poligonos], z.classes)


def _reescalar(blocos: list["Bloco"], prod: str, alvo: float, passo: int) -> None:
    """Recalibra as doses das zonas de `prod` (todos os blocos) para que a média ponderada pela área fique na
    média-alvo (volume comprado), mantendo o arredondamento operacional. A suavização, a classificação em
    zonas e a fusão de manchas pequenas mudam a média dos pixels — sem isso o volume não fecha."""
    zs = [b.zonas[prod] for b in blocos if prod in b.zonas]
    if not zs or alvo is None:
        return
    area = sum(z.area_ha for z in zs)
    if area <= 0:
        return
    originais = sorted({float(d) for z in zs for d, _ in z.poligonos})
    F = 1.0
    melhor = (np.inf, {d: d for d in originais})
    for _ in range(12):
        novo = {d: (regras.arredondar(d * F, passo) if passo >= 1 else round(d * F, 1)) for d in originais}
        tot = sum(novo[float(d)] * g.area / 1e4 for z in zs for d, g in z.poligonos)
        media = tot / area
        erro = abs(media - alvo)
        if erro < melhor[0]:
            melhor = (erro, novo)
        if erro <= 0.002 * alvo or media <= 0:
            break
        F *= alvo / media
    novo = melhor[1]
    for b in blocos:
        if prod in b.zonas:
            b.zonas[prod] = _mapear_doses(b.zonas[prod], novo)
        for m in b.mapas:
            if prod in m.zonas and m.zonas[prod] is not b.zonas.get(prod):
                m.zonas[prod] = _mapear_doses(m.zonas[prod], novo)
        if b.unico and prod in b.zonas:
            b.mapas[0].zonas[prod] = b.zonas[prod]


def _parcelas(z: Zonas, pct1: float, passo: int) -> tuple[Zonas, Zonas]:
    """1ª e 2ª parcelas do KCl na MESMA partição da dose total: 1ª = total × pct arredondada; 2ª = total − 1ª.
    A soma fecha em cada polígono."""
    d1 = {float(d): (regras.arredondar(d * pct1 / 100, passo) if passo >= 1 else round(d * pct1 / 100, 1))
          for d in z.niveis}
    d1.update({float(d): (regras.arredondar(d * pct1 / 100, passo) if passo >= 1 else round(d * pct1 / 100, 1))
               for d, _ in z.poligonos})
    z1 = Zonas(PARCELAS[0], sorted(set(d1.values())), [(d1[float(d)], g) for d, g in z.poligonos], z.classes)
    z2 = Zonas(PARCELAS[1], sorted({float(d) - d1[float(d)] for d in d1}),
               [(float(d) - d1[float(d)], g) for d, g in z.poligonos], z.classes)
    return z1, z2


def gerar_todas_zonas(mapas: list[MapaTalhao], par: Parametros, n_zonas: int = 6,
                      blocos: dict[str, str] | None = None, aj: Ajustes | None = None) -> list[Bloco]:
    """Zonas de todos os talhões. `blocos`: talhão → nome do bloco (talhões com o mesmo nome são
    prescritos juntos: mesmas doses e um shapefile por produto). Talhão sem bloco = separado.

    Com `aj`: (1) as doses das zonas são recalibradas para a média-alvo (volume comprado) de cada produto;
    (2) as parcelas do KCl saem da partição da dose total (a soma fecha polígono a polígono).
    Devolve a lista de blocos (um por talhão separado)."""
    blocos = {k: v.strip() for k, v in (blocos or {}).items() if v and str(v).strip()}
    grupos: dict[str, list[MapaTalhao]] = {}
    for m in mapas:
        chave = blocos.get(m.talhao.nome) or blocos.get(m.talhao.chave)
        grupos.setdefault(chave or f"\0{m.talhao.nome}", []).append(m)
    saida = []
    for chave, ms in grupos.items():
        for m in ms:
            m.zonas = {}
        prods = [p for p in dict.fromkeys(p for m in ms for p in m.doses) if p not in PARCELAS]
        if len(ms) == 1:
            m = ms[0]
            for prod in prods:
                m.zonas[prod] = gerar_zonas(m, prod, n_zonas, passo_arredondamento(prod, par))
            saida.append(Bloco(m.talhao.nome, ms, m.zonas))
            continue
        b = Bloco(chave, ms)
        for prod in prods:
            com = [m for m in ms if prod in m.doses]
            suaves = [_dose_suave(m, prod) for m in com]
            vals = np.concatenate([Ds[mk] for Ds, mk in suaves])
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            niv = niveis_de_dose(vals.min(), vals.max(), n_zonas, passo_arredondamento(prod, par))
            for m in com:
                m.zonas[prod] = gerar_zonas(m, prod, n_zonas, niveis=niv)
            b.zonas[prod] = _juntar_zonas(prod, niv, [m.zonas[prod] for m in com])
        saida.append(b)
    if aj is not None:
        for prod, alvo in alvos_de(aj).items():
            _reescalar(saida, prod, alvo, passo_arredondamento(prod, par))
    pct1 = aj.kcl_pct_1 if aj is not None else None
    if pct1 is None and any(PARCELAS[0] in m.doses for m in mapas):
        r = [np.nanmedian(m.doses[PARCELAS[0]] / m.doses["KCl"]) for m in mapas
             if PARCELAS[0] in m.doses and np.isfinite(m.doses["KCl"]).any()]
        pct1 = 100 * float(np.nanmedian(r)) if r else 50.0
    if any(PARCELAS[0] in m.doses for m in mapas):
        passo = passo_arredondamento("KCl", par)
        for b in saida:
            if "KCl" in b.zonas:
                b.zonas[PARCELAS[0]], b.zonas[PARCELAS[1]] = _parcelas(b.zonas["KCl"], pct1, passo)
            for m in b.mapas:
                if "KCl" in m.zonas:
                    if b.unico:
                        m.zonas[PARCELAS[0]], m.zonas[PARCELAS[1]] = b.zonas[PARCELAS[0]], b.zonas[PARCELAS[1]]
                    else:
                        m.zonas[PARCELAS[0]], m.zonas[PARCELAS[1]] = _parcelas(m.zonas["KCl"], pct1, passo)
    return saida


# -------------------------------------------------------------- shapefiles
def _nome_arquivo(txt: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() else "_" for c in t).strip("_").replace("__", "_")


def shapefile_bytes(zonas: Zonas, epsg: int, nome: str, campo: str = "Taxa_Dest_", casas: int = 4
                    ) -> dict[str, bytes]:
    """Shapefile (WGS84, campo Taxa_Dest_) no mesmo formato do arquivo de referência."""
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON)
    w.field(campo, "N", size=15, decimal=casas)
    for dose, geom in sorted(zonas.poligonos, key=lambda t: -t[0]):
        ll = projetar(geom, epsg, inverso=True)
        partes = []
        for p in (ll.geoms if hasattr(ll, "geoms") else [ll]):
            ext = list(p.exterior.coords)
            if Polygon(ext).exterior.is_ccw:
                ext = ext[::-1]                         # anel externo horário (padrão shapefile)
            partes.append(ext)
            for furo in p.interiors:
                f = list(furo.coords)
                partes.append(f if Polygon(f).exterior.is_ccw else f[::-1])
        w.poly(partes)
        w.record(round(dose, casas))
    w.close()
    return {f"{nome}.shp": shp.getvalue(), f"{nome}.shx": shx.getvalue(),
            f"{nome}.dbf": dbf.getvalue(), f"{nome}.prj": WGS84_PRJ.encode()}


MAX_NOME = 9                     # nomes de shapefile com menos de 10 caracteres (limite de vários monitores)
COD_PRODUTO = {"Calcário": "CAL", "Gesso": "GES", "S elementar": "SEL", "P2O5": "P2O5", "KCl": "KCL",
               "KCl 1ª aplicação": "KC1", "KCl 2ª aplicação": "KC2", "Sementes": "SEM"}
_RUIDO = {"PERIMETRO", "PERIMETROS", "LIMITE", "DE", "DA", "DO", "DAS", "DOS", "E"}
_TIPO = {"TALHAO", "TALHOES", "BLOCO", "GLEBA", "AREA", "FAZENDA", "FAZ", "LOTE", "SETOR"}
PASTAS_MONITOR = [
    ("John Deere", "Rx/", "Rx/CAL_T1.shp"),
    ("Stara", "Dados/Mapas/", "Dados/Mapas/CAL_T1.shp"),
    ("Ag Leader e Trimble", "AgGPS/Prescriptions/", "AgGPS/Prescriptions/CAL_T1.shp"),
    ("Raven", "rxMaps/", "rxMaps/CAL_T1.shp"),
]


def codigo_produto(produto: str) -> str:
    if produto in COD_PRODUTO:
        return COD_PRODUTO[produto]
    if produto.startswith("Produto"):
        return "FOS"                                  # formulação fosfatada
    if produto == P_CORRECAO or produto.startswith("Correção"):
        return "PCOR"                                 # correção de P
    return "".join(c for c in _nome_arquivo(produto).upper() if c.isalnum())[:3] or "PRD"


def codigo_talhao(nome: str, maximo: int = 5) -> str:
    """Código curto do talhão/bloco: 'Talhão 4' → T4, 'TH 1' → TH1, 'Perimetro_TH_12' → TH12, 'Bloco A' → BA."""
    import re
    t = _nome_arquivo(str(nome)).upper()
    partes = []
    for x in (x for x in re.split(r"[^A-Z0-9]+", t) if x):
        # "TALHAO4" → TALHAO + 4 (mas "TH01" fica como está)
        partes += re.findall(r"[A-Z]+|[0-9]+", x) if re.match(r"^[A-Z]{4,}[0-9]+$", x) else [x]
    partes = [x for x in partes if x not in _RUIDO] or partes
    prefixo = partes[0][0] if partes and partes[0] in _TIPO else ""
    resto = [x for x in partes if x not in _TIPO]
    if not resto:
        return (prefixo or "T")[:maximo]
    cod = prefixo + "".join(resto)
    if len(cod) > maximo:
        cod = prefixo + "".join(x if x.isdigit() else x[0] for x in resto)
    return cod[:maximo]


def nomes_curtos(rotulos: list[str], maximo: int = 5) -> dict[str, str]:
    """Códigos únicos para uma lista de talhões/blocos."""
    usados, out = set(), {}
    for r in rotulos:
        c = codigo_talhao(r, maximo)
        base, k = c, 1
        while c in usados:                       # repetido: acrescenta letra (TH1 → TH1B), nunca número
            suf = "BCDEFGHJKLMNPQRSTUVWXYZ"[(k - 1) % 23]
            c = base[:maximo - 1] + suf
            k += 1
        usados.add(c)
        out[r] = c
    return out


def nome_shapefile(produto: str, cod_talhao: str) -> str:
    p = codigo_produto(produto)
    return f"{p}_{cod_talhao[:MAX_NOME - len(p) - 1]}"


def _leia_me(linhas_arquivos: list[str]) -> str:
    txt = ["PRESCRIÇÕES - SHAPEFILES (WGS84, campo de dose: Taxa_Dest_)", "",
           "Uma pasta por talhão ou bloco de aplicação. Nomes curtos (até 9 caracteres), sem espaços:",
           "  CAL = calcário   GES = gesso   SEL = enxofre elementar   FOS = formulação fosfatada",
           "  KCL = cloreto de potássio   KC1/KC2 = 1ª/2ª aplicação   SEM = sementes   P2O5 = fósforo puro", "",
           "Onde colocar no pen drive, conforme o monitor:"]
    for marca, pasta, ex in PASTAS_MONITOR:
        txt.append(f"  {marca:<22}{pasta:<24}(ex.: {ex})")
    txt += ["", "Copie os 4 arquivos de cada mapa (.shp, .shx, .dbf, .prj) juntos.", "", "Arquivos:"] + \
        [f"  {l}" for l in linhas_arquivos]
    return "\r\n".join(txt)


def ler_shapefile_bytes(arquivos: dict[str, bytes], epsg: int) -> list[tuple[float, object]]:
    """Relê um shapefile exportado: [(dose, geometria UTM)]."""
    import logging

    from shapely.geometry import shape
    logging.getLogger("shapefile").setLevel(logging.ERROR)    # avisos de orientação de anéis quase degenerados
    base = next(k[:-4] for k in arquivos if k.endswith(".shp"))
    r = shapefile.Reader(shp=io.BytesIO(arquivos[base + ".shp"]), shx=io.BytesIO(arquivos[base + ".shx"]),
                         dbf=io.BytesIO(arquivos[base + ".dbf"]))
    campos = [f[0] for f in r.fields[1:]]
    i = campos.index("Taxa_Dest_") if "Taxa_Dest_" in campos else 0
    return [(float(rec[i]), projetar(shape(sh.__geo_interface__).buffer(0), epsg)) for sh, rec in
            zip(r.shapes(), r.records())]


def conferir(itens, epsg: int, produtos: list[str] | None = None) -> pd.DataFrame:
    """Conferência final de cada prescrição, relendo o shapefile exportado: área coberta, sobreposições,
    lacunas, doses válidas, volume total (zonas × shapefile) e fechamento das parcelas do KCl."""
    linhas = []
    for it_ in itens:
        rot = it_.talhao.nome if isinstance(it_, MapaTalhao) else it_.nome
        per = it_.talhao.perimetro if isinstance(it_, MapaTalhao) else it_.perimetro
        zonas = it_.zonas
        for prod, zon in zonas.items():
            if produtos is not None and prod not in produtos:
                continue
            geoms = [g for _, g in zon.poligonos]
            doses = np.array([d for d, _ in zon.poligonos], float)
            uniao = union_all(geoms) if geoms else Polygon()
            soma = sum(g.area for g in geoms)
            cobertura = 100 * uniao.intersection(per).area / per.area if per.area else 0
            sobrepos = max(soma - uniao.area, 0) / 1e4
            fora = uniao.difference(per.buffer(1.0)).area / 1e4
            lido = ler_shapefile_bytes(shapefile_bytes(zon, epsg, "conf"), epsg)
            tot_z = zon.total_kg / 1000
            tot_s = sum(d * g.area / 1e4 for d, g in lido) / 1000
            dif = 100 * (tot_s - tot_z) / tot_z if tot_z else 0.0
            valido = bool(len(doses)) and np.isfinite(doses).all() and (doses >= 0).all()
            parcela = ""
            if prod == PARCELAS[0] and PARCELAS[1] in zonas and "KCl" in zonas:
                t12 = (zon.total_kg + zonas[PARCELAS[1]].total_kg) / 1000
                parcela = f"{100 * (t12 - zonas['KCl'].total_kg / 1000) / max(zonas['KCl'].total_kg / 1000, 1e-9):+.2f}%"
            problemas = []
            if cobertura < 99.5:
                problemas.append(f"cobre {cobertura:.1f}% do perímetro")
            if sobrepos > 0.01:
                problemas.append(f"sobreposição de {sobrepos:.2f} ha")
            if fora > 0.01:
                problemas.append(f"{fora:.2f} ha fora do perímetro")
            if not valido:
                problemas.append("dose inválida")
            if abs(dif) > 0.5:
                problemas.append(f"shapefile difere {dif:+.1f}% no volume")
            if parcela and abs(float(parcela.rstrip("%"))) > 0.05:
                problemas.append(f"parcelas somam {parcela} do total")
            linhas.append({"Talhão/bloco": rot, "Produto": prod, "Área (ha)": round(per.area / 1e4, 2),
                           "Cobertura (%)": round(cobertura, 2), "Sobreposição (ha)": round(sobrepos, 3),
                           "Doses": f"{len(set(doses.tolist()))} ({doses.min():g}–{doses.max():g})" if len(doses)
                           else "—", "Total zonas (t)": round(tot_z, 3), "Total shapefile (t)": round(tot_s, 3),
                           "Parcelas 1ª+2ª vs total": parcela, "Situação": "ok" if not problemas else
                           "atenção: " + "; ".join(problemas)})
    return pd.DataFrame(linhas)


def zip_prescricoes(mapas: list[MapaTalhao], epsg: int, fazenda: str = "",
                    produtos: list[str] | None = None) -> bytes:
    """Zip com uma pasta por talhão (ou por bloco) e os shapefiles de cada produto dentro dela, com nomes curtos
    (ex.: CAL_T4.shp) e um LEIA-ME com a estrutura de pastas de cada monitor.

    `mapas`: lista de MapaTalhao ou de Bloco."""
    rotulos = [m.talhao.nome if isinstance(m, MapaTalhao) else m.nome for m in mapas]
    cods = nomes_curtos(rotulos)
    buf = io.BytesIO()
    lista = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for m, rotulo in zip(mapas, rotulos):
            pasta = _nome_arquivo(rotulo) or cods[rotulo]
            for prod, zon in m.zonas.items():
                if produtos is not None and prod not in produtos:
                    continue
                nome = nome_shapefile(prod, cods[rotulo])
                for arq, dados in shapefile_bytes(zon, epsg, nome).items():
                    z.writestr(f"{pasta}/{arq}", dados)
                lista.append(f"{pasta}/{nome}.shp  ->  {rotulo} · {prod}")
        z.writestr("LEIA-ME.txt", _leia_me(lista).encode("utf-8"))
        try:
            conf = conferir(mapas, epsg, produtos)
            z.writestr("CONFERENCIA.txt", ("CONFERÊNCIA DAS PRESCRIÇÕES (shapefiles relidos após a exportação)\r\n\r\n"
                                           + conf.to_string(index=False)).encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            z.writestr("CONFERENCIA.txt", f"Conferência não concluída: {e}".encode("utf-8"))
    return buf.getvalue()


def tabela_volumes(mapas: list[MapaTalhao]) -> pd.DataFrame:
    linhas = []
    for m in mapas:
        for prod, zon in m.zonas.items():
            linhas.append({"Talhão": m.talhao.nome, "Área (ha)": zon.area_ha, "Produto": prod,
                           "Dose média (kg/ha)": zon.dose_media, "Total (t)": zon.total_kg / 1000})
    return pd.DataFrame(linhas)
