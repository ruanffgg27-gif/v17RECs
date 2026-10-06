"""Mapa de semeadura em taxa variável a partir da fertilidade do solo — REGRA DA EQUIPE EM AVALIAÇÃO.

A resposta da população de plantas à fertilidade depende de cultura, cultivar, época, germinação, estabelecimento
e disponibilidade de água; a regra abaixo padroniza o trabalho, mas deve ser validada com faixas experimentais.

Índice de fertilidade (IF) — regra da equipe:
  1. Para cada atributo (P, M.O., argila e soma de bases Ca+Mg+K), o maior valor das amostras vale 100% e
     cada amostra (ou pixel do mapa) vira uma porcentagem desse máximo (regra de três).
  2. IF = (P% × 5 + M.O.% × 20 + argila% × 40 + SB% × 35) / 100. Sem argila medida, a soma de bases recebe
     peso 75: IF = (P% × 5 + M.O.% × 20 + SB% × 75) / 100.

Taxa de sementes — proporção inversa ao IF (onde o solo é mais fértil caem menos sementes):
  taxa = k / IF, limitada a ± `variacao_max` % da média comprada, com k ajustado para que a média ponderada pela
  área dê exatamente a média comprada (sementes/m) — não falta nem sobra semente.
Depois, a taxa é dividida em até `classes` faixas (zonas aplicáveis, como nas prescrições) e as faixas são
recalibradas para manter a média.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .prescricao import MapaTalhao, Zonas, gerar_zonas

PESOS = {"p": 5.0, "mo": 20.0, "argila": 40.0, "sb": 35.0}
PESOS_SEM_ARGILA = {"p": 5.0, "mo": 20.0, "sb": 75.0}
ATRIB_SEMENTES = ["p", "mo", "argila", "ca", "mg", "k"]
NOMES = {"p": "P", "mo": "M.O.", "argila": "Argila", "sb": "SB (Ca+Mg+K)"}


@dataclass
class ConfigSementes:
    media: float = 13.0                  # sementes/m compradas (média da área)
    variacao_max: float = 20.0           # % em torno da média (limite inferior e superior)
    classes: int = 8                     # número de faixas (zonas) do mapa
    espacamento: float = 0.50            # entre linhas (m), para converter em sementes/ha
    unidade_shape: str = "sementes/m"    # "sementes/m" ou "sementes/ha"
    medias_talhao: dict[str, float] = field(default_factory=dict)   # média específica por talhão (opcional)
    passo: float = 0.0                   # degrau entre doses (sementes/m); 0 = automático pela média
    esticar: bool = False                # True: usa toda a faixa ± variacao_max (mais doses distintas)

    def passo_de(self, media: float) -> float:
        """Degrau entre doses. Automático: 0,1 (≥ 8 sem/m, soja), 0,05 (4–8) ou 0,02 (< 4, milho) — com degrau
        fixo de 0,1 uma cultura de 3 sementes/m ficava com só 3 ou 4 doses possíveis."""
        if self.passo and self.passo > 0:
            return float(self.passo)
        return 0.1 if media >= 8 else (0.05 if media >= 4 else 0.02)

    def casas(self, media: float | None = None) -> int:
        return 1 if self.passo_de(self.media if media is None else media) >= 0.1 else 2

    def media_de(self, talhao: str) -> float:
        """Média do talhão (aceita 'Talhão 1', '1', 'TH 01'…: compara também pelo número)."""
        import re
        if str(talhao) in self.medias_talhao:
            return float(self.medias_talhao[str(talhao)])
        num = re.findall(r"\d+", str(talhao))
        for k, v in self.medias_talhao.items():
            if num and re.findall(r"\d+", str(k)) and int(re.findall(r"\d+", str(k))[-1]) == int(num[-1]):
                return float(v)
        return float(self.media)

    def por_ha(self, sem_m: float) -> float:
        return sem_m * 10000.0 / self.espacamento


@dataclass
class ResultadoSementes:
    maximos: dict[str, float]
    amostras: pd.DataFrame               # tabela por amostra (conferência da regra)
    usa_argila: dict[str, bool]          # por talhão
    k: dict[str, float]                  # constante da proporção inversa por talhão
    avisos: list[str] = field(default_factory=list)


def _sb(df: pd.DataFrame) -> pd.Series:
    partes = [df[c] for c in ("ca", "mg", "k") if c in df]
    if len(partes) == 3:
        return partes[0] + partes[1] + partes[2]
    return df["sb"] if "sb" in df else pd.Series(np.nan, index=df.index)


def _argila_medida(df: pd.DataFrame) -> pd.Series:
    """Argila só quando veio do laudo (estimada pela CTC não conta: aí vale o peso 75 na SB)."""
    a = df["argila"] if "argila" in df else pd.Series(np.nan, index=df.index)
    if "argila_estimada" in df:
        a = a.where(~df["argila_estimada"].fillna(False).astype(bool))
    return a


def maximos(dados: pd.DataFrame) -> dict[str, float]:
    d = dados[~dados["subsuperficial"]] if "subsuperficial" in dados else dados
    return {"p": float(d["p"].max()), "mo": float(d["mo"].max()), "argila": float(_argila_medida(d).max()),
            "sb": float(_sb(d).max())}


def indice(p, mo, argila, sb, mx: dict[str, float]):
    """IF (%) para arrays ou séries; argila nan → peso 75 para a SB (por amostra/pixel)."""
    pp = 100 * np.asarray(p, float) / mx["p"]
    mm = 100 * np.asarray(mo, float) / mx["mo"]
    ss = 100 * np.asarray(sb, float) / mx["sb"]
    aa = 100 * np.asarray(argila, float) / mx["argila"] if np.isfinite(mx.get("argila", np.nan)) and \
        mx["argila"] > 0 else np.full(np.shape(pp), np.nan)
    com = (pp * 5 + mm * 20 + aa * 40 + ss * 35) / 100
    sem = (pp * 5 + mm * 20 + ss * 75) / 100
    return np.where(np.isfinite(aa), com, sem)


def tabela_amostras(dados: pd.DataFrame, mx: dict[str, float], cfg: ConfigSementes) -> pd.DataFrame:
    """A conta da equipe amostra por amostra (para conferência no Excel)."""
    d = dados[~dados["subsuperficial"]].copy() if "subsuperficial" in dados else dados.copy()
    arg = _argila_medida(d)
    sb = _sb(d)
    out = pd.DataFrame({
        "Talhão": d["talhao"].values, "Identificação": d.get("identificacao", pd.Series("", index=d.index)).values,
        "P (mg/dm³)": d["p"].values, "M.O. (g/dm³)": d["mo"].values, "Argila (%)": (arg / 10).values,
        "SB Ca+Mg+K (mmolc/dm³)": sb.values,
        "P (% do máx.)": (100 * d["p"] / mx["p"]).values, "M.O. (% do máx.)": (100 * d["mo"] / mx["mo"]).values,
        "Argila (% do máx.)": (100 * arg / mx["argila"]).values if np.isfinite(mx["argila"]) else np.nan,
        "SB (% do máx.)": (100 * sb / mx["sb"]).values})
    out["Peso argila/SB"] = np.where(np.isfinite(out["Argila (% do máx.)"]), "40 / 35", "0 / 75")
    out["Índice de fertilidade (%)"] = indice(d["p"], d["mo"], arg, sb, mx)
    # taxa por amostra (mesma regra, calibrada nas amostras de cada talhão) — só referência
    taxa = np.full(len(out), np.nan)
    for t in out["Talhão"].unique():
        sel = (out["Talhão"] == t).values
        media = cfg.media_de(str(t))
        taxa[sel] = taxa_inversa([out.loc[sel, "Índice de fertilidade (%)"].values], [np.ones(sel.sum(), bool)],
                                 media, cfg.variacao_max)[0][0]
    out["Sementes/m (amostra)"] = taxa
    return out.round(2)


def taxa_inversa(ifs: list[np.ndarray], mascaras: list[np.ndarray], media: float, variacao_max: float,
                 pesos: list[np.ndarray] | None = None) -> tuple[list[np.ndarray], float]:
    """taxa = clip(k / IF, média·(1 − v), média·(1 + v)) com k tal que a média ponderada = `media`.

    Devolve (taxas, k). A média com o corte é monótona em k → bissecção."""
    lo, hi = media * (1 - variacao_max / 100), media * (1 + variacao_max / 100)
    vals = np.concatenate([f[m] for f, m in zip(ifs, mascaras)])
    w = np.concatenate([p[m] for p, m in zip(pesos, mascaras)]) if pesos else np.ones_like(vals)
    ok = np.isfinite(vals) & (vals > 0)
    vals, w = vals[ok], w[ok]
    if not len(vals):
        return [np.full(f.shape, np.nan) for f in ifs], float("nan")

    def media_com(k):
        return float(np.average(np.clip(k / vals, lo, hi), weights=w))
    a, b = lo * vals.min(), hi * vals.max()
    for _ in range(80):
        c = (a + b) / 2
        if media_com(c) < media:
            a = c
        else:
            b = c
    k = (a + b) / 2
    saida = []
    for f, m in zip(ifs, mascaras):
        t = np.full(f.shape, np.nan)
        okm = m & np.isfinite(f) & (f > 0)
        t[okm] = np.clip(k / f[okm], lo, hi)
        saida.append(t)
    return saida, k


def taxa_esticada(IF: np.ndarray, mascara: np.ndarray, media: float, variacao_max: float) -> np.ndarray:
    """Taxa usando TODA a faixa permitida: o lugar mais fértil do talhão (percentil 98 do índice) recebe a média
    − variação máxima, o menos fértil (percentil 2) recebe a média + variação máxima, linear entre os dois; depois
    um deslocamento devolve a média comprada. Continua inversa à fertilidade — só amplia o contraste quando o
    índice varia pouco e a regra proporcional (k ÷ IF) daria doses quase iguais."""
    t = np.full(IF.shape, np.nan)
    ok = mascara & np.isfinite(IF) & (IF > 0)
    if not ok.any():
        return t
    lo, hi = np.percentile(IF[ok], [2, 98])
    if hi - lo < 1e-9:
        t[ok] = media
        return t
    z = np.clip((IF[ok] - lo) / (hi - lo), 0, 1) * 2 - 1            # −1 (menos fértil) … +1 (mais fértil)
    v = variacao_max / 100
    taxa = media * (1 - v * z)
    for _ in range(20):                                              # recentra na média sem sair dos limites
        taxa = np.clip(taxa + (media - taxa.mean()), media * (1 - v), media * (1 + v))
        if abs(taxa.mean() - media) < 1e-6:
            break
    t[ok] = taxa
    return t


def calcular(mapas: list[MapaTalhao], dados: pd.DataFrame, cfg: ConfigSementes) -> ResultadoSementes:
    """Índice de fertilidade e taxa de sementes por pixel (m.atributos['if_sem'], m.doses['Sementes'])."""
    mx = maximos(dados)
    res = ResultadoSementes(mx, tabela_amostras(dados, mx, cfg), {}, {})
    if not np.isfinite(mx["argila"]):
        res.avisos.append("Laudo sem argila medida: soma de bases com peso 75 em todas as amostras.")
    for m in mapas:
        A = m.atributos
        faltam = [c for c in ("p", "mo", "ca", "mg", "k") if c not in A]
        if faltam:
            res.avisos.append(f"{m.talhao.nome}: sem {', '.join(faltam)} — talhão fora do mapa de semeadura.")
            continue
        sb = A["ca"] + A["mg"] + A["k"]
        am = m.talhao.amostras
        usa = "argila" in A and np.isfinite(mx["argila"]) and _argila_medida(am).notna().any()
        res.usa_argila[m.talhao.nome] = bool(usa)
        arg = A["argila"] if usa else np.full(sb.shape, np.nan)
        m.atributos["if_sem"] = np.where(m.grade.mascara, indice(A["p"], A["mo"], arg, sb, mx), np.nan)
        (t,), k = taxa_inversa([m.atributos["if_sem"]], [m.grade.mascara], cfg.media_de(m.talhao.nome),
                               cfg.variacao_max)
        if cfg.esticar:
            t = taxa_esticada(m.atributos["if_sem"], m.grade.mascara, cfg.media_de(m.talhao.nome), cfg.variacao_max)
        m.doses["Sementes"] = t
        res.k[m.talhao.nome] = k
    return res


def zonas(m: MapaTalhao, cfg: ConfigSementes) -> Zonas:
    """Faixas de taxa (até `cfg.classes`), recalibradas para a média comprada, no degrau de dose configurado."""
    alvo = cfg.media_de(m.talhao.nome)
    passo = cfg.passo_de(alvo)
    arred = lambda v: round(round(v / passo) * passo, 4)        # noqa: E731
    z = gerar_zonas(m, "Sementes", n_zonas=cfg.classes, passo=passo, area_min_ha=0.5, largura_min_m=30.0)
    for _ in range(3):
        area = np.array([g.area for _, g in z.poligonos])
        doses = np.array([d for d, _ in z.poligonos])
        desloc = alvo - float(np.average(doses, weights=area))
        if abs(desloc) < passo / 2:
            break
        novo = {d: arred(d + desloc) for d in set(doses)}
        z = Zonas(z.produto, sorted(set(novo.values())), [(novo[d], g) for d, g in z.poligonos], z.classes)
    return z


def tabela_faixas(mapas: list[MapaTalhao], recomendado: dict[str, Zonas] | None = None) -> pd.DataFrame:
    """Uma linha por dose de cada talhão: área, dose recomendada pelo sistema e dose em uso (ajustada à mão ou
    igual à recomendada). É a tabela que o usuário edita."""
    linhas = []
    for m in mapas:
        z = m.zonas.get("Sementes")
        if z is None:
            continue
        rec = (recomendado or {}).get(m.talhao.nome, z)
        area: dict[tuple[float, float], float] = {}
        for (dr, g), (da, _) in zip(rec.poligonos, z.poligonos):
            area[(float(dr), float(da))] = area.get((float(dr), float(da)), 0.0) + g.area / 1e4
        for (dr, da), a in sorted(area.items(), reverse=True):
            linhas.append({"Talhão": m.talhao.nome, "Área (ha)": round(a, 2), "Recomendado (sem/m)": dr,
                           "Ajustado (sem/m)": da})
    return pd.DataFrame(linhas, columns=["Talhão", "Área (ha)", "Recomendado (sem/m)", "Ajustado (sem/m)"])


def ajustar_doses(m: MapaTalhao, recomendado: Zonas, novo: dict[float, float]) -> list[tuple[float, float]]:
    """Troca doses à mão: `novo` = {dose recomendada: dose desejada}. As geometrias não mudam; faixas que ficarem
    com a mesma dose continuam como polígonos separados no shapefile (mesma taxa). Devolve os pares alterados."""
    mapa = {float(d): float(novo.get(float(d), d)) for d in {float(d) for d, _ in recomendado.poligonos}}
    m.zonas["Sementes"] = Zonas(recomendado.produto, sorted(set(mapa.values())),
                                [(mapa[float(d)], g) for d, g in recomendado.poligonos], recomendado.classes)
    return sorted(((d, n) for d, n in mapa.items() if abs(d - n) > 1e-9), reverse=True)


def gerar(mapas: list[MapaTalhao], dados: pd.DataFrame, cfg: ConfigSementes) -> ResultadoSementes:
    res = calcular(mapas, dados, cfg)
    for m in mapas:
        if "Sementes" in m.doses:
            m.zonas = {"Sementes": zonas(m, cfg)}
    return res


def tabela_resumo(mapas: list[MapaTalhao], cfg: ConfigSementes) -> pd.DataFrame:
    linhas = []
    for m in mapas:
        z = m.zonas.get("Sementes")
        if z is None:
            continue
        doses = [d for d, _ in z.poligonos]
        linhas.append({"Talhão": m.talhao.nome, "Área (ha)": round(z.area_ha, 2),
                       "Média comprada (sem/m)": cfg.media_de(m.talhao.nome),
                       "Média do mapa (sem/m)": round(z.dose_media, 2),
                       "Mínima (sem/m)": min(doses), "Máxima (sem/m)": max(doses), "Faixas": len(set(doses)),
                       "Sementes/ha (média)": round(cfg.por_ha(z.dose_media)),
                       "Total (milhões de sementes)": round(cfg.por_ha(z.dose_media) * z.area_ha / 1e6, 2)})
    return pd.DataFrame(linhas)


def zip_sementes(mapas: list[MapaTalhao], epsg: int, cfg: ConfigSementes) -> bytes:
    """Shapefiles (SEM_T4.shp…) na unidade escolhida, com LEIA-ME de pastas por monitor."""
    import io
    import zipfile

    from .prescricao import _leia_me, _nome_arquivo, nome_shapefile, nomes_curtos, shapefile_bytes
    por_ha = cfg.unidade_shape == "sementes/ha"
    cods = nomes_curtos([m.talhao.nome for m in mapas])
    buf, lista = io.BytesIO(), []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in mapas:
            z = m.zonas.get("Sementes")
            if z is None:
                continue
            if por_ha:
                z = Zonas(z.produto, [cfg.por_ha(d) for d in z.niveis],
                          [(round(cfg.por_ha(d)), g) for d, g in z.poligonos], z.classes)
            nome = nome_shapefile("Sementes", cods[m.talhao.nome])
            pasta = _nome_arquivo(m.talhao.nome) or cods[m.talhao.nome]
            for arq, dados in shapefile_bytes(z, epsg, nome, casas=0 if por_ha else
                                              cfg.casas(cfg.media_de(m.talhao.nome))).items():
                zf.writestr(f"{pasta}/{arq}", dados)
            lista.append(f"{pasta}/{nome}.shp  ->  {m.talhao.nome} · sementes ({cfg.unidade_shape})")
        zf.writestr("LEIA-ME.txt", _leia_me(lista).encode("utf-8"))
        try:
            from .prescricao import conferir
            conf = conferir([m for m in mapas if "Sementes" in m.zonas], epsg, ["Sementes"])
            zf.writestr("CONFERENCIA.txt", ("CONFERÊNCIA DOS MAPAS DE SEMEADURA (sementes/m; shapefiles relidos)\r\n\r\n"
                                            + conf.to_string(index=False)).encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            zf.writestr("CONFERENCIA.txt", f"Conferência não concluída: {e}".encode("utf-8"))
    return buf.getvalue()
