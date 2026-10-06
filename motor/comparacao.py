"""Comparação entre dois laudos da mesma área (ano anterior × ano recente).

Os atributos do ano anterior são interpolados na MESMA grade do ano recente (mesmo perímetro e pixel), para que
os mapas e as áreas por classe sejam diretamente comparáveis. As prescrições continuam sendo só as do laudo mais
recente; do ano anterior calculamos, pela mesma regra, os volumes que teriam sido recomendados.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .geo import Projeto
from .interpretacao import ATRIBUTOS, INVERTIDOS, completar_derivados, posicao_invertida
from .parametros import AjusteProduto, Ajustes, Parametros
from .prescricao import ATRIB_MAPA, BASE_DOSE, MapaTalhao, interpolar_talhao


@dataclass
class Comparacao:
    rotulos: tuple[str, str]                    # (ano anterior, ano recente)
    pares: list[tuple[MapaTalhao, MapaTalhao]]  # (anterior, recente) — mesma grade
    so_recente: list[str] = field(default_factory=list)
    so_anterior: list[str] = field(default_factory=list)
    volumes: pd.DataFrame = field(default_factory=pd.DataFrame)
    avisos: list[str] = field(default_factory=list)

    @property
    def mapas_a(self) -> list[MapaTalhao]:
        return [a for a, _ in self.pares]

    @property
    def mapas_b(self) -> list[MapaTalhao]:
        return [b for _, b in self.pares]


def ano_do_nome(nome: str) -> str:
    """'Laudo 8438-2026 …' → '2026' (último ano de 4 dígitos entre 1990 e 2099)."""
    anos = [a for a in re.findall(r"(?<!\d)(19[9]\d|20\d\d)(?!\d)", str(nome))]
    return anos[-1] if anos else ""


def superficies_na_grade(mapas_b: list[MapaTalhao], projeto_a: Projeto, atributos=None, modo: str = "estatistico",
                         modo_doses: str | None = None, **opcoes) -> Comparacao:
    """Interpola as amostras do ano anterior na grade de cada talhão do ano recente."""
    atributos = list(dict.fromkeys(list(atributos or ATRIB_MAPA) + BASE_DOSE))
    por_chave = {t.chave: t for t in projeto_a.talhoes}
    pares, so_b = [], []
    for mb in mapas_b:
        ta = por_chave.get(mb.talhao.chave)
        if ta is None or not len(ta.amostras):
            so_b.append(mb.talhao.nome)
            continue
        a = completar_derivados(ta.amostras)
        ma = MapaTalhao(replace(ta, perimetro=mb.talhao.perimetro, perimetro_ll=mb.talhao.perimetro_ll,
                                nome=mb.talhao.nome), mb.grade)
        interpolar_talhao(ma, a, atributos, modo, modo_doses, so_se=set(mb.atributos), **opcoes)
        pares.append((ma, mb))
    chaves_b = {m.talhao.chave for m in mapas_b}
    so_a = [t.nome for t in projeto_a.talhoes if t.chave not in chaves_b]
    c = Comparacao(("", ""), pares, so_b, so_a)
    if so_b:
        c.avisos.append("Sem laudo do ano anterior (fora da comparação): " + ", ".join(so_b))
    if so_a:
        c.avisos.append("Talhões só no laudo anterior (ignorados): " + ", ".join(so_a))
    return c


def _sem_alvo(a: AjusteProduto) -> AjusteProduto:
    return AjusteProduto(pct=a.pct, media_alvo=None)


def volumes_anterior(c: Comparacao, par: Parametros, abertura: dict, aj: Ajustes) -> None:
    """Doses do ano anterior pela MESMA regra e parâmetros (sem o ajuste de volume comprado, que é da safra atual).

    Guarda em c.volumes: Talhão, Produto, Área (ha), dose média e total de cada ano."""
    from .prescricao import calcular_doses
    aj_a = replace(aj, calcario=_sem_alvo(aj.calcario), gesso=_sem_alvo(aj.gesso), p2o5=_sem_alvo(aj.p2o5),
                   kcl=_sem_alvo(aj.kcl))
    mapas_a = [a for a in c.mapas_a if all(k in a.para_doses() for k in ("ca", "mg", "hal", "k", "p", "s", "argila"))]
    if mapas_a:
        calcular_doses(mapas_a, par, abertura, aj_a)
    linhas = []
    for ma, mb in c.pares:
        for prod, zon in mb.zonas.items():
            if prod.startswith("KCl ") or (prod == "P2O5" and any(p.startswith("Produto") for p in mb.zonas)):
                continue
            area = zon.area_ha
            tot_b = zon.total_kg / 1000
            dose_a = tot_a = np.nan
            if prod in ma.doses:
                D = ma.doses[prod][ma.grade.mascara]
                if np.isfinite(D).any():
                    dose_a = float(np.nanmean(D))
                    tot_a = dose_a * area / 1000
            linhas.append({"Talhão": mb.talhao.nome, "Produto": prod, "Área (ha)": area,
                           f"Dose média {c.rotulos[0]} (kg/ha)": dose_a, f"Total {c.rotulos[0]} (t)": tot_a,
                           f"Dose média {c.rotulos[1]} (kg/ha)": zon.dose_media, f"Total {c.rotulos[1]} (t)": tot_b,
                           "Variação (%)": 100 * (tot_b - tot_a) / tot_a if tot_a and np.isfinite(tot_a) else np.nan})
    c.volumes = pd.DataFrame(linhas)


# atributos em que subir acima de "Excelente" pode ser excesso (não conta como melhora)
EXCESSO_POSSIVEL = {"ph", "v", "sat_ca", "b"}


def posicao(chave: str, v: float) -> float:
    """0 (crítico) … 8 (muito alto) — mesma escala do panorama; considera os atributos 'menor é melhor'.
    Para pH, V%, Ca na CTC e B a posição é limitada a 'Excelente' (7): passar para 'Muito alto' não é melhora."""
    faixas = sorted(ATRIBUTOS[chave][2], key=lambda t: t[1])
    lim = [faixas[0][1]] + [f[2] for f in faixas]
    p = float(np.interp(v, lim, np.arange(len(lim))))
    p = posicao_invertida(p) if chave in INVERTIDOS else p
    return min(p, 7.0) if chave in EXCESSO_POSSIVEL else p


def media_area(mapas: list[MapaTalhao], chave: str) -> float:
    v = np.concatenate([m.atributos[chave][m.grade.mascara] for m in mapas if chave in m.atributos])
    return float(np.nanmean(v)) if len(v) else float("nan")


def tabela_evolucao(c: Comparacao, chaves: list[str], fator=None) -> pd.DataFrame:
    """Média da área em cada ano, variação e se a condição melhorou (pela escala de interpretação)."""
    fator = fator or {}
    linhas = []
    for k in chaves:
        a = media_area(c.mapas_a, k) * fator.get(k, 1.0)
        b = media_area(c.mapas_b, k) * fator.get(k, 1.0)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        linha = {"chave": k, "A": a, "B": b, "Δ": b - a, "Δ%": 100 * (b - a) / a if a else np.nan}
        if k in ATRIBUTOS:
            from .interpretacao import classificar
            linha.update({"classe A": classificar(k, a / fator.get(k, 1.0)),
                          "classe B": classificar(k, b / fator.get(k, 1.0)),
                          "melhora": posicao(k, b / fator.get(k, 1.0)) - posicao(k, a / fator.get(k, 1.0))})
        linhas.append(linha)
    return pd.DataFrame(linhas)
