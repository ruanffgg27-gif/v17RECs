"""Perfil climático do local: chuva (CHIRPS; reserva NASA POWER) e temperatura (NASA POWER), desde 1981.

As bases são grades (CHIRPS ~5 km; NASA POWER ~50 km): talhões vizinhos compartilham a mesma célula.
Os números descrevem o CONTEXTO climático do local, não chuva medida no talhão, e são estatísticas
históricas — não previsão.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import fontes
from .base import Area

MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
ORDEM_AGRICOLA = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]


@dataclass
class Clima:
    diario: pd.DataFrame = field(repr=False)          # colunas: chuva, tmax, tmin
    fonte_chuva: str = ""
    fonte_temp: str = ""
    mensal: pd.DataFrame = field(default_factory=pd.DataFrame)       # ano × mês (mm)
    normal: pd.DataFrame = field(default_factory=pd.DataFrame)       # mês: média, p20, p80
    anual: pd.DataFrame = field(default_factory=pd.DataFrame)        # ano agrícola: total
    atual: dict = field(default_factory=dict)
    veranicos: pd.DataFrame = field(default_factory=pd.DataFrame)    # mês: % anos com ≥10 e ≥15 dias
    estacao: pd.DataFrame = field(default_factory=pd.DataFrame)      # ano agrícola: início, fim, duração
    extremos: pd.DataFrame = field(default_factory=pd.DataFrame)     # mês: dias/ano com calor e frio
    anos_extremos: dict = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)
    vento: object | None = None                                      # vento.Vento (regime de vento e pulverização)


def obter(area: Area, inicio: date = date(1981, 1, 1)) -> Clima | None:
    from concurrent.futures import ThreadPoolExecutor

    from . import vento as vt
    lon, lat = area.centro
    fim_power = fontes.hoje() - timedelta(days=5)

    def seguro(f):
        try:
            return f()
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(3) as ex:
        f_power = ex.submit(seguro, lambda: fontes.power_diario(lon, lat, inicio, fim_power))
        f_chirps = ex.submit(seguro, lambda: fontes.chirps_climateserv(lon, lat, inicio,
                                                                      fontes.hoje() - timedelta(days=2)))
        f_vento = ex.submit(seguro, lambda: vt.obter(area))
        power, chirps, vento = f_power.result(), f_chirps.result(), f_vento.result()
    if power is None and chirps is None:
        return None
    df = pd.DataFrame(index=pd.date_range(inicio, max(
        (power.index.max() if power is not None else pd.Timestamp(inicio)),
        (chirps.index.max() if chirps is not None else pd.Timestamp(inicio))), freq="D"))
    avisos = []
    if chirps is not None:
        df["chuva"] = chirps
        fonte_chuva = "CHIRPS 2.0 (UCSB/CHC, ~5 km) via ClimateSERV"
    else:
        df["chuva"] = power["PRECTOTCORR"] if power is not None else np.nan
        fonte_chuva = "NASA POWER (MERRA-2/IMERG, ~50 km)"
        avisos.append("CHIRPS indisponível no momento; chuva da NASA POWER (grade mais grossa)")
    if power is not None:
        df["tmax"] = power["T2M_MAX"]
        df["tmin"] = power["T2M_MIN"]
        fonte_temp = "NASA POWER (MERRA-2, ~50 km)"
    else:
        df["tmax"] = np.nan
        df["tmin"] = np.nan
        fonte_temp = ""
        avisos.append("Temperaturas indisponíveis (NASA POWER não respondeu)")
    c = analisar(df, fonte_chuva, fonte_temp)
    c.vento = vento
    if vento is None:
        avisos.append("Vento indisponível (NASA POWER não respondeu)")
    else:
        avisos += vento.avisos
    c.avisos = avisos + c.avisos
    return c


def ano_agricola(ts: pd.Timestamp) -> int:
    return ts.year if ts.month >= 7 else ts.year - 1


def rotulo_ano(a: int) -> str:
    return f"{a}/{(a + 1) % 100:02d}"


def secas(chuva: pd.Series, lim_mm: float = 1.0) -> pd.DataFrame:
    """Sequências de dias secos (chuva < lim_mm): início e duração."""
    seco = (chuva.fillna(0) < lim_mm).astype(int).values
    idx = chuva.index
    out, i, n = [], 0, len(seco)
    while i < n:
        if seco[i]:
            j = i
            while j < n and seco[j]:
                j += 1
            out.append({"inicio": idx[i], "dias": j - i})
            i = j
        else:
            i += 1
    return pd.DataFrame(out)


def inicio_estacao(chuva: pd.Series, a: int) -> pd.Timestamp | None:
    """Início das chuvas no ano agrícola a: 1º dia após 1/set com ≥ 20 mm em 3 dias e sem sequência seca
    ≥ 10 dias nos 30 dias seguintes (critério agronômico usual)."""
    s = chuva[pd.Timestamp(a, 9, 1):pd.Timestamp(a + 1, 1, 31)].fillna(0)
    if len(s) < 60:
        return None
    soma3 = s.rolling(3).sum()
    for dia in soma3.index[soma3 >= 20]:
        prox = chuva[dia + pd.Timedelta(days=1):dia + pd.Timedelta(days=30)]
        sq = secas(prox)
        if sq.empty or sq["dias"].max() < 10:
            return dia - pd.Timedelta(days=2)
    return None


def fim_estacao(chuva: pd.Series, a: int) -> pd.Timestamp | None:
    """Fim das chuvas: 1º dia após 1/mar a partir do qual 20 dias somam < 10 mm."""
    s = chuva[pd.Timestamp(a + 1, 3, 1):pd.Timestamp(a + 1, 7, 31)].fillna(0)
    if len(s) < 60:
        return None
    frente = s[::-1].rolling(20).sum()[::-1]
    ok = frente.index[frente < 10]
    return ok[0] if len(ok) else None


def analisar(df: pd.DataFrame, fonte_chuva: str = "", fonte_temp: str = "") -> Clima:
    c = Clima(df, fonte_chuva, fonte_temp)
    chuva = df["chuva"]
    ultimo = chuva.last_valid_index()
    # mensal e normais (só meses completos)
    mm = chuva.resample("MS").agg(["sum", "count"])
    mm = mm[mm["count"] >= 25]
    mm["ano"], mm["mes"] = mm.index.year, mm.index.month
    c.mensal = mm.pivot(index="ano", columns="mes", values="sum")
    hist = c.mensal.loc[c.mensal.index < ultimo.year]
    c.normal = pd.DataFrame({"media": hist.mean(), "p20": hist.quantile(0.2), "p80": hist.quantile(0.8)})
    # anos agrícolas completos
    ag = chuva.groupby(chuva.index.map(ano_agricola)).agg(["sum", "count"])
    ag = ag[ag["count"] >= 360]
    c.anual = pd.DataFrame({"Ano agrícola": [rotulo_ano(a) for a in ag.index], "Chuva (mm)": ag["sum"].values},
                           index=ag.index)
    # safra atual vs histórico (acumulado desde 1º de julho)
    a_at = ano_agricola(ultimo)
    ini = pd.Timestamp(a_at, 7, 1)
    dias = (ultimo - ini).days
    acum_at = chuva[ini:ultimo].fillna(0).cumsum()
    curvas = {}
    for a in c.anual.index:
        s = chuva[pd.Timestamp(a, 7, 1):pd.Timestamp(a, 7, 1) + pd.Timedelta(days=364)].fillna(0).cumsum()
        curvas[a] = s.values[:365]
    if curvas and dias > 10:
        M = np.array([v for v in curvas.values() if len(v) >= 365])
        mesmo = M[:, min(dias, 364)]
        tot_at = float(acum_at.iloc[-1])
        c.atual = {"ano": rotulo_ano(a_at), "ate": ultimo.date(), "acumulado": tot_at,
                   "mediana": float(np.median(mesmo)), "pct_normal": float(100 * tot_at / max(np.median(mesmo), 1)),
                   "percentil": float(100 * (mesmo < tot_at).mean()),
                   "curva_atual": acum_at.values, "p10": np.percentile(M, 10, axis=0),
                   "p50": np.percentile(M, 50, axis=0), "p90": np.percentile(M, 90, axis=0)}
        # anos semelhantes, mais secos e mais úmidos (no mesmo período)
        ordem = sorted(curvas, key=lambda a: curvas[a][min(dias, len(curvas[a]) - 1)])
        c.anos_extremos = {"secos": [rotulo_ano(a) for a in ordem[:3]],
                           "umidos": [rotulo_ano(a) for a in ordem[-3:][::-1]]}
    # veranicos por mês da estação chuvosa (out–mar)
    sq = secas(chuva[:ultimo])
    anos = sorted(set(chuva[:ultimo].index.map(ano_agricola)))[:-1] if len(chuva) else []
    linhas = []
    for mes in (10, 11, 12, 1, 2, 3):
        n10 = n15 = tot = 0
        for a in anos:
            ano_cal = a if mes >= 7 else a + 1
            ini_m = pd.Timestamp(ano_cal, mes, 1)
            fim_m = ini_m + pd.offsets.MonthEnd(0)
            if fim_m > ultimo:
                continue
            tot += 1
            d = sq[(sq["inicio"] >= ini_m) & (sq["inicio"] <= fim_m)] if not sq.empty else sq
            mx = int(d["dias"].max()) if len(d) else 0
            n10 += mx >= 10
            n15 += mx >= 15
        if tot:
            linhas.append({"Mês": MESES[mes - 1], "Anos": tot, "≥ 10 dias secos (%)": 100 * n10 / tot,
                           "≥ 15 dias secos (%)": 100 * n15 / tot})
    c.veranicos = pd.DataFrame(linhas)
    # início e fim da estação chuvosa
    est = []
    for a in anos:
        i0, f0 = inicio_estacao(chuva, a), fim_estacao(chuva, a)
        est.append({"Ano agrícola": rotulo_ano(a), "Início": i0, "Fim": f0,
                    "Duração (dias)": (f0 - i0).days if (i0 is not None and f0 is not None) else np.nan})
    c.estacao = pd.DataFrame(est)
    # extremos de temperatura
    if df["tmax"].notna().sum() > 365 * 5:
        t = df.dropna(subset=["tmax", "tmin"])
        n_anos = t.index.year.nunique()
        g = t.groupby(t.index.month)
        c.extremos = pd.DataFrame({
            "Mês": [MESES[m - 1] for m in g.size().index],
            "Tmáx média (°C)": g["tmax"].mean().values, "Tmín média (°C)": g["tmin"].mean().values,
            "Dias ≥ 35 °C (por ano)": (g["tmax"].apply(lambda s: (s >= 35).sum()) / n_anos).values,
            "Dias ≤ 5 °C (por ano)": (g["tmin"].apply(lambda s: (s <= 5).sum()) / n_anos).values,
            "Dias ≤ 2 °C (por ano)": (g["tmin"].apply(lambda s: (s <= 2).sum()) / n_anos).values})
    return c


def texto_estacao(c: Clima) -> dict:
    """Datas típicas (mediana e quartis) de início e fim das chuvas, em dia/mês."""
    def doy_ag(ts):
        return (ts - pd.Timestamp(ano_agricola(ts), 7, 1)).days

    def data_de(d):
        return (pd.Timestamp(2001, 7, 1) + pd.Timedelta(days=int(round(d)))).strftime("%d/%m")
    out = {}
    for col in ("Início", "Fim"):
        v = [doy_ag(t) for t in c.estacao[col].dropna()] if not c.estacao.empty else []
        if len(v) >= 5:
            out[col] = {"mediana": data_de(np.median(v)), "p25": data_de(np.percentile(v, 25)),
                        "p75": data_de(np.percentile(v, 75)), "n": len(v), "valores": v}
    return out
