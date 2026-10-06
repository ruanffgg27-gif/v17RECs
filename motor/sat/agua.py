"""Água no solo: balanço hídrico climatológico recente e umidade superficial do NISAR.

Só o perímetro é obrigatório. Cultura, data de semeadura, profundidade radicular, capacidade de água disponível
(CAD) e irrigação são OPCIONAIS e vêm do usuário — o relatório separa o que é dado público do que foi informado.

1. BALANÇO HÍDRICO (últimos 90 dias)
   - Chuva diária: CHIRPS 2.0 (~5 km) onde já houver dado; nos dias mais recentes, ainda não publicados no CHIRPS,
     NASA POWER (PRECTOTCORR, ~50 km). Cada dia guarda a sua fonte. Dia sem dado em nenhuma fonte fica SEM DADO
     (não vira zero) e é contado à parte.
   - ET₀ diária: Penman-Monteith FAO-56 (Allen et al., 1998, eq. 6) com temperatura máxima e mínima, umidade
     relativa, vento a 2 m, radiação solar e pressão da NASA POWER. Onde falta radiação, umidade ou vento (comum
     nos últimos dias), usa-se Hargreaves-Samani (FAO-56, eq. 52), que só precisa das temperaturas — o método de
     cada dia fica registrado. Sem temperatura, o dia fica sem ET₀.
   - Sem parâmetros de solo e cultura apresenta-se só chuva, ET₀ e o saldo P − ET₀. Esse saldo é um indicador
     CLIMÁTICO (oferta menos demanda atmosférica de referência): NÃO é água disponível às plantas.

2. ARMAZENAMENTO NA ZONA RADICULAR — só com cultura, data de semeadura, profundidade radicular e CAD informados
   - Balanço diário FAO-56 com coeficiente de cultura único (cap. 6 e 8):
       ETc = Kc · ET₀;   ETa = Ks · ETc;   Ks = (CAD_total − D) / ((1 − p) · CAD_total) quando D > p · CAD_total
       D(hoje) = D(ontem) − chuva − irrigação + ETa;   D < 0 → excedente (drenagem/escoamento), D = 0
     CAD_total = CAD (mm por cm de solo) × profundidade radicular (cm), constante no período.
   - Kc por fase (inicial, desenvolvimento, média, final) e fração p de esgotamento: FAO-56, tabelas 11, 12 e 22
     (valores em CULTURAS). Fora do ciclo (antes da semeadura ou depois da colheita) usa-se Kc = 0,30 (solo sem
     cultura). São coeficientes GENÉRICOS — cultivar, população e manejo mudam o consumo real.
   - Inicialização: a simulação começa 90 dias ANTES da janela apresentada, com três condições iniciais (solo
     cheio, pela metade e seco). A curva mostrada parte de metade; a faixa entre "cheio" e "seco" mostra quanto o
     resultado ainda depende do chute inicial. Faixa larga no fim = resultado pouco confiável.
   - Toda a chuva é considerada infiltrada (sem escoamento superficial nem ascensão capilar).
   - Irrigação: "não informada" NÃO é tratada como inexistente — o armazenamento sai como CENÁRIO SEM IRRIGAÇÃO,
     com esse nome; se a área é irrigada, o armazenamento real é maior. Com lâminas informadas, elas entram no
     balanço. Dia sem chuva ou ET₀ conhecidos não atualiza o armazenamento (fica como no dia anterior) e é
     contado; com mais de 10% de dias assim na janela, o armazenamento não é apresentado.

3. UMIDADE SUPERFICIAL NISAR — ver motor/sat/nisar.py. É apresentada ao lado do balanço, sem misturar: não corrige
   o armazenamento e não representa o perfil do solo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import fontes
from .base import Area

JANELA = 90            # dias apresentados
AQUECIMENTO = 90       # dias simulados antes da janela (inicialização)
KC_SEM_CULTURA = 0.30
POWER_ET0 = ("T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PS", "PRECTOTCORR")

# FAO-56 (Allen et al., 1998): duração das fases em dias (tab. 11), Kc inicial/médio/final (tab. 12) e fração p de
# esgotamento sem estresse para ETc ≈ 5 mm/dia (tab. 22). Valores genéricos de referência.
CULTURAS = {
    "soja": ("Soja", (20, 35, 60, 25), (0.40, 1.15, 0.50), 0.50),
    "milho": ("Milho (grão)", (30, 40, 50, 30), (0.30, 1.20, 0.35), 0.55),
    "algodao": ("Algodão", (30, 50, 60, 55), (0.35, 1.18, 0.60), 0.65),
    "feijao": ("Feijão", (20, 30, 40, 20), (0.40, 1.15, 0.35), 0.45),
    "trigo": ("Trigo", (15, 30, 65, 40), (0.30, 1.15, 0.30), 0.55),
    "sorgo": ("Sorgo (grão)", (20, 35, 45, 30), (0.30, 1.05, 0.55), 0.55),
    "cana": ("Cana-de-açúcar (soca)", (30, 50, 180, 60), (0.40, 1.25, 0.75), 0.65),
    "pastagem": ("Pastagem", (10, 20, 300, 35), (0.40, 0.95, 0.85), 0.60),
}
IRRIGACAO = {"nao_informado": "Não informado", "sequeiro": "Sequeiro (sem irrigação)", "irrigado": "Irrigado"}


@dataclass(frozen=True)
class ParametrosAgua:
    """Informações OPCIONAIS fornecidas pelo usuário."""
    cultura: str = ""                       # chave de CULTURAS
    semeadura: date | None = None
    ciclo_dias: int | None = None           # se informado, as fases da FAO são proporcionalmente ajustadas
    prof_raiz_cm: float | None = None
    cad_mm_cm: float | None = None          # capacidade de água disponível, mm de água por cm de solo
    irrigacao: str = "nao_informado"
    laminas: tuple = ()                     # ((data, mm), …) irrigações informadas

    @property
    def completos(self) -> bool:
        return bool(self.cultura in CULTURAS and self.semeadura and self.prof_raiz_cm and self.cad_mm_cm)

    def faltam(self) -> list[str]:
        return [n for n, ok in (("cultura", self.cultura in CULTURAS), ("data de semeadura", bool(self.semeadura)),
                                ("profundidade radicular", bool(self.prof_raiz_cm)),
                                ("capacidade de água disponível (CAD)", bool(self.cad_mm_cm))) if not ok]

    def informados(self) -> list[tuple[str, str]]:
        out = []
        if self.cultura in CULTURAS:
            out.append(("Cultura", CULTURAS[self.cultura][0]))
        if self.semeadura:
            out.append(("Semeadura", f"{self.semeadura:%d/%m/%Y}"))
        if self.ciclo_dias:
            out.append(("Ciclo", f"{int(self.ciclo_dias)} dias"))
        if self.prof_raiz_cm:
            out.append(("Profundidade radicular", f"{self.prof_raiz_cm:g} cm"))
        if self.cad_mm_cm:
            out.append(("CAD", f"{self.cad_mm_cm:g} mm/cm".replace(".", ",")))
        out.append(("Irrigação", IRRIGACAO.get(self.irrigacao, self.irrigacao)
                    + (f" · {len(self.laminas)} lâmina(s) informada(s)" if self.laminas else "")))
        return out


@dataclass
class Agua:
    serie: pd.DataFrame = field(repr=False)   # índice diário: chuva, fonte_chuva, et0, metodo_et0, saldo
    par: ParametrosAgua = field(default_factory=ParametrosAgua)
    inicio: date | None = None
    fim: date | None = None
    resumo: dict = field(default_factory=dict)             # totais e contagens da janela
    armazenamento: pd.DataFrame | None = field(default=None, repr=False)
    cad_total: float = float("nan")
    cenario: str = ""                                      # nome do cenário simulado
    motivo_sem_armazenamento: str = ""
    kc_info: dict = field(default_factory=dict)
    nisar: object | None = None                            # nisar.Nisar
    fontes: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ ET0
def _ra(lat_graus: float, dia_do_ano: np.ndarray) -> np.ndarray:
    """Radiação no topo da atmosfera (MJ m⁻² dia⁻¹) — FAO-56, eq. 21."""
    phi = np.radians(lat_graus)
    dr = 1 + 0.033 * np.cos(2 * np.pi * dia_do_ano / 365)
    delta = 0.409 * np.sin(2 * np.pi * dia_do_ano / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def et0_fao56(df: pd.DataFrame, lat: float, altitude: float = 400.0) -> pd.DataFrame:
    """ET₀ diária (mm). Colunas esperadas (NASA POWER, comunidade AG): T2M_MAX, T2M_MIN (°C), RH2M (%), WS2M (m/s),
    ALLSKY_SFC_SW_DWN (MJ m⁻² dia⁻¹), PS (kPa). Devolve colunas `et0` e `metodo` ("PM", "HS" ou "")."""
    tmax, tmin = df["T2M_MAX"].astype(float), df["T2M_MIN"].astype(float)
    t = (tmax + tmin) / 2
    j = df.index.dayofyear.to_numpy()
    ra = pd.Series(_ra(lat, j), index=df.index)
    nan = pd.Series(np.nan, index=df.index)
    rh = df["RH2M"].astype(float) if "RH2M" in df else nan
    u2 = df["WS2M"].astype(float) if "WS2M" in df else nan
    rs = df["ALLSKY_SFC_SW_DWN"].astype(float) if "ALLSKY_SFC_SW_DWN" in df else nan
    if rs.notna().sum() >= 5 and rs.median() < 9.5:
        rs = rs * 3.6                         # veio em kWh m⁻² dia⁻¹ (outra "comunidade" da POWER) → MJ m⁻² dia⁻¹
    ps = df["PS"].astype(float) if "PS" in df else nan
    ps = ps.fillna(101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26)          # eq. 7
    es = (0.6108 * np.exp(17.27 * tmax / (tmax + 237.3)) + 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))) / 2
    ea = es * rh / 100                       # com UR média diária (FAO-56, eq. 19, forma simplificada)
    delta = 4098 * (0.6108 * np.exp(17.27 * t / (t + 237.3))) / (t + 237.3) ** 2
    gama = 0.000665 * ps
    rso = (0.75 + 2e-5 * altitude) * ra
    rns = 0.77 * rs
    sigma = 4.903e-9
    rnl = sigma * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea.clip(lower=0))) * \
        (1.35 * (rs / rso).clip(0.3, 1.0) - 0.35)
    rn = rns - rnl
    pm = (0.408 * delta * rn + gama * 900 / (t + 273) * u2 * (es - ea)) / (delta + gama * (1 + 0.34 * u2))
    hs = 0.0023 * (t + 17.8) * np.sqrt((tmax - tmin).clip(lower=0)) * ra * 0.408   # eq. 52 (Ra em mm/dia)
    et0 = pm.where(pm.notna(), hs).clip(lower=0)
    metodo = pd.Series(np.where(pm.notna(), "PM", np.where(hs.notna(), "HS", "")), index=df.index)
    return pd.DataFrame({"et0": et0, "metodo": metodo})


# ------------------------------------------------------------------ séries
def _chirps_recente(lon: float, lat: float, inicio: date, fim: date) -> pd.Series | None:
    """CHIRPS no período (um pedido só ao ClimateSERV). Devolve só os dias publicados."""
    import json
    mj = 0.04
    anel = [[lon - mj, lat - mj], [lon + mj, lat - mj], [lon + mj, lat + mj], [lon - mj, lat + mj],
            [lon - mj, lat - mj]]
    geom = json.dumps({"type": "Polygon", "coordinates": [anel]}).replace(" ", "")
    d = fontes._chirps_pedido(geom, inicio, fim, 150)
    return pd.Series(d).sort_index() if d else None


def montar_serie(power: pd.DataFrame | None, chirps: pd.Series | None, lat: float, altitude: float,
                 inicio: date, fim: date) -> pd.DataFrame:
    """Série diária (inicio…fim) com chuva, fonte da chuva, ET₀, método e saldo. Lacunas ficam como NaN."""
    idx = pd.date_range(inicio, fim, freq="D")
    s = pd.DataFrame(index=idx)
    s["chuva"] = np.nan
    s["fonte_chuva"] = ""
    if chirps is not None and len(chirps):
        c = chirps.reindex(idx)
        s.loc[c.notna(), "chuva"] = c[c.notna()]
        s.loc[c.notna(), "fonte_chuva"] = "CHIRPS"
    if power is not None and "PRECTOTCORR" in power:
        pw = power["PRECTOTCORR"].reindex(idx)
        usar = s["chuva"].isna() & pw.notna()
        s.loc[usar, "chuva"] = pw[usar]
        s.loc[usar, "fonte_chuva"] = "POWER"
    if power is not None and {"T2M_MAX", "T2M_MIN"} <= set(power.columns):
        e = et0_fao56(power.reindex(idx), lat, altitude)
        s["et0"], s["metodo_et0"] = e["et0"], e["metodo"]
    else:
        s["et0"], s["metodo_et0"] = np.nan, ""
    s["saldo"] = s["chuva"] - s["et0"]              # NaN se faltar um dos dois
    return s


def resumir(s: pd.DataFrame) -> dict:
    n = len(s)
    ok = s["saldo"].notna()
    return {"dias": n, "dias_chuva": int(s["chuva"].notna().sum()), "dias_et0": int(s["et0"].notna().sum()),
            "dias_saldo": int(ok.sum()),
            "chuva": float(s["chuva"].sum(min_count=1)) if s["chuva"].notna().any() else float("nan"),
            "et0": float(s["et0"].sum(min_count=1)) if s["et0"].notna().any() else float("nan"),
            # o saldo soma só os dias em que chuva E ET0 são conhecidas
            "saldo": float(s.loc[ok, "saldo"].sum()) if ok.any() else float("nan"),
            "chuva_dias_saldo": float(s.loc[ok, "chuva"].sum()) if ok.any() else float("nan"),
            "et0_dias_saldo": float(s.loc[ok, "et0"].sum()) if ok.any() else float("nan"),
            "dias_chirps": int((s["fonte_chuva"] == "CHIRPS").sum()),
            "dias_power": int((s["fonte_chuva"] == "POWER").sum()),
            "dias_pm": int((s["metodo_et0"] == "PM").sum()), "dias_hs": int((s["metodo_et0"] == "HS").sum()),
            "ultima_chuva": (s.index[s["chuva"] >= 1.0].max().date() if (s["chuva"] >= 1.0).any() else None)}


# ------------------------------------------------------------------ armazenamento (FAO-56)
def curva_kc(cultura: str, semeadura: date, datas: pd.DatetimeIndex, ciclo_dias: int | None = None) -> pd.DataFrame:
    """Kc e fase de cada dia. Fora do ciclo: KC_SEM_CULTURA e fase "sem cultura"."""
    _, fases, (kini, kmid, kend), _ = CULTURAS[cultura]
    fases = np.array(fases, float)
    if ciclo_dias:
        fases = fases * float(ciclo_dias) / fases.sum()
    a, b, c, d = np.cumsum(fases)
    dias = (datas - pd.Timestamp(semeadura)).days.to_numpy().astype(float)
    kc = np.full(len(datas), KC_SEM_CULTURA)
    fase = np.full(len(datas), "sem cultura", dtype=object)
    m = (dias >= 0) & (dias < a)
    kc[m], fase[m] = kini, "inicial"
    m = (dias >= a) & (dias < b)
    kc[m], fase[m] = kini + (kmid - kini) * (dias[m] - a) / (b - a), "desenvolvimento"
    m = (dias >= b) & (dias < c)
    kc[m], fase[m] = kmid, "média"
    m = (dias >= c) & (dias <= d)
    kc[m], fase[m] = kmid + (kend - kmid) * (dias[m] - c) / (d - c), "final"
    return pd.DataFrame({"kc": kc, "fase": fase, "dias_apos_semeadura": dias}, index=datas)


def simular(s: pd.DataFrame, kc: pd.Series, cad_total: float, p: float, d0_frac: float,
            irrigacao: pd.Series | None = None) -> pd.DataFrame:
    """Balanço diário. `d0_frac`: fração da CAD já esgotada no primeiro dia (0 = solo cheio, 1 = seco).
    Dias sem chuva ou ET₀ conhecidas não atualizam o estado (coluna `sem_dado`)."""
    D = d0_frac * cad_total
    raw = p * cad_total
    out = []
    irr = irrigacao.reindex(s.index).fillna(0.0) if irrigacao is not None else pd.Series(0.0, index=s.index)
    for dia, chuva, et0, k, lam in zip(s.index, s["chuva"].to_numpy(), s["et0"].to_numpy(), kc.to_numpy(),
                                       irr.to_numpy()):
        if not (np.isfinite(chuva) and np.isfinite(et0)):
            out.append((dia, cad_total - D, np.nan, np.nan, np.nan, 0.0, True))
            continue
        etc = k * et0
        ks = 1.0 if D <= raw else max(0.0, (cad_total - D) / ((1 - p) * cad_total))
        eta = ks * etc
        D = D - chuva - lam + eta
        exced = 0.0
        if D < 0:
            exced, D = -D, 0.0
        D = min(D, cad_total)
        out.append((dia, cad_total - D, etc, eta, ks, exced, False))
    r = pd.DataFrame(out, columns=["data", "arm", "etc", "eta", "ks", "excedente", "sem_dado"]).set_index("data")
    r["arm_pct"] = 100 * r["arm"] / cad_total
    return r


def armazenamento(serie_longa: pd.DataFrame, par: ParametrosAgua, inicio_janela: date
                  ) -> tuple[pd.DataFrame | None, dict, str, str]:
    """(tabela da janela, informações de Kc, nome do cenário, motivo se não calculado)."""
    if not par.completos:
        return None, {}, "", "faltam: " + ", ".join(par.faltam())
    nome, fases, kcs, p = CULTURAS[par.cultura]
    cad_total = float(par.cad_mm_cm) * float(par.prof_raiz_cm)
    k = curva_kc(par.cultura, par.semeadura, serie_longa.index, par.ciclo_dias)
    irr = None
    if par.irrigacao == "irrigado" and par.laminas:
        irr = pd.Series({pd.Timestamp(d): float(mm) for d, mm in par.laminas}).groupby(level=0).sum()
        cenario = f"com as {len(par.laminas)} irrigações informadas"
    elif par.irrigacao == "irrigado":
        cenario = "cenário SEM irrigação (área irrigada, lâminas não informadas — o armazenamento real é maior)"
    elif par.irrigacao == "sequeiro":
        cenario = "sequeiro (informado pelo usuário)"
    else:
        cenario = "cenário SEM irrigação (irrigação não informada — se houver, o armazenamento real é maior)"
    base = simular(serie_longa, k["kc"], cad_total, p, 0.5, irr)
    cheio = simular(serie_longa, k["kc"], cad_total, p, 0.0, irr)
    seco = simular(serie_longa, k["kc"], cad_total, p, 1.0, irr)
    t = base.join(k)
    t["arm_max"], t["arm_min"] = cheio["arm"], seco["arm"]
    t["irrigacao"] = irr.reindex(t.index).fillna(0.0) if irr is not None else 0.0
    jan = t[t.index >= pd.Timestamp(inicio_janela)]
    frac_sem = float(jan["sem_dado"].mean()) if len(jan) else 1.0
    info = {"cultura": nome, "fases_dias": tuple(int(round(v)) for v in (np.array(fases, float) * (
        float(par.ciclo_dias) / sum(fases) if par.ciclo_dias else 1.0))), "kc": kcs, "p": p,
        "cad_total": cad_total, "kc_sem_cultura": KC_SEM_CULTURA,
        "faixa_inicio": float(jan["arm_max"].iloc[0] - jan["arm_min"].iloc[0]) if len(jan) else float("nan"),
        "faixa_fim": float(jan["arm_max"].iloc[-1] - jan["arm_min"].iloc[-1]) if len(jan) else float("nan"),
        "dias_sem_dado": int(jan["sem_dado"].sum()), "dias_estresse": int((jan["ks"] < 0.999).sum()),
        "excedente": float(jan["excedente"].sum()), "eta": float(jan["eta"].sum(min_count=1)),
        "etc": float(jan["etc"].sum(min_count=1))}
    if frac_sem > 0.10:
        return None, info, cenario, (f"{info['dias_sem_dado']} dias da janela sem chuva ou ET₀ conhecidas "
                                     "(mais de 10%) — armazenamento não apresentado")
    return jan, info, cenario, ""


# ------------------------------------------------------------------ orquestração
def analisar(power: pd.DataFrame | None, chirps: pd.Series | None, lat: float, altitude: float,
             par: ParametrosAgua, fim: date, nisar_res=None) -> Agua | None:
    """Monta o resultado a partir das séries já obtidas (função pura — usada também nos testes)."""
    if power is None and chirps is None:
        return None
    ini_jan = fim - timedelta(days=JANELA - 1)
    ini_longo = ini_jan - timedelta(days=AQUECIMENTO)
    longa = montar_serie(power, chirps, lat, altitude, ini_longo, fim)
    s = longa[longa.index >= pd.Timestamp(ini_jan)].copy()
    a = Agua(s, par, ini_jan, fim, resumir(s), nisar=nisar_res)
    arm, info, cenario, motivo = armazenamento(longa, par, ini_jan)
    a.armazenamento, a.kc_info, a.cenario, a.motivo_sem_armazenamento = arm, info, cenario, motivo
    a.cad_total = info.get("cad_total", float("nan"))
    r = a.resumo
    a.fontes = []
    if r["dias_chirps"]:
        a.fontes.append(f"Chuva: CHIRPS 2.0 (UCSB/CHC, ~5 km) em {r['dias_chirps']} dias")
    if r["dias_power"]:
        a.fontes.append(f"Chuva: NASA POWER (PRECTOTCORR, ~50 km) em {r['dias_power']} dias"
                        + (" — dias ainda não publicados no CHIRPS" if r["dias_chirps"] else ""))
    if r["dias_pm"] or r["dias_hs"]:
        a.fontes.append(f"ET₀: Penman-Monteith FAO-56 em {r['dias_pm']} dias"
                        + (f" e Hargreaves-Samani em {r['dias_hs']} (sem radiação, umidade ou vento)"
                           if r["dias_hs"] else "") + " · dados NASA POWER (~50 km)")
    sem_chuva, sem_et0 = r["dias"] - r["dias_chuva"], r["dias"] - r["dias_et0"]
    if sem_chuva:
        a.avisos.append(f"{sem_chuva} dia(s) sem dado de chuva na janela (não contados como zero)")
    if sem_et0:
        a.avisos.append(f"{sem_et0} dia(s) sem ET₀ na janela (não contados como zero)")
    if arm is not None and info["faixa_fim"] > 0.15 * info["cad_total"]:
        a.avisos.append("O armazenamento ainda depende da condição inicial assumida (faixa larga no fim da janela)")
    if nisar_res is not None:
        a.avisos += list(getattr(nisar_res, "avisos", []))
    return a


def obter(area: Area, par: ParametrosAgua | None = None, credencial_earthdata: dict | None = None,
          altitude: float | None = None, fim: date | None = None) -> Agua | None:
    """Busca as séries públicas e o NISAR. `altitude` (m): se não vier, usa a da célula da NASA POWER ou 400 m."""
    from concurrent.futures import ThreadPoolExecutor

    from . import nisar
    par = par or ParametrosAgua()
    lon, lat = area.centro
    hoje = fontes.hoje()
    ini = hoje - timedelta(days=JANELA + AQUECIMENTO + 12)

    def seguro(f):
        try:
            return f()
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(3) as ex:
        f_p = ex.submit(seguro, lambda: fontes.power_diario(lon, lat, ini, hoje - timedelta(days=1), POWER_ET0))
        f_c = ex.submit(seguro, lambda: _chirps_recente(lon, lat, ini, hoje))
        f_n = ex.submit(seguro, lambda: nisar.obter(area, JANELA, hoje, credencial_earthdata))
        power, chirps, nis = f_p.result(), f_c.result(), f_n.result()
    if power is None and chirps is None:
        return None
    if nis is None:
        nis = nisar.Nisar("erro", f"{nisar.SEM_OBS} — a consulta ao NISAR falhou.")
    # fim da janela = último dia com temperatura (ET₀) ou, na falta, com chuva
    if fim is None:
        if power is not None and power["T2M_MAX"].notna().any():
            fim = power.index[power["T2M_MAX"].notna()].max().date()
        else:
            fim = (chirps.index.max() if chirps is not None else pd.Timestamp(hoje)).date()
    return analisar(power, chirps, lat, 400.0 if altitude is None else altitude, par, fim, nis)
