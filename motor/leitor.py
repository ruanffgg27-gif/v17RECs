"""Leitura de laudos em Excel.

Encontra a linha de cabeçalho, reconhece as colunas por nome (tolerando
acentos, espaços, maiúsculas e variações comuns), lê a linha de unidades
quando existe e converte tudo para as unidades padrão do motor:
Ca, Mg, K, H+Al, Al → mmolc/dm³ · P, S → mg/dm³ · Argila → g/kg.
"""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from . import textura

# campo padrão → nomes aceitos (já normalizados: minúsculas, sem acento, só letras/números)
ALIASES: dict[str, list[str]] = {
    "n_lab": ["nlab", "nolab", "numlab", "numerolab", "nlaboratorio", "numerolaboratorio", "registro", "nregistro"],
    "proprietario": ["proprietario", "produtor", "cliente"],
    "propriedade": ["propriedade", "fazenda"],
    "identificacao": ["identificacao", "amostra", "idamostra", "identificacaodaamostra", "descricao"],
    "talhao": ["talhao", "gleba", "area", "lote"],
    "profundidade": ["profundidade", "prof", "profundidadecm", "camada"],
    "p": ["presina", "presin", "p", "fosforo", "presinatrocaionica", "pmehlich", "pmehlich1", "pmehlichi", "pmeh",
          "pm1", "mehlich", "mehlich1", "pmelich", "pmelich1", "fosforomehlich", "fosforoResina".lower()],
    "k": ["k", "potassio"],
    "ca": ["ca", "calcio"],
    "mg": ["mg", "magnesio"],
    "hal": ["hal", "acidezpotencial"],
    "al": ["al", "aluminio"],
    "s": ["sso4", "s", "enxofre", "so4", "ssulfato"],
    "argila": ["argila"],
    # usados só na interpretação
    "ph": ["phcacl2", "ph", "phcacl", "phcac12", "phcloretodecalcio", "phh2o", "phagua", "phemagua", "phh20"],
    "mo": ["mo", "materiaorganica", "mos"],
    "sb": ["sb", "somadebases"],
    "ctc": ["ctc", "ctcph7", "ctctotal"],
    "v": ["v", "saturacaoporbases", "vpct"],
    "m": ["m", "saturacaoporaluminio", "saturacaoporal"],
    "prem": ["prem", "premanescente", "fosfororemanescente"],
    "b": ["b", "boro"],
    "zn": ["zn", "zinco"],
    "cu": ["cu", "cobre"],
    "fe": ["fe", "ferro"],
    "mn": ["mn", "manganes"],
    "areia": ["areia", "areiatotal"],
    "silte": ["silte"],
}

ROTULOS = {
    "n_lab": "N Lab", "proprietario": "Proprietário", "propriedade": "Propriedade",
    "identificacao": "Identificação", "talhao": "Talhão", "profundidade": "Profundidade (cm)",
    "p": "P resina (mg/dm³)", "k": "K (mmolc/dm³)", "ca": "Ca (mmolc/dm³)",
    "mg": "Mg (mmolc/dm³)", "hal": "H+Al (mmolc/dm³)", "al": "Al (mmolc/dm³)",
    "s": "S-SO4 (mg/dm³)", "argila": "Argila (g/kg)",
    "ph": "pH CaCl2", "mo": "M.O. (g/dm³)", "sb": "SB (mmolc/dm³)", "ctc": "CTC (mmolc/dm³)",
    "v": "V%", "m": "m%", "prem": "P-rem (mg/L)", "b": "B (mg/dm³)", "zn": "Zn (mg/dm³)",
    "cu": "Cu (mg/dm³)", "fe": "Fe (mg/dm³)", "mn": "Mn (mg/dm³)", "areia": "Areia (g/kg)",
    "silte": "Silte (g/kg)",
}
NUMERICOS = ["p", "k", "ca", "mg", "hal", "al", "s", "argila",
             "ph", "mo", "sb", "ctc", "v", "m", "prem", "b", "zn", "cu", "fe", "mn", "areia", "silte"]
BASICOS = ["p", "k", "ca", "mg", "hal", "al", "s", "argila"]


def camada(prof) -> tuple[float | None, float | None]:
    """Início e fim da camada em CENTÍMETROS: '0-20', '0 a 20', '20/40', '20–40 cm' → (0, 20);
    '0,20-0,40 m' ou '0.2-0.4' → (20, 40). Sem número → (None, None)."""
    txt = str(prof or "").lower()
    nums = re.findall(r"\d+(?:[.,]\d+)?", txt)
    if not nums:
        return None, None
    a = float(nums[0].replace(",", "."))
    b = float(nums[1].replace(",", ".")) if len(nums) > 1 else None
    em_metros = (re.search(r"\d\s*m\b", txt) and "cm" not in txt) or (b is not None and b <= 1.5 and a < b)
    if em_metros:
        a, b = a * 100, (b * 100 if b is not None else None)
    return round(a, 1), (round(b, 1) if b is not None else None)


def classe_camada(prof) -> str:
    """'0-20' | 'sub' (começa em ≥ 20 cm) | 'sem' (profundidade vazia) | 'outra' (0-10, 10-20, 0-40…)."""
    topo, base = camada(prof)
    if topo is None:
        return "sem"
    if base is None:                       # um número só: '20' = 0-20; '40' = 20-40
        return "0-20" if topo == 20 else ("sub" if topo == 40 else "outra")
    if topo == 0 and base == 20:
        return "0-20"
    if topo >= 20:
        return "sub"
    return "outra"


def metodo_p(cabecalho: str) -> str:
    n = normalizar(cabecalho)
    if "mehlich" in n or "melich" in n or n in ("pm1", "pmeh"):
        return "Mehlich-1"
    if "resin" in n:
        return "resina"
    return "não informado"


def metodo_ph(cabecalho: str) -> str:
    n = normalizar(cabecalho)
    if "cacl" in n or "cac12" in n or "cloreto" in n:
        return "CaCl₂"
    if "h2o" in n or "agua" in n or "h20" in n:
        return "água"
    return "não informado"


def e_subsuperficial(prof, topo_minimo: float = 20) -> bool:
    """Camada que começa em ≥ 20 cm (ex.: 20-40) não entra na recomendação."""
    topo, _ = camada(prof)
    return topo is not None and topo >= topo_minimo
OBRIGATORIOS = ["ca", "mg", "hal"]


def normalizar(txt) -> str:
    if txt is None or (isinstance(txt, float) and pd.isna(txt)):
        return ""
    s = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _mapear(cabecalho: list) -> dict[str, int]:
    """campo → índice da coluna. Primeira ocorrência vence."""
    lookup = {alias: campo for campo, lista in ALIASES.items() for alias in lista}
    mapa: dict[str, int] = {}
    for i, nome in enumerate(cabecalho):
        campo = lookup.get(normalizar(nome))
        if campo and campo not in mapa:
            mapa[campo] = i
    return mapa


def _parece_unidade(celula) -> bool:
    n = normalizar(celula)
    return bool(n) and any(t in n for t in ("mmol", "cmol", "mgdm", "gdm", "gkg", "dag", "meq")) \
        or str(celula).strip() in ("%", "-")


@dataclass
class Laudo:
    dados: pd.DataFrame                  # colunas padrão (n_lab, ..., argila) + _linha
    original: pd.DataFrame               # planilha como veio (para anexar na saída)
    colunas: dict[str, str]              # campo → nome original da coluna
    unidades: dict[str, str]             # campo → unidade lida
    avisos: list[str] = field(default_factory=list)
    aba: str = ""
    linha_cab: int = 0      # índice (0-based) da linha de cabeçalho na planilha
    inicio: int = 1         # índice (0-based) da primeira linha de dados
    origens: list = field(default_factory=list)   # [(arquivo, Laudo)] quando vários laudos foram juntados
    ausentes: list = field(default_factory=list)  # atributos que não vieram no laudo (ex.: argila, micros)
    argila_metodo: str | None = None               # como a argila foi estimada (None = toda medida)
    avisos_dados: list = field(default_factory=list)   # mensagens sobre dados ausentes/estimados
    metodos: dict = field(default_factory=dict)        # {"p": "resina"|"Mehlich-1"|"não informado", "ph": …}
    fora_camada: pd.DataFrame | None = None            # amostras de outras camadas (0-10, 0-40…), fora das contas

    def __setstate__(self, estado):
        """Compatível com laudos guardados em cache por versões anteriores (campos novos ganham o padrão)."""
        padrao = {"origens": [], "ausentes": [], "argila_metodo": None, "avisos_dados": [], "avisos": [],
                  "aba": "", "linha_cab": 0, "inicio": 1, "metodos": {}, "fora_camada": None}
        self.__dict__.update({**padrao, **estado})


def _converter(campo: str, serie: pd.Series, unidade: str, avisos: list[str]) -> pd.Series:
    u = normalizar(unidade)
    rot = ROTULOS[campo]
    if campo in ("ca", "mg", "k", "hal", "al", "sb", "ctc"):
        if "cmol" in u or "meq100" in u:
            avisos.append(f"{rot}: convertido de cmolc/dm³ para mmolc/dm³ (×10)")
            return serie * 10
        if campo == "k" and "mgdm" in u:
            avisos.append("K: convertido de mg/dm³ para mmolc/dm³ (÷39,1)")
            return serie / 39.1
    if campo == "mo" and (str(unidade).strip() == "%" or "dagkg" in u or "dagdm" in u):
        avisos.append("M.O.: convertida de %/dag para g/dm³ (×10)")
        return serie * 10
    if campo in ("argila", "areia", "silte"):
        if str(unidade).strip() == "%" or "dagkg" in u:
            avisos.append(f"{rot}: convertida de %/dag/kg para g/kg (×10)")
            return serie * 10
    return serie


def ler_laudo(arquivo, nome: str = "") -> Laudo:
    """Lê um laudo .xlsx/.xls/.csv (caminho ou arquivo em memória)."""
    if isinstance(arquivo, (bytes, bytearray)):
        arquivo = io.BytesIO(arquivo)
    nome = nome or getattr(arquivo, "name", str(arquivo))
    if str(nome).lower().endswith(".csv"):
        abas = {"csv": pd.read_csv(arquivo, header=None, sep=None, engine="python", decimal=",")}
    else:
        abas = pd.read_excel(arquivo, header=None, sheet_name=None)

    for aba, bruto in abas.items():
        for linha in range(min(20, len(bruto))):
            mapa = _mapear(list(bruto.iloc[linha]))
            if all(c in mapa for c in OBRIGATORIOS) and len(mapa) >= 4:
                return _montar(bruto, linha, mapa, str(aba))
    raise ValueError(
        "Não encontrei o cabeçalho do laudo. A planilha precisa ter uma linha com os "
        "títulos das colunas, incluindo pelo menos Ca, Mg e H+Al."
    )


def _montar(bruto: pd.DataFrame, linha_cab: int, mapa: dict[str, int], aba: str) -> Laudo:
    avisos: list[str] = []
    cab = list(bruto.iloc[linha_cab])
    inicio = linha_cab + 1
    unidades = {c: "" for c in mapa}
    if inicio < len(bruto):
        prox = bruto.iloc[inicio]
        n_unid = sum(_parece_unidade(prox.iloc[i]) for i in mapa.values())
        if n_unid >= 2:
            unidades = {c: str(prox.iloc[i]).strip() if pd.notna(prox.iloc[i]) else "" for c, i in mapa.items()}
            inicio += 1

    corpo = bruto.iloc[inicio:].reset_index(drop=True)
    dados = pd.DataFrame({c: corpo.iloc[:, i] for c, i in mapa.items()})
    dados["_linha"] = range(inicio + 1, inicio + 1 + len(corpo))   # linha no Excel (1-based)

    for c in NUMERICOS:
        if c in dados:
            dados[c] = pd.to_numeric(
                dados[c].astype(str).str.replace(",", ".", regex=False).str.strip()
                .replace({"": None, "nan": None, "None": None, "-": None}),
                errors="coerce",
            )
            dados[c] = _converter(c, dados[c], unidades.get(c, ""), avisos)
        else:
            dados[c] = float("nan")

    # remove linhas vazias / rodapés (sem Ca, Mg e H+Al)
    validas = dados[OBRIGATORIOS].notna().all(axis=1)
    descartadas = int((~validas & dados[NUMERICOS].notna().any(axis=1)).sum())
    if descartadas:
        avisos.append(f"{descartadas} linha(s) ignorada(s) por não terem Ca, Mg e H+Al")
    dados = dados[validas].reset_index(drop=True)

    for c in ("n_lab", "proprietario", "propriedade", "identificacao", "talhao", "profundidade"):
        if c not in dados:
            dados[c] = ""
        dados[c] = dados[c].apply(lambda v: "" if pd.isna(v) else
                                  (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()))

    # camada: 0-20 entra nas contas; ≥ 20 cm é subsuperficial; outras camadas ficam de fora (com aviso)
    classes = dados["profundidade"].apply(classe_camada)
    if "profundidade" not in mapa:
        avisos.append("Laudo sem coluna de profundidade: todas as amostras foram consideradas 0-20 cm")
    elif (classes == "sem").all():
        avisos.append("Profundidade em branco em todas as amostras: consideradas 0-20 cm")
    elif (classes == "sem").any():
        ids = dados.loc[classes == "sem", "identificacao"].astype(str).tolist()
        avisos.append(f"{len(ids)} amostra(s) sem profundidade consideradas 0-20 cm: {', '.join(ids[:8])}"
                      + (" …" if len(ids) > 8 else "") + " — confira")
    fora = classes == "outra"
    fora_camada = None
    if fora.any():
        fora_camada = dados[fora].copy()
        cam = sorted(set(fora_camada["profundidade"].astype(str)))
        avisos.append(f"{int(fora.sum())} amostra(s) de outra camada ({', '.join(cam[:5])}) ficaram FORA das "
                      "recomendações e dos mapas — as regras são para 0-20 cm")
        dados = dados[~fora].reset_index(drop=True)
        classes = classes[~fora].reset_index(drop=True)
    dados["subsuperficial"] = (classes == "sub").values
    n_sub = int(dados["subsuperficial"].sum())
    if n_sub:
        avisos.append(f"{n_sub} amostra(s) de camada subsuperficial (ex.: 20-40 cm) identificadas: "
                      "aparecem destacadas, mas não entram nas recomendações")

    for c, rot in (("p", "P"), ("k", "K"), ("s", "S"), ("al", "Al")):
        if c not in mapa:
            avisos.append(f"Coluna de {rot} não encontrada no laudo")
    metodos = {"p": metodo_p(cab[mapa["p"]]) if "p" in mapa else "ausente",
               "ph": metodo_ph(cab[mapa["ph"]]) if "ph" in mapa else "ausente"}
    if metodos["p"] == "Mehlich-1":
        avisos.append("P por Mehlich-1: a regra de P₂O₅ e as classes de P da equipe são para P resina — a dose de "
                      "P não é calculada até você confirmar nos ajustes")
    elif metodos["p"] == "não informado":
        avisos.append(f"Método de P não informado na coluna “{cab[mapa['p']]}”: considerado P resina (confira)")
    if metodos["ph"] == "água":
        avisos.append("pH em água: as classes de pH da equipe são para pH em CaCl₂ (em água o valor costuma ser "
                      "0,5–0,7 maior)")

    # sinais de unidade trocada (só avisa, não altera)
    if not unidades.get("ca") and dados["ca"].median() < 10 and dados["hal"].median() < 10:
        avisos.append("Ca e H+Al muito baixos: confira se o laudo está em cmolc/dm³ (o sistema espera mmolc/dm³)")
    if "argila" in mapa and not unidades.get("argila") and dados["argila"].max() <= 100:
        avisos.append("Argila ≤ 100 em todas as amostras: confira se está em % (o sistema espera g/kg)")

    # dados ausentes: argila estimada pela CTC; demais só avisados
    faltam = textura.ausentes(dados, mapa)
    dados, metodo = textura.completar_argila(dados)
    sup = ~dados["subsuperficial"]
    n_est = int((dados["argila_estimada"] & sup).sum())
    msgs = textura.mensagem_ausentes(faltam, metodo, n_est, int(sup.sum()))

    original = bruto.copy()
    original.columns = [str(x) if pd.notna(x) else "" for x in cab]
    original = original.iloc[linha_cab + 1:].reset_index(drop=True)
    return Laudo(dados, original, {c: str(cab[i]) for c, i in mapa.items()}, unidades, avisos, aba,
                 linha_cab, inicio, ausentes=faltam, argila_metodo=metodo if n_est else None, avisos_dados=msgs,
                 metodos=metodos, fora_camada=fora_camada)
