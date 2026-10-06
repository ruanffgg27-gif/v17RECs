"""Junta laudos (arquivos Excel) da mesma fazenda num só — recomendações e book únicos."""
from __future__ import annotations

import re
import string
import unicodedata

import pandas as pd

from .leitor import Laudo

# palavras que não ajudam a identificar a fazenda/produtor
_VAZIAS = {"FAZENDA", "FAZ", "FZ", "SITIO", "SIT", "CHACARA", "ESTANCIA", "GRANJA", "AGROPECUARIA",
           "AGRO", "DA", "DE", "DO", "DAS", "DOS", "E", "LTDA", "ME", "SA", "S", "A", "EPP", "EIRELI"}


def normalizar_nome(txt) -> str:
    """'Faz. Santa Fé II' → 'SANTA FE II'; 'José da Silva' → 'JOSE SILVA'."""
    t = unicodedata.normalize("NFKD", str(txt or "")).encode("ascii", "ignore").decode().upper()
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    return " ".join(p for p in t.split() if p not in _VAZIAS)


def _mais_comum(serie: pd.Series) -> str:
    s = serie[serie.astype(str).str.strip() != ""]
    return str(s.mode().iloc[0]) if len(s) else ""


def identificacao(laudo: Laudo) -> tuple[str, str]:
    """(produtor, propriedade) mais frequentes no laudo."""
    d = laudo.dados
    return _mais_comum(d["proprietario"]), _mais_comum(d["propriedade"])


def _parecidos(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    pa, pb = set(a.split()), set(b.split())
    return len(pa & pb) / max(min(len(pa), len(pb)), 1) >= 0.67


def sugerir_grupos(laudos: list[Laudo]) -> list[int]:
    """Número de grupo (1, 2, …) para cada laudo: mesmo grupo = mesma fazenda.

    Critério: propriedade parecida (nome normalizado igual, contido no outro ou ≥ 2/3 das palavras em
    comum) e, quando os dois laudos informam o produtor, produtor também parecido. Laudos sem
    propriedade só se juntam pelo produtor.
    """
    ids = [tuple(normalizar_nome(x) for x in identificacao(l)) for l in laudos]
    grupo = list(range(len(laudos)))

    def raiz(i):
        while grupo[i] != i:
            i = grupo[i]
        return i

    for i in range(len(laudos)):
        for j in range(i):
            (pi, fi), (pj, fj) = ids[i], ids[j]
            if fi and fj:
                mesmo = _parecidos(fi, fj) and (not pi or not pj or _parecidos(pi, pj))
            else:
                mesmo = _parecidos(pi, pj) and not fi and not fj
            if mesmo:
                grupo[raiz(i)] = raiz(j)
    rotulos: dict[int, int] = {}
    return [rotulos.setdefault(raiz(i), len(rotulos) + 1) for i in range(len(laudos))]


def juntar(laudos: list[Laudo], nomes: list[str]) -> tuple[Laudo, list[str]]:
    """Um laudo só com as amostras de todos. Devolve (laudo, avisos).

    - A coluna `_arquivo` guarda de qual arquivo veio cada amostra (vai para a planilha).
    - Talhões com o mesmo nome em arquivos diferentes ganham a letra do arquivo ('1 (B)').
    - `original` guarda as planilhas originais de cada arquivo (uma aba cada na saída).
    """
    if len(laudos) == 1:
        return laudos[0], []
    avisos: list[str] = []
    letras = string.ascii_uppercase
    vistos: dict[str, int] = {}
    partes = []
    for i, (l, nome) in enumerate(zip(laudos, nomes)):
        d = l.dados.copy()
        d["_arquivo"] = nome
        d["_origem"] = i
        renomear = {}
        for t in dict.fromkeys(d["talhao"]):
            k = str(t).strip().upper()
            if k in vistos and vistos[k] != i:
                renomear[t] = f"{t} ({letras[i % 26]})"
            else:
                vistos.setdefault(k, i)
        if renomear:
            d["talhao"] = d["talhao"].replace(renomear)
            avisos.append(f"{nome}: talhão(ões) {', '.join(map(str, renomear))} já existiam em outro arquivo "
                          f"e foram renomeados para {', '.join(renomear.values())}")
        partes.append(d)
    dados = pd.concat(partes, ignore_index=True)
    colunas, unidades = {}, {}
    for l in laudos:
        colunas.update({k: v for k, v in l.colunas.items() if k not in colunas})
        unidades.update({k: v for k, v in l.unidades.items() if k not in unidades})
    av_leitura = [f"{n}: {a}" for l, n in zip(laudos, nomes) for a in l.avisos]
    junto = Laudo(dados=dados, original=laudos[0].original, colunas=colunas, unidades=unidades,
                  avisos=av_leitura + avisos, aba=" + ".join(dict.fromkeys(l.aba for l in laudos)),
                  linha_cab=laudos[0].linha_cab, inicio=laudos[0].inicio, origens=list(zip(nomes, laudos)),
                  ausentes=[c for c in laudos[0].ausentes if all(c in l.ausentes for l in laudos)],
                  argila_metodo=next((l.argila_metodo for l in laudos if l.argila_metodo), None),
                  avisos_dados=[f"{n}: {a}" for l, n in zip(laudos, nomes) for a in l.avisos_dados],
                  metodos=_metodos_juntos(laudos),
                  fora_camada=(pd.concat([l.fora_camada for l in laudos if getattr(l, "fora_camada", None) is not None],
                                         ignore_index=True)
                               if any(getattr(l, "fora_camada", None) is not None for l in laudos) else None))
    return junto, avisos


def _metodos_juntos(laudos) -> dict:
    """Métodos do laudo juntado: o mais restritivo vence (Mehlich-1 em qualquer arquivo → Mehlich-1)."""
    out = {}
    for k, ordem in (("p", ["Mehlich-1", "não informado", "resina", "ausente"]),
                     ("ph", ["água", "não informado", "CaCl₂", "ausente"])):
        vals = [(getattr(l, "metodos", {}) or {}).get(k) for l in laudos]
        vals = [v for v in vals if v]
        out[k] = next((o for o in ordem if o in vals), vals[0] if vals else "não informado")
    return out
