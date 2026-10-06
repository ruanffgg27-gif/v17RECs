"""Umidade superficial do solo do NISAR (NASA/ISRO) — produto oficial processado L3 SME2.

O que foi conferido nas fontes oficiais em 06/10/2026 (ASF DAAC — https://nisar-docs.asf.alaska.edu — e NASA CMR):

- Produto: "NISAR L3 Soil Moisture EASE-Grid 2.0" (SME2), coleção PROVISÓRIA `NISAR_L3_SME2_PROVISIONAL_V1`
  (CMR C2854344945-ASF, DOI 10.5067/NIL3SME2-P1), liberada em 20/07/2026. A documentação avisa que os produtos
  provisórios "ainda não estão totalmente calibrados"; a versão validada está prevista para o 4º trimestre de 2026.
- Grade: EASE-Grid 2.0 global (EPSG:6933), pixel de 200 m. Unidade: m³/m³ (umidade volumétrica).
- Repetição: ciclo de 12 dias, em média duas observações a cada 12 dias (órbitas ascendente e descendente).
- Camadas no arquivo HDF5 (grupo /science/LSAR/SME2/grids): soilMoisture, soilMoistureUncertainty,
  retrievalQualityFlag (uint16) e surfaceQualityFlag (int16).
- retrievalQualityFlag — "0 = sem problema; diferente de zero = anomalia": bit 0 recuperação recomendada; bit 1
  recuperação tentada; bit 2 recuperação bem-sucedida; bit 3 umidade entre 0,02 e 0,60; bits 4–7 anomalias de σ0.
- Não há recuperação sobre área urbana, água permanente, neve/gelo, solo congelado e chuva excessiva; acima de
  5 kg/m² de água na vegetação a recuperação é feita, mas sinalizada.
- Acesso: a busca no catálogo (CMR) é aberta; BAIXAR OU LER os arquivos exige login Earthdata (gratuito).

O que NÃO foi possível conferir (e por isso o código lê do próprio arquivo, sem assumir): o valor de preenchimento
(_FillValue), o significado dos bits de surfaceQualityFlag e os nomes dos vetores de coordenadas. A profundidade
de solo representada pela medida não consta nos documentos consultados — este módulo não afirma nenhuma.

Regras deste módulo:
- só o produto oficial, na resolução nativa (200 m), com a máscara de qualidade do próprio produto;
- o valor do talhão é a mediana dos pixels VÁLIDOS cujo centro cai dentro do perímetro; informa-se quantos pixels
  existem, quantos são válidos, quantos estão inteiros dentro do talhão (os demais são de borda e misturam o
  vizinho) — em talhões pequenos pode não haver nenhum pixel inteiro;
- a umidade é SUPERFICIAL: não é extrapolada para o perfil, não corrige o balanço hídrico e não é substituída por
  NDMI ou qualquer outro índice;
- sem observação válida (ou sem credencial para ler os arquivos): "Sem observação válida disponível".
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import requests

CMR_GRANULOS = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLECAO = "C2854344945-ASF"
PRODUTO = "NISAR L3 SME2 (Soil Moisture EASE-Grid 2.0)"
NOME_CURTO = "NISAR_L3_SME2_PROVISIONAL_V1"
MATURIDADE = "provisório (não totalmente calibrado)"
EPSG_GRADE = 6933
RES_M = 200.0
GRUPO = "/science/LSAR/SME2/grids"
SEM_OBS = "Sem observação válida disponível"
UA = {"User-Agent": "atria-satelites/1.0"}
BIT_RECOMENDADA, BIT_TENTADA, BIT_SUCESSO, BIT_FAIXA = 1, 2, 4, 8
LIMITE_MB = 80.0                 # teto de dados baixados por arquivo (leitura parcial por faixas de bytes)


@dataclass
class Observacao:
    data: datetime
    granulo: str
    passagem: str                      # "trilha 082 · descendente"
    n_pixels: int                      # pixels de 200 m com o centro dentro do talhão
    n_validos: int
    n_inteiros: int                    # pixels inteiramente dentro do talhão
    n_recomendados: int
    mediana: float = float("nan")      # m³/m³, pixels válidos
    p10: float = float("nan")
    p90: float = float("nan")
    incerteza: float = float("nan")    # mediana da incerteza do produto (m³/m³)
    sinal_superficie: float = float("nan")   # fração dos pixels com surfaceQualityFlag ≠ 0
    grade: dict | None = field(default=None, repr=False)   # {"x","y","sm","valido","dentro"} da janela lida

    @property
    def cobertura(self) -> float:
        return 100 * self.n_validos / self.n_pixels if self.n_pixels else 0.0

    @property
    def valida(self) -> bool:
        return self.n_validos > 0 and np.isfinite(self.mediana)


@dataclass
class Nisar:
    situacao: str                      # "ok" | "sem_credencial" | "sem_granulo" | "sem_valido" | "erro"
    mensagem: str = ""
    catalogados: int = 0               # passagens no catálogo no período (não exige login)
    mais_recente_catalogo: datetime | None = None
    observacoes: list[Observacao] = field(default_factory=list)   # da mais antiga para a mais recente
    avisos: list[str] = field(default_factory=list)

    @property
    def ultima_valida(self) -> Observacao | None:
        v = [o for o in self.observacoes if o.valida]
        return v[-1] if v else None


# ------------------------------------------------------------------ catálogo (aberto)
def _passagem(nome: str) -> tuple[str, str]:
    """Do nome do grânulo: (chave ciclo-trilha-direção, rótulo). Formato oficial:
    NISAR_L3_PR_SME2_<ciclo>_<trilha>_<A|D>_<quadro>_…"""
    p = nome.split("_")
    try:
        i = p.index("SME2")
        ciclo, trilha, direcao = p[i + 1], p[i + 2], p[i + 3]
        return f"{ciclo}-{trilha}-{direcao}", f"trilha {trilha} · {'ascendente' if direcao == 'A' else 'descendente'}"
    except (ValueError, IndexError):
        return nome, ""


def buscar_granulos(bbox, inicio: date, fim: date, timeout: float = 40, pedir=None) -> list[dict]:
    """Grânulos do período que tocam a caixa (lon0, lat0, lon1, lat1), do mais recente para o mais antigo:
    [{nome, data, url, chave, passagem}]. Não exige login."""
    params = {"collection_concept_id": COLECAO, "bounding_box": ",".join(f"{v:.5f}" for v in bbox),
              "temporal": f"{inicio.isoformat()}T00:00:00Z,{fim.isoformat()}T23:59:59Z",
              "sort_key": "-start_date", "page_size": 200}
    if pedir is not None:
        js = pedir(params)
    else:
        r = requests.get(CMR_GRANULOS, params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        js = r.json()
    out = []
    for e in js.get("feed", {}).get("entry", []):
        nome = e.get("producer_granule_id") or e.get("title") or ""
        nome = nome[:-3] if nome.endswith(".h5") else nome
        url = next((l["href"] for l in e.get("links", [])
                    if l.get("href", "").startswith("https://") and l["href"].endswith(".h5")
                    and "QA_STATS" not in l["href"]), None)
        try:
            quando = datetime.fromisoformat(e["time_start"].replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:  # noqa: BLE001
            continue
        if not url:
            continue
        chave, rot = _passagem(nome)
        out.append({"nome": nome, "data": quando, "url": url, "chave": chave, "passagem": rot})
    return out


# ------------------------------------------------------------------ leitura remota (exige login Earthdata)
class SessaoEarthdata(requests.Session):
    """Sessão que mantém a credencial só no servidor de login da NASA (urs.earthdata.nasa.gov) ao seguir os
    redirecionamentos — padrão recomendado pela NASA para baixar dados com usuário e senha."""
    HOST = "urs.earthdata.nasa.gov"

    def __init__(self, usuario: str = "", senha: str = "", token: str = ""):
        super().__init__()
        self.headers.update(UA)
        self._token = token
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        elif usuario:
            self.auth = (usuario, senha)

    def rebuild_auth(self, prepared_request, response):
        from urllib.parse import urlparse
        h = prepared_request.headers
        if "Authorization" in h:
            orig = urlparse(response.request.url).hostname
            dest = urlparse(prepared_request.url).hostname
            if orig != dest and dest != self.HOST and orig != self.HOST:
                del h["Authorization"]


class ArquivoRemoto(io.RawIOBase):
    """Arquivo somente-leitura sobre HTTP com pedidos por faixa de bytes e cache em blocos — o HDF5 é lido aos
    pedaços (metadados + a janela do talhão), sem baixar o arquivo inteiro (30–130 MB)."""

    def __init__(self, url: str, sessao, bloco: int = 2 * 1024 * 1024, limite_mb: float = LIMITE_MB, timeout=90):
        super().__init__()
        self._s, self._bloco, self._timeout, self._origem = sessao, bloco, timeout, url
        self._cache: dict[int, bytes] = {}
        self._pos = 0
        self.baixado = 0
        self._limite = limite_mb * 1024 * 1024
        # 1º pedido: segue os redirecionamentos (login) e guarda o endereço final e o tamanho
        r = sessao.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, allow_redirects=True)
        if r.status_code in (401, 403):
            raise PermissionError("acesso negado pelo Earthdata (credencial inválida ou sem autorização para o "
                                  "aplicativo de dados da ASF)")
        r.raise_for_status()
        faixa = r.headers.get("Content-Range", "")
        if r.status_code != 206 or "/" not in faixa:
            raise OSError("o servidor não aceitou leitura por faixas de bytes")
        self._tam = int(faixa.rsplit("/", 1)[1])
        self._url = r.url
        from urllib.parse import urlparse
        # endereço final assinado (outro servidor): os pedidos seguintes vão direto, SEM a credencial
        self._direto = urlparse(r.url).hostname != urlparse(url).hostname

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        self._pos = offset if whence == io.SEEK_SET else (self._pos + offset if whence == io.SEEK_CUR
                                                          else self._tam + offset)
        return self._pos

    def _pedir(self, a: int, b: int) -> bytes:
        if self.baixado + (b - a + 1) > self._limite:
            raise OSError(f"leitura interrompida: mais de {self._limite / 1048576:.0f} MB para um só arquivo")
        cab = {"Range": f"bytes={a}-{b}"}
        if self._direto:
            r = requests.get(self._url, headers={**UA, **cab}, timeout=self._timeout)
            if r.status_code in (400, 401, 403):         # assinatura vencida: refaz pelo caminho com login
                r = self._s.get(self._origem, headers=cab, timeout=self._timeout, allow_redirects=True)
                self._url = r.url
        else:
            r = self._s.get(self._url, headers=cab, timeout=self._timeout)
        r.raise_for_status()
        self.baixado += len(r.content)
        return r.content

    def readinto(self, buf):
        n = min(len(buf), self._tam - self._pos)
        if n <= 0:
            return 0
        ini, fim = self._pos, self._pos + n
        b0, b1 = ini // self._bloco, (fim - 1) // self._bloco
        faltam = [k for k in range(b0, b1 + 1) if k not in self._cache]
        if faltam:                                   # blocos contíguos num pedido só
            a, z = faltam[0], faltam[-1]
            dados = self._pedir(a * self._bloco, min((z + 1) * self._bloco, self._tam) - 1)
            for k in range(a, z + 1):
                self._cache[k] = dados[(k - a) * self._bloco:(k - a + 1) * self._bloco]
        partes = b"".join(self._cache[k] for k in range(b0, b1 + 1))
        d = ini - b0 * self._bloco
        buf[:n] = partes[d:d + n]
        self._pos = fim
        return n


def _coordenadas(g) -> tuple[np.ndarray, np.ndarray]:
    """Vetores x e y (m, EPSG:6933) do grupo de grades. Os nomes não puderam ser conferidos na especificação
    atual; procura os usuais e, na falta, qualquer vetor 1-D com o tamanho das colunas/linhas."""
    sm = g["soilMoisture"]
    lin, col = sm.shape[-2], sm.shape[-1]
    x = y = None
    for nome in ("xCoordinates", "x", "x_coordinates", "easting"):
        if nome in g and g[nome].shape == (col,):
            x = g[nome][...]
            break
    for nome in ("yCoordinates", "y", "y_coordinates", "northing"):
        if nome in g and g[nome].shape == (lin,):
            y = g[nome][...]
            break
    if x is None or y is None:
        for nome, d in g.items():
            if getattr(d, "ndim", 0) == 1 and np.issubdtype(d.dtype, np.floating):
                if x is None and d.shape == (col,) and (lin != col or "x" in nome.lower()):
                    x = d[...]
                elif y is None and d.shape == (lin,) and (lin != col or "y" in nome.lower()):
                    y = d[...]
    if x is None or y is None:
        raise KeyError("coordenadas da grade não encontradas no arquivo")
    return np.asarray(x, float), np.asarray(y, float)


def _preenchimento(d):
    fv = d.attrs.get("_FillValue")
    if fv is None and getattr(d, "fillvalue", None) is not None:
        fv = d.fillvalue
    try:
        return float(np.ravel(fv)[0])
    except Exception:  # noqa: BLE001
        return None


def extrair(arquivo, poligono_6933, granulo: dict | None = None) -> Observacao | None:
    """Lê a janela do talhão de um arquivo SME2 aberto (h5py.File ou caminho/objeto de arquivo) e resume os
    pixels. Devolve None se o talhão não tem nenhum pixel dentro deste grânulo."""
    import h5py
    from shapely import contains_xy
    from shapely.geometry import box
    fechar = not isinstance(arquivo, h5py.File)
    f = h5py.File(arquivo, "r") if fechar else arquivo
    try:
        g = f[GRUPO]
        x, y = _coordenadas(g)
        res = float(abs(np.median(np.diff(x)))) if len(x) > 1 else RES_M
        x0, y0, x1, y1 = poligono_6933.bounds
        cols = np.where((x >= x0 - res) & (x <= x1 + res))[0]
        lins = np.where((y >= y0 - res) & (y <= y1 + res))[0]
        if not len(cols) or not len(lins):
            return None
        c0, c1, l0, l1 = cols.min(), cols.max() + 1, lins.min(), lins.max() + 1
        xs, ys = x[c0:c1], y[l0:l1]

        def janela(nome):
            if nome not in g:
                return None
            d = g[nome]
            a = d[..., l0:l1, c0:c1]
            return np.asarray(a).reshape(l1 - l0, c1 - c0), _preenchimento(d)
        sm, fv = janela("soilMoisture")
        sm = sm.astype(float)
        if fv is not None:
            sm[np.isclose(sm, fv)] = np.nan
        X, Y = np.meshgrid(xs, ys)
        dentro = contains_xy(poligono_6933, X, Y)
        if not dentro.any():
            return None
        q = janela("retrievalQualityFlag")
        flag = q[0].astype(np.int64) if q is not None else np.zeros(sm.shape, np.int64)
        # válido: recuperação tentada, bem-sucedida e dentro de 0,02–0,60 (bits 1–3 = 0), valor finito
        valido = np.isfinite(sm) & ((flag & (BIT_TENTADA | BIT_SUCESSO | BIT_FAIXA)) == 0) & (sm >= 0.02) & (sm <= 0.60)
        recomendado = valido & ((flag & BIT_RECOMENDADA) == 0)
        inteiro = np.zeros(sm.shape, bool)
        for i, j in zip(*np.where(dentro)):
            inteiro[i, j] = poligono_6933.contains(box(X[i, j] - res / 2, Y[i, j] - res / 2,
                                                       X[i, j] + res / 2, Y[i, j] + res / 2))
        sel = dentro & valido
        u = janela("soilMoistureUncertainty")
        s = janela("surfaceQualityFlag")
        gr = granulo or {}
        o = Observacao(gr.get("data", datetime(1900, 1, 1)), gr.get("nome", ""), gr.get("passagem", ""),
                       int(dentro.sum()), int(sel.sum()), int((dentro & inteiro).sum()),
                       int((dentro & recomendado).sum()),
                       grade={"x": xs, "y": ys, "sm": np.where(valido, sm, np.nan), "valido": valido,
                              "dentro": dentro, "inteiro": inteiro, "res": res})
        if sel.any():
            v = sm[sel]
            o.mediana, o.p10, o.p90 = float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))
            if u is not None:
                inc = u[0].astype(float)[sel]
                inc = inc[np.isfinite(inc) & (inc >= 0) & (inc < 1)]
                o.incerteza = float(np.median(inc)) if len(inc) else float("nan")
        if s is not None:
            o.sinal_superficie = float((s[0][dentro] != 0).mean())
        return o
    finally:
        if fechar:
            f.close()


def obter(area, dias: int = 90, fim: date | None = None, credencial: dict | None = None, max_passagens: int = 8,
          buscar=None, abrir=None) -> Nisar:
    """Observações NISAR do talhão nos últimos `dias`. `credencial`: {"token": …} ou {"usuario": …, "senha": …}
    (login Earthdata). `buscar`/`abrir` permitem injetar fontes nos testes."""
    from ..geo import projetar
    fim = fim or date.today()
    inicio = fim - timedelta(days=dias)
    try:
        granulos = (buscar or buscar_granulos)(area.bbox_ll(RES_M), inicio, fim)
    except Exception as e:  # noqa: BLE001
        return Nisar("erro", f"{SEM_OBS} — o catálogo da NASA (CMR) não respondeu ({type(e).__name__}).")
    if not granulos:
        return Nisar("sem_granulo", f"{SEM_OBS} — nenhuma passagem do NISAR catalogada sobre o talhão entre "
                                    f"{inicio:%d/%m/%Y} e {fim:%d/%m/%Y}.")
    # uma passagem pode vir em vários arquivos (quadros vizinhos, coberturas parciais): agrupa
    passagens: dict[str, list[dict]] = {}
    for gnl in granulos:
        passagens.setdefault(gnl["chave"], []).append(gnl)
    n = Nisar("sem_valido", catalogados=len(passagens), mais_recente_catalogo=granulos[0]["data"])
    cred = credencial or {}
    if abrir is None and not (cred.get("token") or cred.get("usuario")):
        n.situacao = "sem_credencial"
        n.mensagem = (f"{SEM_OBS} — há {len(passagens)} passagem(ns) do NISAR catalogada(s) sobre o talhão (a mais "
                      f"recente em {granulos[0]['data']:%d/%m/%Y}), mas a leitura dos arquivos exige login Earthdata "
                      "(gratuito), que não está configurado no app.")
        return n
    try:
        import h5py  # noqa: F401
    except ImportError:
        n.situacao, n.mensagem = "erro", f"{SEM_OBS} — biblioteca h5py não instalada no servidor."
        return n
    pol = projetar(area.poligono_ll, EPSG_GRADE)

    def abrir_remoto(gnl):                       # uma sessão por arquivo: as passagens são lidas em paralelo
        return ArquivoRemoto(gnl["url"], SessaoEarthdata(cred.get("usuario", ""), cred.get("senha", ""),
                                                         cred.get("token", "")))

    def ler_passagem(chave):
        """Melhor arquivo da passagem (mais pixels válidos no talhão) + erros de leitura."""
        melhor, erros, negado = None, [], None
        for gnl in passagens[chave][:3]:
            try:
                o = extrair((abrir or abrir_remoto)(gnl), pol, gnl)
            except PermissionError as e:
                negado = str(e)
                break
            except Exception as e:  # noqa: BLE001
                erros.append(f"{gnl['nome'][-40:]}: {type(e).__name__}")
                continue
            if o is not None and (melhor is None or o.n_validos > melhor.n_validos):
                melhor = o
        return melhor, erros, negado

    from concurrent.futures import ThreadPoolExecutor
    chaves = list(passagens)[:max_passagens]                 # já do mais recente para o mais antigo
    with ThreadPoolExecutor(min(4, len(chaves))) as ex:
        resultados = list(ex.map(ler_passagem, chaves))
    erros = []
    for melhor, errs, negado in resultados:
        if negado:
            n.situacao, n.mensagem = "erro", f"{SEM_OBS} — {negado}."
            return n
        erros += errs
        if melhor is not None:
            n.observacoes.append(melhor)
    n.observacoes.sort(key=lambda o: o.data)
    if erros:
        n.avisos.append(f"NISAR: {len(erros)} arquivo(s) não puderam ser lidos ({'; '.join(erros[:2])})")
    u = n.ultima_valida
    if u is None:
        n.situacao = "sem_valido"
        n.mensagem = (f"{SEM_OBS} — {len(n.observacoes)} passagem(ns) lida(s), nenhuma com pixel válido dentro do "
                      "talhão (máscara de qualidade do produto).") if n.observacoes else \
                     f"{SEM_OBS} — os arquivos das passagens catalogadas não puderam ser lidos."
        return n
    n.situacao = "ok"
    if u.n_inteiros == 0:
        n.avisos.append("NISAR: nenhum pixel de 200 m cabe inteiro no talhão — todos os valores são de borda e "
                        "incluem área vizinha")
    elif u.n_pixels < 6:
        n.avisos.append(f"NISAR: só {u.n_pixels} pixels de 200 m no talhão — valor pouco representativo")
    if u.cobertura < 50:
        n.avisos.append(f"NISAR: só {u.cobertura:.0f}% dos pixels do talhão são válidos na observação mais recente")
    return n
