"""ATRIA · Plataforma de recomendação — interface Streamlit.

Rodar localmente:  streamlit run app.py
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).parent / "assets"
VERDE, LARANJA, VERMELHO_SUB = "#104A2A", "#E25A10", "#FF8A8A"
APP_VERSAO = "17.0"          # tem de ser igual a motor/__init__.py → VERSAO

st.set_page_config(page_title="ATRIA · Recomendação de solo", page_icon=str(ASSETS / "selo.png"),
                   layout="wide")

try:
    from motor import VERSAO
except ImportError:
    VERSAO = "antiga"
try:
    from motor import (Ajustes, AjusteProduto, Parametros, gerar_excel, gerar_excel_interpretacao,
                       ler_laudo, recomendar)
    from motor import interpretacao as it
    from motor.processar import (COL_CAL, COL_GES, COL_K, COL_K1, COL_P, COL_S0, COL_SUB,
                                 colunas_doses, colunas_doses_totais, nome_produto_p)
    from motor.juntar import identificacao, juntar as juntar_laudos, sugerir_grupos
    from motor.regras import fatores_calcario, ler_formula
    erro_import = None
except Exception as e:  # noqa: BLE001
    erro_import = e
if erro_import is not None or VERSAO != APP_VERSAO:
    st.error(f"**Os arquivos do app estão em versões diferentes** (app.py = {APP_VERSAO}, pasta `motor` = {VERSAO}).\n\n"
             "No GitHub, a pasta **`motor`** precisa ser atualizada junto com o `app.py`: abra a pasta `motor` do "
             "repositório, clique em *Add file → Upload files* e arraste para lá **todos os arquivos da pasta "
             "`motor`** do .zip (os `.py`). Depois, em *Manage app → ⋮ → Reboot app*.")
    if erro_import is not None:
        st.caption(f"Detalhe técnico: {type(erro_import).__name__}: {erro_import}")
    st.stop()

# ------------------------------------------------------------------ visual
# Retrô analógico (anos 80/90): papel de formulário contínuo, contornos de tinta, teclas com sombra dura que
# "afundam" ao clicar e faixas de decalque nas cores da marca. Tudo o que é clicável tem contorno e sombra.
MOSTARDA = "#E9B23A"


# Duas paletas. Claro: papel de formulário, contornos de tinta. Escuro: "fliperama" — fundo quase preto, contornos
# em verde fluorescente, sombras em rosa-choque e destaques em amarelo; texto sempre claro sobre fundo escuro.
# A troca é feita PELO NAVEGADOR, com a função CSS light-dark(): o Streamlit marca o app com `color-scheme: light`
# ou `dark` conforme o menu ⋮ (System / Light / Dark) e cada cor escolhe o seu valor na hora — sem depender de o
# script rodar de novo quando o usuário muda o tema.
CLARO = {"tinta": "#1D2A22", "texto": "#1D2A22", "texto2": "#4D5B52", "titulo": VERDE, "fundo": "#F6F2E7",
         "cartao": "#FFFDF6", "campo": "#FFFFFF", "ficha": "#FBF8EE", "aba": "#E8E2CF", "aba_txt": "#1D2A22",
         "aba_ativa": VERDE, "aba_ativa_txt": "#FFFFFF", "destaque": MOSTARDA, "destaque_txt": "#1D2A22",
         "hover": "#FFF4D6", "primario": LARANJA, "primario_hover": "#C94A08", "primario_txt": "#FFFFFF",
         "sombra": "#1D2A22", "sombra2": "#DCEAD6", "tecla": "#EFE9D6", "foco": LARANJA,
         "lateral1": "#E6F0E1", "lateral2": "#DCEAD6", "picote": "rgba(29,42,34,.13)",
         "inativo": "#9AA39C", "valor": VERDE, "faixa1": VERDE, "faixa2": LARANJA, "faixa3": MOSTARDA,
         "etiqueta": "rgba(246,242,231,0)", "etiqueta_sombra": "rgba(0,0,0,0)"}
ESCURO = {"tinta": "#B8FF3C", "texto": "#F4F8EE", "texto2": "#C3D1C6", "titulo": "#7DFF6A", "fundo": "#0D1310",
          "cartao": "#17211B", "campo": "#060A08", "ficha": "#131B17", "aba": "#1D2922", "aba_txt": "#EAF3E4",
          "aba_ativa": "#B8FF3C", "aba_ativa_txt": "#0B120D", "destaque": "#FFE23D", "destaque_txt": "#11150F",
          "hover": "#263A2C", "primario": "#FF2E93", "primario_hover": "#FF5CAB", "primario_txt": "#FFFFFF",
          "sombra": "#FF2E93", "sombra2": "#7A1E55", "tecla": "#26342B", "foco": "#FFE23D",
          "lateral1": "#111A15", "lateral2": "#0B110E", "picote": "rgba(184,255,60,.22)",
          "inativo": "#5F6E63", "valor": "#FFE23D", "faixa1": "#7DFF6A", "faixa2": "#FF2E93", "faixa3": "#FFE23D",
          "etiqueta": "#F6F2E7", "etiqueta_sombra": "#FF2E93"}
_VARS_CLARO = "; ".join(f"--{k.replace('_', '-')}: {v}" for k, v in CLARO.items())
_VARS = "; ".join(f"--{k.replace('_', '-')}: light-dark({v}, {ESCURO[k]})" for k, v in CLARO.items())
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
  :root { __VARS_CLARO__; --sombra-dura: 4px 4px 0 var(--sombra); }
  @supports (color: light-dark(#000, #fff)) { :root { __VARS__; } }
  html, body, .stApp, [data-testid="stMarkdownContainer"], label, input, textarea, button {
      font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
  h1, h2, h3, h4, h5 { font-family: 'Rubik', 'Segoe UI', sans-serif; color: var(--titulo) !important;
      letter-spacing: -.01em; }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--fundo); }
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--texto); }
  [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] li,
  [data-testid="stCaptionContainer"] { color: var(--texto2) !important; }
  [data-testid="stMarkdownContainer"] code { color: var(--destaque-txt); background: var(--destaque);
      border-radius: 4px; padding: .05rem .3rem; }
  a { color: var(--primario); }

  /* título com faixas de decalque */
  .atria-titulo { font-family: 'Rubik', sans-serif; font-size: 2rem; font-weight: 800; color: var(--titulo);
                  margin: .7rem 0 0 0; line-height: 1.15; }
  .atria-sub { color: var(--texto2); margin: .25rem 0 .5rem 0; max-width: 62rem; }
  .atria-faixa { height: 14px; margin: .3rem 0 1.1rem 0; border-radius: 2px;
                 background: linear-gradient(to bottom, var(--faixa1) 0 5px, transparent 5px 7px,
                                             var(--faixa2) 7px 11px, transparent 11px 12px, var(--faixa3) 12px 14px); }

  /* barra lateral lisa, com a margem picotada bem discreta */
  [data-testid="stSidebar"] { border-right: 2px solid var(--tinta);
      background: linear-gradient(to bottom, var(--lateral1) 0%, var(--lateral2) 100%); }
  [data-testid="stSidebar"] > div:first-child {
      background: radial-gradient(circle at 9px 14px, var(--picote) 2.8px, transparent 3.2px) 0 0 / 18px 28px repeat-y; }
  [data-testid="stSidebar"] h2 { font-size: 1.1rem; }
  [data-testid="stCaptionContainer"] { opacity: 1 !important; }
  __LOGO__

  /* teclas: botões com contorno e sombra dura; afundam ao clicar */
  .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
  [data-testid="stPopover"] > div > button {
      border: 2px solid var(--tinta) !important; border-radius: 7px; box-shadow: var(--sombra-dura);
      font-family: 'Rubik', sans-serif; font-weight: 700; letter-spacing: .01em; background: var(--cartao);
      color: var(--texto); transition: transform .06s ease, box-shadow .06s ease; }
  .stButton > button p, .stDownloadButton > button p, [data-testid="stFormSubmitButton"] > button p,
  [data-testid="stPopover"] > div > button p { color: inherit !important; }
  .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover,
  [data-testid="stPopover"] > div > button:hover {
      transform: translate(-1px, -1px); box-shadow: 5px 5px 0 var(--sombra); background: var(--destaque);
      color: var(--destaque-txt); }
  .stButton > button:active, .stDownloadButton > button:active, [data-testid="stFormSubmitButton"] > button:active {
      transform: translate(3px, 3px); box-shadow: 1px 1px 0 var(--sombra); }
  button[kind="primary"], button[kind="primaryFormSubmit"] { background: var(--primario) !important;
      color: var(--primario-txt) !important; }
  button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
      background: var(--primario-hover) !important; color: var(--primario-txt) !important; }
  button:disabled { box-shadow: 2px 2px 0 var(--inativo) !important; border-color: var(--inativo) !important;
      opacity: .65; }
  button:focus-visible, input:focus-visible, [role="tab"]:focus-visible { outline: 3px solid var(--foco);
      outline-offset: 2px; }

  /* caixas de envio de arquivo: bandeja pontilhada, bem visível */
  [data-testid="stFileUploaderDropzone"] { background: var(--cartao); border: 2.5px dashed var(--tinta);
      border-radius: 8px; box-shadow: var(--sombra-dura); transition: background .1s ease, border-color .1s ease; }
  [data-testid="stFileUploaderDropzone"]:hover { background: var(--hover); border-color: var(--foco); }
  [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small,
  [data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--texto) !important; }
  [data-testid="stFileUploaderDropzone"] button { border: 2px solid var(--tinta) !important; border-radius: 6px;
      box-shadow: 3px 3px 0 var(--sombra); background: var(--destaque); color: var(--destaque-txt) !important;
      font-weight: 700; }
  [data-testid="stFileUploaderDropzone"] button * { color: var(--destaque-txt) !important; }
  [data-testid="stFileUploader"] label p { font-family: 'Rubik', sans-serif; font-weight: 700;
      color: var(--titulo); }
  [data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] { color: var(--texto) !important; }

  /* campos: contorno de tinta; foco destacado */
  [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
  [data-testid="stNumberInputContainer"] { background: var(--campo) !important;
      border-color: var(--tinta) !important; border-width: 2px !important; border-radius: 6px !important; }
  [data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within,
  [data-testid="stNumberInputContainer"]:focus-within { border-color: var(--foco) !important;
      box-shadow: 3px 3px 0 var(--foco); }
  [data-baseweb="base-input"] { border: 0 !important; }
  [data-testid="stNumberInput"] button { background: var(--tecla); border-left: 2px solid var(--tinta);
      color: var(--texto); }
  [data-testid="stNumberInput"] button:hover { background: var(--destaque); color: var(--destaque-txt); }
  [data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stTextInput"] input,
  [data-testid="stTextInputRootElement"], [data-testid="stSelectbox"] [data-baseweb="select"] > div,
  [data-testid="stMultiSelect"] [data-baseweb="select"] > div, [data-testid="stNumberInput"] input,
  [data-testid="stTextArea"] textarea { background: var(--campo) !important; color: var(--texto) !important;
      caret-color: var(--foco); }
  input::placeholder, textarea::placeholder { color: var(--texto2) !important; opacity: .8; }
  [data-testid="stSelectbox"] [data-baseweb="select"] div { background-color: var(--campo); color: var(--texto); }
  [data-baseweb="select"] svg { fill: var(--texto); }
  [data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [data-baseweb="menu"] {
      background: var(--cartao); border: 2px solid var(--tinta); }
  [data-baseweb="popover"] [role="option"] { color: var(--texto); }
  [data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [role="option"][aria-selected="true"] {
      background: var(--destaque); color: var(--destaque-txt); }
  [data-baseweb="tag"] { background: var(--destaque) !important; color: var(--destaque-txt) !important; }
  [data-baseweb="tag"] * { color: var(--destaque-txt) !important; fill: var(--destaque-txt) !important; }
  [data-testid="stWidgetLabel"] p { font-weight: 600; color: var(--texto); }
  [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
  [data-testid="stRadio"] [role="radiogroup"] p { font-weight: 500; color: var(--texto); }
  [data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg { stroke: var(--texto2); }

  /* abas: divisórias de fichário */
  [data-testid="stTabs"] [role="tablist"] { gap: 6px; border-bottom: 2px solid var(--tinta); padding-top: 4px; }
  [data-testid="stTab"] { border: 2px solid var(--tinta); border-bottom: 0; border-radius: 9px 9px 0 0;
      padding: .45rem 1rem; background: var(--aba); font-family: 'Rubik', sans-serif; cursor: pointer;
      color: var(--aba-txt); }
  [data-testid="stTab"] p { font-weight: 700; font-size: .95rem; color: var(--aba-txt); }
  [data-testid="stTab"]:hover { background: var(--destaque); color: var(--destaque-txt); }
  [data-testid="stTab"]:hover p { color: var(--destaque-txt); }
  [data-testid="stTab"][aria-selected="true"] { background: var(--aba-ativa); color: var(--aba-ativa-txt); }
  [data-testid="stTab"][aria-selected="true"] p { color: var(--aba-ativa-txt); }
  [data-testid="stTab"]::after, [data-testid="stTab"]::before { display: none !important; }

  /* formulários, expansores, métricas e tabelas: fichas com contorno */
  [data-testid="stForm"] { border: 2px solid var(--tinta); border-radius: 10px; background: var(--ficha);
      box-shadow: 6px 6px 0 var(--sombra2); }
  [data-testid="stExpander"] details { border: 2px solid var(--tinta); border-radius: 8px;
      background: var(--ficha); }
  [data-testid="stExpander"] summary { font-weight: 600; color: var(--texto); }
  [data-testid="stExpander"] summary p { color: inherit; }
  [data-testid="stExpander"] summary:hover { background: var(--hover); color: var(--texto); }
  [data-testid="stExpander"] summary svg { fill: var(--texto); }
  [data-testid="stMetric"] { border: 2px solid var(--tinta); border-radius: 8px; background: var(--cartao);
      padding: .55rem .8rem; box-shadow: 3px 3px 0 var(--sombra2); }
  [data-testid="stMetricLabel"] p { color: var(--texto2); }
  [data-testid="stMetricValue"] * { color: var(--valor) !important; }
  [data-testid="stMetricValue"] { color: var(--valor); font-family: 'IBM Plex Mono', monospace; font-weight: 600;
      font-size: 1.45rem; }
  [data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 2px solid var(--tinta); border-radius: 6px; }
  [data-testid="stAlert"] { border: 2px solid var(--tinta); border-radius: 8px; }
  [data-testid="stStatusWidget"], [data-testid="stStatus"] details { border-color: var(--tinta); }
  [data-testid="stCheckbox"] label, [data-testid="stRadio"] label { cursor: pointer; }
  hr { border-color: var(--tinta); opacity: .5; }

  .sub-legenda { display:inline-block; width:14px; height:14px; background:__SUB__;
                 border: 1.5px solid var(--tinta); border-radius:3px; vertical-align:middle; margin-right:6px; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>""".replace("__VARS__", _VARS).replace("__VARS_CLARO__", _VARS_CLARO).replace("__LOGO__", (
    # no escuro a marca (verde-escuro) ganha uma etiqueta de papel para continuar legível
    '[data-testid="stSidebarHeader"] img, [data-testid="stLogo"], [data-testid="stSidebarLogo"] '
    '{ background: var(--etiqueta); border-radius: 8px; padding: 5px 9px; '
    'box-shadow: 3px 3px 0 var(--etiqueta-sombra); box-sizing: content-box; }')).replace("__SUB__", VERMELHO_SUB), unsafe_allow_html=True)


def br(x, casas: int = 0) -> str:
    """Número com separador de milhar brasileiro."""
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def segredo(chave: str):
    try:
        return st.secrets.get(chave)
    except Exception:
        return None


# ------------------------------------------------------------ senha opcional
def liberado() -> bool:
    senha = segredo("senha")
    if not senha or st.session_state.get("ok"):
        return True
    st.image(str(ASSETS / "logo.png"), width=260)
    s = st.text_input("Senha de acesso", type="password")
    if s:
        if s == senha:
            st.session_state.ok = True
            st.rerun()
        st.error("Senha incorreta")
    return False


if not liberado():
    st.stop()

st.logo(str(ASSETS / "logo.png"), icon_image=str(ASSETS / "selo.png"), size="large")


# ------------------------------------------------------------------- música
PLAYLIST_PADRAO = "https://www.youtube.com/watch?v=JcrBHxWHX74&list=RDJcrBHxWHX74&start_radio=1"


def url_embed(link: str) -> str | None:
    """Converte link do YouTube (vídeo, playlist ou Mix) ou do Spotify em link de player incorporado."""
    link = (link or "").strip()
    video = re.search(r"(?:[?&]v=|youtu\.be/|/embed/|/shorts/)([\w-]{11})", link)
    lista = re.search(r"[?&]list=([\w-]+)", link)
    if video and lista:      # vídeo dentro de uma lista ou Mix (RD...): começa pelo vídeo e segue a lista
        return (f"https://www.youtube.com/embed/{video.group(1)}?list={lista.group(1)}"
                f"&autoplay=1&loop=1&rel=0")
    if lista:                # playlist comum
        return f"https://www.youtube.com/embed/videoseries?list={lista.group(1)}&autoplay=1&loop=1"
    if video:
        return f"https://www.youtube.com/embed/{video.group(1)}?autoplay=1&loop=1&playlist={video.group(1)}"
    m = re.search(r"open\.spotify\.com/(?:intl-\w+/)?(playlist|album|track)/(\w+)", link)
    if m:
        return f"https://open.spotify.com/embed/{m.group(1)}/{m.group(2)}"
    return None


with st.sidebar:
    musica = st.toggle("🎵 Trilha sonora", value=False, help="Liga/desliga a música de fundo")
    if musica:
        link = st.session_state.get("playlist_link") or segredo("playlist") or PLAYLIST_PADRAO
        emb = url_embed(link)
        if emb:
            altura = 152 if "spotify" in emb else 180
            components.html(
                f'<iframe src="{emb}" width="100%" height="{altura}" frameborder="0" '
                f'style="border-radius:10px" allow="autoplay; encrypted-media; clipboard-write" '
                f'allowfullscreen></iframe>', height=altura + 8)
        with st.expander("Trocar playlist", expanded=not emb):
            st.text_input("Link da playlist (YouTube ou Spotify)", key="playlist_link",
                          placeholder="https://www.youtube.com/playlist?list=...")
            st.caption("Para fixar para todos, coloque `playlist = \"link\"` nos Secrets do app.")
    st.divider()

PADRAO = Parametros()

# ----------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Calcário utilizado")
    st.caption(f"Modelo calibrado para CaO {PADRAO.calcario.ref_cao:g}% · MgO "
               f"{PADRAO.calcario.ref_mgo:g}% · PRNT {PADRAO.calcario.ref_prnt:g}")
    cao = st.number_input("CaO (%)", 0.0, 60.0, PADRAO.calcario.cao, 0.1)
    mgo = st.number_input("MgO (%)", 0.0, 30.0, PADRAO.calcario.mgo, 0.1)
    prnt = st.number_input("PRNT (%)", 30.0, 150.0, PADRAO.calcario.prnt, 1.0)

    with st.expander("Regras (avançado)"):
        st.markdown("**Calcário**")
        c_min = st.number_input("Dose mínima (kg/ha)", 0.0, 5000.0, PADRAO.calcario.dose_min, 50.0)
        c_max = st.number_input("Dose máxima (kg/ha)", 0.0, 10000.0, PADRAO.calcario.dose_max, 100.0)
        fab = st.number_input("Fator de abertura", 1.0, 3.0, PADRAO.calcario.fator_abertura, 0.1)
        ordem = st.radio("Na abertura, o fator é aplicado…",
                         ["depois dos limites (720–6840 kg/ha)", "antes dos limites (máx. = dose máxima)"])
        k_med = st.checkbox("Usar K do laudo na CTC* (em vez da constante 5,6)", PADRAO.calcario.usar_k_medido)
        st.markdown("**Gesso**")
        g_min = st.number_input("Gesso mínimo (kg/ha)", 0.0, 2000.0, PADRAO.gesso.dose_min, 50.0)
        g_max = st.number_input("Gesso máximo (kg/ha)", 0.0, 5000.0, PADRAO.gesso.dose_max, 100.0)
        s1 = st.number_input("S alvo, argila < 200 g/kg", 0.0, 100.0, PADRAO.gesso.s_alvo_arenoso, 1.0)
        s2 = st.number_input("S alvo, argila 200–400 g/kg", 0.0, 100.0, PADRAO.gesso.s_alvo_medio, 1.0)
        s3 = st.number_input("S alvo, argila > 400 g/kg", 0.0, 100.0, PADRAO.gesso.s_alvo_argiloso, 1.0)
        st.markdown("**P2O5 e KCl**")
        p_min = st.number_input("P2O5 mínimo (kg/ha)", 0.0, 300.0, PADRAO.fosforo.dose_min, 5.0)
        p_max = st.number_input("P2O5 máximo (kg/ha)", 0.0, 300.0, PADRAO.fosforo.dose_max, 5.0)
        k_min = st.number_input("KCl mínimo (kg/ha)", 0.0, 500.0, PADRAO.potassio.dose_min, 10.0)

par = Parametros(
    calcario=replace(PADRAO.calcario, cao=cao, mgo=mgo, prnt=prnt, dose_min=c_min, dose_max=c_max,
                     fator_abertura=fab, abertura_antes_dos_limites=ordem.startswith("antes"),
                     usar_k_medido=k_med),
    gesso=replace(PADRAO.gesso, dose_min=g_min, dose_max=g_max,
                  s_alvo_arenoso=s1, s_alvo_medio=s2, s_alvo_argiloso=s3),
    fosforo=replace(PADRAO.fosforo, dose_min=p_min, dose_max=p_max),
    potassio=replace(PADRAO.potassio, dose_min=k_min),
)
f = fatores_calcario(par.calcario)
if any(abs(f[k] - 1) > 1e-9 for k in ("ca", "mg", "prnt")):
    st.sidebar.info(f"Correção da dose de calcário: critério Ca ×{f['ca'] * f['prnt']:.2f} · "
                    f"critério Mg ×{f['mg'] * f['prnt']:.2f}")

# ------------------------------------------------------------------ topo
st.image(str(ASSETS / "banner.jpg"), width="stretch")
st.markdown('<div class="atria-titulo">Recomendação de calcário, gesso, P2O5 e KCl</div>'
            '<div class="atria-sub">Anexe o laudo de análise de solo (Excel). O sistema lê as colunas, '
            'aplica as regras e gera a planilha de recomendações e a interpretação por talhão.</div>'
            '<div class="atria-faixa"></div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False, max_entries=40)
def ler_laudo_cache(conteudo: bytes, nome: str, versao: str = VERSAO):
    return ler_laudo(conteudo, nome)



def bloco_ajuste(col, titulo: str, media_regra: float, chave: str, rotulo_produto: str,
                 antes=None) -> AjusteProduto:
    """Controles de ajuste de um produto; devolve o AjusteProduto escolhido."""
    with col:
        st.markdown(f"**{titulo}**")
        if antes:
            antes()
        st.caption(f"Média pela regra: **{br(media_regra)} kg/ha**")
        comprou = st.checkbox("Cliente já comprou", key=f"{chave}_comprou",
                              help=f"Se o cliente já comprou {rotulo_produto}, informe a média (kg/ha) "
                                   "que o volume comprado permite. A coluna inteira é ajustada, "
                                   "mantendo as proporções entre as amostras.")
        if comprou:
            alvo = st.number_input("Média a atingir (kg/ha)", min_value=0.0,
                                   value=float(round(media_regra or 0)), step=10.0, format="%.0f",
                                   key=f"{chave}_alvo")
            return AjusteProduto(media_alvo=alvo)
        pct = st.number_input("Ajuste (%)", -90.0, 300.0, 0.0, 5.0, format="%.0f", key=f"{chave}_pct",
                              help="Ex.: 10 = aumenta todas as doses em 10%; −15 = reduz 15%.")
        return AjusteProduto(pct=pct)


def sem_attrs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.attrs = {}
    return df


@st.cache_data(show_spinner=False, max_entries=80)
def png_interpretacao(tipo: str, df: pd.DataFrame, *extra) -> bytes:
    """Figuras da aba de interpretação, guardadas em cache (não são refeitas a cada clique)."""
    from types import SimpleNamespace
    df = df.copy()
    df.attrs = {}
    with it.TRAVA_MPL:
        if tipo == "mapa":
            fig = it.figura_mapa(df)
        elif tipo == "panorama":
            fig = it.figura_panorama(df)
        elif tipo == "classes":
            fig = it.figura_classes_amostras(SimpleNamespace(dados=df))
        elif tipo == "bases":
            fig = it.figura_bases(df, *extra)
        else:
            fig = it.figura_doses(df, list(extra[0]))
        return it.figura_png(fig)


def adiado(funcao, *args):
    """Função sem argumentos para o download_button: o arquivo só é montado quando o usuário clica."""
    def gerar():
        with it.TRAVA_MPL:
            return funcao(*args)
    return gerar


def estilo_linhas(df: pd.DataFrame, sub: pd.Series):
    def cor(row):
        return [f"background-color: {VERMELHO_SUB}; color: #1D2A22" if sub.loc[row.name] else "" for _ in row]
    doses = colunas_doses(df)
    df = df.copy()
    df.attrs = {}
    df["N Lab"] = df["N Lab"].astype(str)
    for c in doses:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.style.apply(cor, axis=1)
            .format({c: (lambda v: "" if pd.isna(v) else br(v)) for c in doses}))


def avisos_distancia(proj, laudo) -> list[str]:
    """Laudos juntados: avisa quando os talhões de um arquivo estão longe dos demais."""
    if not getattr(laudo, "origens", None) or "_arquivo" not in laudo.dados:
        return []
    from motor.geo import chave_talhao
    from shapely.ops import unary_union
    arq_de = {chave_talhao(t): a for t, a in zip(laudo.dados["talhao"], laudo.dados["_arquivo"])}
    por_arq: dict[str, list] = {}
    for t in proj.talhoes:
        por_arq.setdefault(arq_de.get(t.chave, "?"), []).append(t.perimetro)
    if len(por_arq) < 2:
        return []
    geoms = {a: unary_union(g) for a, g in por_arq.items()}
    avisos = []
    for a, g in geoms.items():
        resto = unary_union([h for b, h in geoms.items() if b != a])
        d = g.distance(resto) / 1000
        if d > 5:
            avisos.append(f"Os talhões de **{a}** estão a {d:.1f} km dos demais. Confira se são mesmo da mesma "
                          "fazenda (se não forem, separe os laudos no quadro 🔗 lá em cima).")
    return avisos


def editor_blocos(proj, k0: str) -> dict[str, str]:
    """Blocos de aplicação: talhões com o mesmo nome de bloco saem numa prescrição/shapefile só."""
    with st.expander("🧩 Unir talhões na prescrição (opcional)"):
        st.caption("Para o produtor que prefere poucos arquivos: talhões com o **mesmo nome de bloco** "
                   "(ex.: *Bloco A*, *Pivô 1*) recebem as mesmas doses de zona e saem num **shapefile único por "
                   "produto**. Deixe em branco para manter o talhão separado. Os mapas de fertilidade e os "
                   "volumes por talhão não mudam.")
        todos = st.checkbox("Todos os talhões num bloco só", key=f"todos_bloco_{k0}")
        if todos:
            return {t.nome: "Fazenda" for t in proj.talhoes}
        tab = st.data_editor(
            pd.DataFrame({"Talhão": [t.nome for t in proj.talhoes],
                          "Área (ha)": [round(t.area_ha, 1) for t in proj.talhoes],
                          "Bloco": [""] * len(proj.talhoes)}),
            hide_index=True, disabled=["Talhão", "Área (ha)"], key=f"blocos_{k0}",
            column_config={"Bloco": st.column_config.TextColumn("Bloco de aplicação",
                                                                help="Mesmo nome = prescritos juntos")})
        blocos = {t: str(b).strip() for t, b in zip(tab["Talhão"], tab["Bloco"]) if b and str(b).strip()}
        contagem = pd.Series(list(blocos.values())).value_counts() if blocos else pd.Series(dtype=int)
        juntos = contagem[contagem > 1]
        if len(juntos):
            st.success(" · ".join(f"**{b}**: {n} talhões" for b, n in juntos.items()))
        return blocos


def bloco_shapefiles(r: dict, k0: str) -> None:
    """Escolha dos produtos para exportar em shapefile (P₂O₅ puro só com liberação explícita)."""
    from motor.prescricao import zip_prescricoes
    itens = r.get("blocos") or r["mapas"]
    produtos = list(dict.fromkeys(p for m in itens for p in m.zonas))
    tem_produto_p = any(p.startswith("Produto") for p in produtos)
    parcelado = any("aplicação" in p for p in produtos)
    st.markdown("##### Shapefiles de prescrição")
    liberar_p = False
    if not tem_produto_p:
        st.warning("Sem formulação fosfatada, a prescrição de fósforo está em **P₂O₅ (nutriente puro, "
                   "equivalente a um produto 00-100-00 que não existe)** — não serve para o controlador. "
                   "Informe a formulação comprada no formulário acima e gere de novo.")
        liberar_p = st.checkbox("Liberar mesmo assim o shapefile de P₂O₅ (só para conversão manual)",
                                key=f"libp_{k0}")
    puros = {"P2O5", "P2O5 correção"}                    # nutriente puro: só com liberação explícita
    if "P2O5 correção" in produtos and tem_produto_p:
        st.caption("A **correção de P** está em P₂O₅ (sem formulação): informe a formulação da correção em "
                   "“Ajustar doses” para o shapefile sair em dose de produto.")
        liberar_p = st.checkbox("Liberar o shapefile da correção em P₂O₅ (só para conversão manual)",
                                key=f"libpc_{k0}")
    opcoes = [p for p in produtos if p not in puros or liberar_p]
    padrao = [p for p in opcoes if p not in puros and not (parcelado and p == "KCl")]
    escolha = st.multiselect("Produtos para baixar", opcoes, default=padrao, key=f"prods_{k0}",
                             help="Uma pasta por talhão, um shapefile por produto (WGS84, campo Taxa_Dest_).")
    if escolha:
        mapas, epsg = itens, r["epsg"]
        st.download_button(f"🗂️ Baixar shapefiles ({len(escolha)} produto{'s' if len(escolha) > 1 else ''}, .zip)",
                           lambda: zip_prescricoes(mapas, epsg, produtos=escolha),
                           file_name=f"Prescricoes - {r['nome']}.zip", mime="application/zip",
                           help="Uma pasta por talhão ou bloco de aplicação, um shapefile por produto.",
                           width="stretch", key=f"dlz_{k0}")
        ajuda_pastas_monitor()


def ajuda_pastas_monitor() -> None:
    """Ícone com a estrutura de pastas que cada marca de monitor espera no pen drive (só informativo)."""
    from motor.prescricao import PASTAS_MONITOR
    linhas = ["**Onde colocar os shapefiles no pen drive** (crie as pastas e copie os 4 arquivos de cada "
              "mapa: .shp, .shx, .dbf e .prj):", ""]
    for marca, pasta, ex in PASTAS_MONITOR:
        linhas.append(f"- **{marca}:** `{pasta}` → ex.: `{ex}`")
    linhas += ["", "Os nomes dos arquivos já saem curtos (até 9 caracteres, sem espaços), ex.: `CAL_T4` = calcário "
               "do talhão 4. Siglas: CAL calcário · GES gesso · SEL enxofre elementar · FOS formulação fosfatada · "
               "KCL cloreto de potássio · KC1/KC2 1ª/2ª aplicação · SEM sementes. O zip traz um LEIA-ME com a "
               "lista completa."]
    st.markdown("📁 **Pastas por marca de monitor**", help="\n".join(linhas))


def ler_kmls(arquivos) -> list:
    from motor.geo import ler_kml
    kmls = []
    for f in arquivos:
        try:
            kmls.append(ler_kml(f.getvalue(), f.name))
        except Exception as e:  # noqa: BLE001
            st.error(f"{f.name}: {e}")
    return kmls


def conferencia_ui(proj, chave: str) -> bool:
    """Tabela amostra × ponto e trava: divergências precisam ser conferidas antes de gerar mapas e prescrições."""
    conf = getattr(proj, "conferencia", pd.DataFrame())
    pend = getattr(proj, "pendencias", [])
    metodos = sorted(set(conf["Ligação"])) if len(conf) else []
    rot = "🔗 Conferência amostra × ponto" + (f" · ligação por {', '.join(metodos)}" if metodos else "")
    with st.expander(rot + (" · ⚠️ divergências" if pend else ""), expanded=bool(pend)):
        if "ordem do laudo" in metodos:
            st.caption("Parte das amostras não tem número na identificação: a ligação foi feita pela ordem das linhas "
                       "do laudo — confira se a ordem corresponde aos pontos.")
        if len(conf):
            st.dataframe(conf, hide_index=True, width="stretch", height=min(38 + 35 * len(conf), 320))
    if not pend:
        return True
    st.error("**A ligação entre amostras e pontos tem divergências** — os mapas podem ficar com os valores no lugar "
             "errado:\n\n" + "\n".join(f"- {p}" for p in pend))
    return st.checkbox("Conferi a tabela acima e quero gerar assim mesmo", key=f"confpend{chave}")


def ligar_kmls(laudo, k0: str, rotulo: str = "Arquivos KML/KMZ", perimetros=None, sufixo: str = "",
               mostrar_tabela: bool = True, atrib_extra: dict | None = None):
    """Upload dos KMLs, tabela de ligação arquivo → talhão e montagem do projeto (None se incompleto).

    `perimetros`: KMLs de perímetro já enviados (book comparativo: o ano anterior usa os mesmos perímetros e
    só os seus pontos)."""
    from dataclasses import replace as _rep

    from motor.geo import montar_projeto, numero_talhao
    kmls_up = st.file_uploader(rotulo, type=["kml", "kmz"], accept_multiple_files=True, key=f"kml{sufixo}_{k0}")
    if not kmls_up:
        if perimetros is None:
            st.info("Nomeie os arquivos com o talhão (ex.: Perimetro_TH_1.kml, Pontos_TH_1.kml) — "
                    "ou indique o talhão na tabela que aparece após o envio.")
        return None
    kmls = ler_kmls(kmls_up)
    if perimetros is not None:                    # só os pontos deste envio; perímetros vêm do ano recente
        kmls = [_rep(k, poligonos=[]) for k in kmls if len(k.pontos)]
    talhoes_laudo = list(dict.fromkeys(laudo.dados["talhao"]))
    tab = pd.DataFrame({"Arquivo": [k.nome for k in kmls],
                        "Tipo": [{"perimetro": "Perímetro", "pontos": "Pontos"}.get(k.tipo, k.tipo) for k in kmls],
                        "Pontos": [len(k.pontos) for k in kmls],
                        "Talhão": [numero_talhao(k.nome) or "" for k in kmls]})
    tab = st.data_editor(tab, hide_index=True, disabled=["Arquivo", "Tipo", "Pontos"], key=f"kmltab{sufixo}_{k0}",
                         column_config={"Talhão": st.column_config.SelectboxColumn(
                             "Talhão", options=[str(t) for t in talhoes_laudo], required=True)})
    atrib = dict(zip(tab["Arquivo"], tab["Talhão"].astype(str)))
    try:
        todos = kmls + list(perimetros or [])
        proj = montar_projeto(laudo.dados, todos, {**(atrib_extra or {}), **atrib})
    except Exception as e:  # noqa: BLE001
        st.error(f"Não foi possível ligar os KMLs ao laudo: {e}")
        return None
    for a in proj.avisos:
        if not any(a.endswith(p_.split(": ", 1)[-1]) for p_ in proj.pendencias):
            st.warning(a)
    if not proj.talhoes:
        return None
    proj.liberado = conferencia_ui(proj, f"{sufixo}_{k0}")
    if mostrar_tabela:
        st.dataframe(pd.DataFrame([{"Talhão": t.nome, "Área (ha)": round(t.area_ha, 2),
                                    "Pontos ligados": len(t.amostras), "Amostras 20-40": len(t.sub),
                                    "ha/ponto": round(t.area_ha / max(len(t.amostras), 1), 1)}
                                   for t in proj.talhoes]), hide_index=True)
    for a in avisos_distancia(proj, laudo):
        st.warning(a)
    st.session_state[f"kmls{sufixo}_{k0}"] = kmls
    st.session_state[f"atrib{sufixo}_{k0}"] = atrib
    return proj


def secao_ano_anterior(laudo, k0: str, nome_laudo: str):
    """Laudo e pontos do ano anterior para o book comparativo. Devolve {'projeto', 'rotulos'} ou None."""
    from motor.comparacao import ano_do_nome
    from motor.geo import montar_projeto
    st.markdown("#### 📈 Ano anterior")
    arq_ant = st.file_uploader("Laudo(s) do ano anterior (mesma área)", type=["xlsx", "xls", "csv"],
                               accept_multiple_files=True, key=f"lant_{k0}")
    if not arq_ant:
        st.info("Envie o laudo do ano anterior para montar o book comparativo.")
        return None
    laudo_ant, nome_ant = laudo_unico(arq_ant)
    if laudo_ant is None:
        return None
    c1, c2 = st.columns(2)
    rot_a = c1.text_input("Rótulo do ano anterior", ano_do_nome(nome_ant) or "Anterior", key=f"rota_{k0}")
    rot_b = c2.text_input("Rótulo do ano recente", ano_do_nome(nome_laudo) or "Recente", key=f"rotb_{k0}")
    if rot_a.strip() == rot_b.strip():
        st.warning("Use rótulos diferentes para os dois anos (ex.: 2024 e 2026).")
        return None
    st.caption("Pontos do ano anterior: se a amostragem mudou, envie os KML **de pontos** daquele ano (os perímetros "
               "são os do ano recente). Sem envio, usamos os mesmos pontos do ano recente, na mesma ordem.")
    perims = [k for k in st.session_state.get(f"kmls_{k0}", []) if k.poligonos]
    proj_ant = None
    if st.session_state.get(f"kmlant_{k0}"):
        proj_ant = ligar_kmls(laudo_ant, k0, "KML de pontos do ano anterior", perimetros=perims, sufixo="ant",
                              mostrar_tabela=False, atrib_extra=st.session_state.get(f"atrib_{k0}"))
        if proj_ant is None:
            return None
    else:
        ligar_kmls(laudo_ant, k0, "KML de pontos do ano anterior (opcional)", perimetros=perims, sufixo="ant",
                   mostrar_tabela=False)
        try:
            proj_ant = montar_projeto(laudo_ant.dados, st.session_state.get(f"kmls_{k0}", []),
                                      st.session_state.get(f"atrib_{k0}", {}))
        except Exception as e:  # noqa: BLE001
            st.error(f"Não foi possível ligar o laudo anterior aos pontos do ano recente: {e}")
            return None
    if not hasattr(proj_ant, "liberado"):
        proj_ant.liberado = conferencia_ui(proj_ant, f"ant2_{k0}")
    comuns = {t.chave for t in proj_ant.talhoes if len(t.amostras)}
    st.caption(f"Laudo anterior: {len(comuns)} talhão(ões) com pontos ligados · "
               + " · ".join(f"{t.nome}: {len(t.amostras)} pts" for t in proj_ant.talhoes[:12]))
    for a in proj_ant.avisos[:6]:
        st.caption(f"⚠️ {a}")
    return {"projeto": proj_ant, "rotulos": (rot_a.strip(), rot_b.strip()), "liberado": proj_ant.liberado,
            "assinatura": assinatura(laudo_ant.dados, [(t.nome, len(t.amostras)) for t in proj_ant.talhoes])}


def excel_comparacao(c) -> bytes:
    import io

    from motor.book import NOMES
    from motor.book_comparacao import NOME_TAB
    from motor.comparacao import tabela_evolucao
    from motor.prescricao import ATRIB_MAPA
    chaves = [k for k in ATRIB_MAPA if all(k in a.atributos and k in b.atributos for a, b in c.pares)]
    ev = tabela_evolucao(c, chaves)
    ev.insert(0, "Atributo", [NOME_TAB.get(k, NOMES.get(k, k)) for k in ev["chave"]])
    ev = ev.drop(columns=["chave"]).rename(columns={"A": c.rotulos[0], "B": c.rotulos[1], "Δ": "Variação",
                                                     "Δ%": "Variação (%)", "classe A": f"Classe {c.rotulos[0]}",
                                                     "classe B": f"Classe {c.rotulos[1]}",
                                                     "melhora": "Melhora (classes)"})
    por_tal = []
    for a, b in c.pares:
        linha = {"Talhão": b.talhao.nome, "Área (ha)": round(b.talhao.area_ha, 2)}
        for k in chaves:
            linha[f"{k} {c.rotulos[0]}"] = float(np.nanmean(a.atributos[k]))
            linha[f"{k} {c.rotulos[1]}"] = float(np.nanmean(b.atributos[k]))
        por_tal.append(linha)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        ev.round(2).to_excel(w, sheet_name="Evolução (área)", index=False)
        pd.DataFrame(por_tal).round(2).to_excel(w, sheet_name="Médias por talhão", index=False)
        if len(c.volumes):
            c.volumes.round(2).to_excel(w, sheet_name="Volumes", index=False)
        for ws in w.sheets.values():
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(max(11, len(str(col[0].value)) + 2), 32)
    return buf.getvalue()


def mostrar_comparacao(c, k0: str, nome: str) -> None:
    from motor.book import NOMES, ORDEM_FERT
    from motor.book_comparacao import NOME_TAB
    from motor.comparacao import tabela_evolucao
    for a in c.avisos:
        st.caption(f"⚠️ {a}")
    chaves = [k for k in ORDEM_FERT if all(k in a.atributos and k in b.atributos for a, b in c.pares)]
    ev = tabela_evolucao(c, chaves)
    if len(ev):
        ra, rb = c.rotulos
        vis = pd.DataFrame({"Atributo": [NOME_TAB.get(k, NOMES[k]) for k in ev["chave"]],
                            ra: ev["A"].round(2), rb: ev["B"].round(2), "Variação": ev["Δ"].round(2),
                            "Condição": ["▲ melhorou" if m > 0.05 else "▼ piorou" if m < -0.05 else "= estável"
                                         for m in ev.get("melhora", pd.Series(0, index=ev.index)).fillna(0)]})
        st.markdown(f"##### Evolução da fertilidade · {ra} × {rb}")
        st.dataframe(vis, hide_index=True, width="stretch")
    st.download_button("📊 Baixar comparação (Excel)", adiado(excel_comparacao, c),
                       file_name=f"Comparacao - {nome}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                       key=f"dlcmp_{k0}")


def alerta_subsuperficie(laudo) -> str:
    """Com S elementar no lugar do gesso: avisa se a camada 20-40 indica necessidade de gesso como condicionador
    (Ca ≤ 5 mmolc/dm³, Al ≥ 5 mmolc/dm³ ou m ≥ 20%)."""
    d = laudo.dados[laudo.dados["subsuperficial"]]
    if d.empty:
        return ""
    m = d["m"] if "m" in d and d["m"].notna().any() else 100 * d["al"] / (d["ca"] + d["mg"] + d["k"] + d["al"])
    crit = (d["ca"] <= 5) | (d["al"] >= 5) | (m >= 20)
    if not crit.any():
        return ""
    tal = sorted(set(d.loc[crit, "talhao"].astype(str)))
    return (f"**S elementar não substitui o gesso como condicionador de subsuperfície.** Na camada 20–40 cm, "
            f"{int(crit.sum())} amostra(s) (talhões {', '.join(tal[:8])}) têm Ca ≤ 5 mmolc/dm³, Al ≥ 5 mmolc/dm³ ou "
            "m ≥ 20% — situação em que o gesso (Ca + S) é indicado para melhorar o ambiente radicular em profundidade.")


def assinatura(*partes) -> str:
    """Impressão digital das entradas de um resultado (para avisar quando ele fica desatualizado)."""
    import hashlib
    h = hashlib.sha1()
    for p in partes:
        if isinstance(p, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(p.astype(str), index=False).values.tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:16]


def assinatura_entradas(laudo, proj, par, ajustes, abertura, blocos=None, extra=None) -> str:
    kmls = [(t.nome, len(t.amostras), round(t.area_ha, 3)) for t in proj.talhoes] if proj is not None else []
    return assinatura(laudo.dados, kmls, par, ajustes, sorted((str(k), bool(v)) for k, v in abertura.items()),
                      sorted((blocos or {}).items()), extra)


def mostrar_conferencia(conf) -> None:
    if conf is None or not len(conf):
        return
    ruins = conf[conf["Situação"] != "ok"]
    rot = (f"✅ Prescrições conferidas: {len(conf)} mapas — cobertura, sobreposição, doses e volume do shapefile relido"
           if ruins.empty else f"⚠️ Conferência das prescrições: {len(ruins)} de {len(conf)} mapas pedem atenção")
    with st.expander(rot, expanded=not ruins.empty):
        st.dataframe(conf, hide_index=True, width="stretch")


def aba_montar_book(laudo, par, abertura, ajustes, k0, nome_laudo, modo: str = "normal"):
    """Mapas (krigagem), prescrições em shapefile e book em PDF."""
    from motor.book import MARCAS, DadosBook, montar_book
    from motor.externos import FONTES_SATELITE
    from motor.geo import ler_kml, montar_projeto, numero_talhao
    from motor.prescricao import (BASE_DOSE, calcular_doses, conferir, filtrar_prescricoes, gerar_todas_zonas,
                                  superficies, tabela_volumes)

    st.markdown("Anexe os **KML/KMZ de perímetro e de pontos** de cada talhão. Os pontos de cada talhão "
                "são ligados às amostras do laudo pelo número (AMOSTRA 07 ↔ ponto 7); amostras de 20-40 cm "
                "herdam o ponto da amostra de mesmo número.")
    proj = ligar_kmls(laudo, k0)
    if proj is None:
        return

    mapa_blocos = editor_blocos(proj, k0)
    ajustes_base = ajustes

    comp = None
    if modo == "comparacao":
        comp = secao_ano_anterior(laudo, k0, nome_laudo)
        if comp is None:
            return

    d0 = laudo.dados
    with st.form(f"form_book_{k0}"):
        c1, c2, c3 = st.columns(3)
        marca = c1.radio("Book para", list(MARCAS), horizontal=True, index=list(MARCAS).index("Atria"))
        produtor = c2.text_input("Produtor", next((x for x in d0["proprietario"] if x), "").strip().title())
        propriedade = c3.text_input("Propriedade", next((x for x in d0["propriedade"] if x), "").strip().title())
        c4, c5, c6, c7 = st.columns(4)
        municipio = c4.text_input("Município (UF)", placeholder="ex.: Paraíso das Águas (MS)")
        data_coleta = c5.text_input("Data da coleta", placeholder="dd/mm/aaaa")
        safra = c6.text_input("Ano das prescrições", str(pd.Timestamp.today().year))
        subam = c7.number_input("Subamostras por ponto", 1, 50, 10)
        c8, c9 = st.columns(2)
        prod_cal = c8.text_input("Descrição do calcário",
                                 f"Calcário · {par.calcario.cao:g}% CaO; {par.calcario.mgo:g}% MgO; PRNT {par.calcario.prnt:g}")
        prod_ges = c9.text_input("Descrição do gesso", "Gesso agrícola · 19% Ca²⁺; 15% S")
        c10, c11, c12 = st.columns(3)
        chave_google = segredo("google_maps_key")
        opcoes = list(FONTES_SATELITE) + ["Sem imagem"]
        fonte = c10.selectbox("Imagem de satélite", opcoes,
                              index=0 if chave_google else opcoes.index("Esri World Imagery"),
                              help="Google exige a chave `google_maps_key` nos Secrets (Map Tiles API).")
        res_m = c11.select_slider("Tamanho do pixel (m)", [5, 10, 15, 20], value=10)
        n_zonas = c12.select_slider("Zonas por prescrição", [3, 4, 5, 6, 7, 8], value=6)
        st.markdown("**Fósforo e potássio nas prescrições**")
        c13, c14, c15 = st.columns(3)
        formula_book = c13.text_input(
            "Formulação fosfatada comprada (N-P₂O₅-K₂O)", ajustes.formula_p,
            placeholder="ex.: 11-52-00", help="As prescrições de fósforo saem em dose do PRODUTO. Sem formulação, "
                                              "o mapa mostra P₂O₅ (nutriente) e o shapefile de P₂O₅ fica bloqueado.")
        parc_book = c14.toggle("Parcelar KCl em 2 aplicações", ajustes.kcl_parcelado)
        pct_book = c15.slider("1ª aplicação do KCl (%)", 10, 90, int(ajustes.kcl_pct_1), 5)
        from motor.prescricao import FAMILIAS
        fam_op = [f for f in FAMILIAS if f != "Correção de P" or ajustes.correcao_p2o5() > 0]
        familias = st.multiselect(
            "Prescrições que entram no book e nos shapefiles", fam_op, default=fam_op,
            help="Tire o que o produtor não vai aplicar nesta safra (ex.: gessagem). Vale só para as prescrições: "
                 "páginas, tabela de volumes e shapefiles. Os mapas de fertilidade continuam completos.")
        st.markdown("**Interpolação**")
        c20, c21 = st.columns(2)
        modo_mapa = c20.radio(
            "Mapas de fertilidade", ["Fiel às amostras", "Estatística (v14)"], horizontal=True,
            help="'Fiel às amostras': o mapa segue o valor medido em cada ponto (cada amostra composta representa a "
                 "sua célula da grade), como nos books feitos no SMS. 'Estatística': semivariograma ajustado na área "
                 "inteira — quando a malha não mostra estrutura espacial, o mapa tende à média e esconde a variação "
                 "entre pontos.")
        doses_iguais = c21.checkbox(
            "Prescrições com a mesma interpolação dos mapas", value=True,
            help="Ligado (padrão): as doses são calculadas sobre as mesmas superfícies dos mapas de fertilidade — "
                 "mapa e prescrição mostram o mesmo padrão. Desligado: as doses usam a interpolação estatística "
                 "da v14 (zonas mais largas), mesmo com os mapas 'fiéis às amostras'.")
        c18, c19 = st.columns(2)
        sem_estr = c18.radio("Dose de atributo sem estrutura espacial comprovada",
                             ["Mapa exploratório", "Valor médio uniforme"],
                             horizontal=True, help="Quando a krigagem não prevê melhor que a média na validação cruzada "
                                                   "(malha amostral não detecta variação espacial). 'Valor médio' deixa a "
                                                   "dose daquele atributo uniforme no talhão. Não altera os mapas "
                                                   "'fiéis às amostras'.")
        extremos = c19.checkbox("Limitar valores extremos isolados (evita 'alvos' no mapa)", value=True,
                                help="Desligue se uma mancha real de deficiência estiver sendo atenuada.")
        c16, c17 = st.columns(2)
        discreto = c16.checkbox("Mapas de fertilidade em faixas sólidas (1 cor por classe — melhor para impressão)",
                                value=True)
        ndvi = c17.checkbox("Incluir página de NDVI (Sentinel-2, últimos 12 meses)", value=True,
                            help="Vigor da vegetação por imagens de satélite gratuitas (Copernicus). "
                                 "Acrescenta cerca de 20–40 s na geração.")
        liberado = getattr(proj, "liberado", True) and (comp is None or comp.get("liberado", True))
        gerar = st.form_submit_button("🗺️ Gerar mapas, shapefiles e book", type="primary", width="stretch",
                                      disabled=not liberado)

    chave_res = f"book_{k0}"
    if gerar and formula_book and not nome_produto_p(formula_book):
        st.error("Formulação fosfatada inválida. Use o formato 11-52-00.")
        gerar = False
    if gerar:
        ajustes = replace(ajustes, formula_p=formula_book.strip(), kcl_parcelado=parc_book,
                          kcl_pct_1=float(pct_book))
        barra = st.status("Gerando…", expanded=True)
        try:
            barra.write("Interpolando atributos (krigagem)…")
            opcoes = {"limitar_extremos": extremos,
                      "sem_estrutura": "uniforme" if sem_estr.startswith("Valor") else "exploratorio"}
            ROTULO_ATRIB = {"ca": "Ca", "mg": "Mg", "hal": "H+Al", "al": "Al", "k": "K", "p": "P", "s": "S",
                            "argila": "argila"}
            modo = "fiel" if modo_mapa.startswith("Fiel") else "estatistico"
            modos = {"modo": modo, "modo_doses": modo if doses_iguais else "estatistico"}
            mapas = superficies(proj, float(res_m), **modos, **opcoes)
            sem = {}
            for m in mapas:                 # o aviso vale para o que entra no cálculo das doses
                for k_, aj_ in m.ajustes_doses().items():
                    if k_ in BASE_DOSE and getattr(aj_, "sem_estrutura", False):
                        sem.setdefault(m.talhao.nome, []).append(k_)
            barra.write("Calculando doses por pixel e zonas de manejo…")
            avisos_doses: list[str] = [
                f"{t}: a malha amostral não comprova a variação espacial de "
                f"{', '.join(ROTULO_ATRIB.get(k_, k_) for k_ in ks)} dentro do talhão — "
                + ("dose calculada pelo valor médio" if opcoes["sem_estrutura"] == "uniforme"
                   else "a dose varia pouco em torno da média (confira antes de aplicar em taxa variável)")
                for t, ks in sem.items()]
            calcular_doses(mapas, par, abertura, ajustes, avisos_doses)
            fora = filtrar_prescricoes(mapas, familias)
            if fora:
                avisos_doses.append("Prescrições deixadas de fora por escolha do usuário: " + ", ".join(fora).lower())
            if not any(m.doses for m in mapas):
                raise ValueError("Nenhuma prescrição selecionada: escolha ao menos uma em “Prescrições que entram no "
                                 "book e nos shapefiles”.")
            blocos = gerar_todas_zonas(mapas, par, n_zonas, mapa_blocos, aj=ajustes)
            barra.write("Conferindo as prescrições (relendo os shapefiles)…")
            conferencia = conferir(blocos, proj.epsg)
            dados = DadosBook(marca=marca, produtor=produtor, propriedade=propriedade, municipio=municipio,
                              data_coleta=data_coleta, ano_safra=safra, subamostras=int(subam),
                              produto_calcario=prod_cal, produto_gesso=prod_ges,
                              fonte_satelite=None if fonte == "Sem imagem" else fonte,
                              laudo_numero=re.sub(r"\.(xlsx|xls|csv)$", "", nome_laudo, flags=re.I)[:60],
                              chave_google=chave_google, cores_discretas=discreto, incluir_ndvi=ndvi,
                              metodos=getattr(laudo, "metodos", {}) or {}, interpolacao={**opcoes, **modos})
            cmp_res = None
            if comp is not None:
                from motor import book_comparacao
                from motor.comparacao import superficies_na_grade, volumes_anterior
                barra.write(f"Interpolando o laudo de {comp['rotulos'][0]} na mesma grade…")
                cmp_res = superficies_na_grade(mapas, comp["projeto"], **modos, **opcoes)
                cmp_res.rotulos = comp["rotulos"]
                volumes_anterior(cmp_res, par, abertura, ajustes)
                dados.titulo_capa = ("COMPARATIVO DA FERTILIDADE DO SOLO",
                                     f"{comp['rotulos'][0]} × {comp['rotulos'][1]} · PRESCRIÇÕES {safra}".strip())
                with it.TRAVA_MPL:
                    pdf, avisos = book_comparacao.montar(proj, mapas, dados, par, ajustes, cmp_res, laudo.dados,
                                                         progresso=barra.write, blocos=blocos)
            else:
                with it.TRAVA_MPL:
                    pdf, avisos = montar_book(proj, mapas, dados, par, ajustes, laudo.dados, progresso=barra.write,
                                              blocos=blocos)
            st.session_state[chave_res] = {"comparacao": cmp_res, "conferencia": conferencia,
                "assinatura": assinatura_entradas(laudo, proj, par, ajustes_base, abertura, mapa_blocos,
                                                  (comp["rotulos"], comp["assinatura"]) if comp else None),
                "pdf": pdf, "mapas": mapas, "blocos": blocos, "epsg": proj.epsg, "avisos": avisos_doses + avisos,
                "ajustes": ajustes,
                "volumes": tabela_volumes(mapas), "nome": f"{marca} - {propriedade or 'Book'}",
                "qualidade": pd.DataFrame([{"Talhão": m.talhao.nome, "Atributo": k, "Uso": uso,
                                            "Ajuste": aj.descricao(),
                                            "R² (validação cruzada)": round(aj.r2_cv, 2),
                                            "Estrutura espacial": "não comprovada" if getattr(aj, "sem_estrutura",
                                                                                              False) else "ok",
                                            "Extremos limitados": aj.outliers}
                                           for m in mapas
                                           for uso, ajs in (("mapa", m.ajustes), ("dose", m.ajustes_base))
                                           for k, aj in ajs.items()])}
            barra.update(label="Pronto!", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            barra.update(label="Falhou", state="error")
            st.exception(e)

    r = st.session_state.get(chave_res)
    if r:
        if r.get("assinatura") != assinatura_entradas(laudo, proj, par, ajustes_base, abertura, mapa_blocos,
                                                      (comp["rotulos"], comp["assinatura"]) if comp else None):
            st.warning("⚠️ **Resultado desatualizado:** o laudo, os KMLs, os parâmetros ou os ajustes mudaram depois "
                       "que este book foi gerado. Gere novamente antes de baixar.")
        for a in r["avisos"]:
            st.caption(f"⚠️ {a}")
        mostrar_conferencia(r.get("conferencia"))
        cr = r.get("comparacao")
        st.download_button("📕 Baixar book comparativo (PDF)" if cr is not None else "📕 Baixar book (PDF)", r["pdf"],
                           file_name=f"Book {'comparativo ' if cr is not None else ''}- {r['nome']}.pdf",
                           mime="application/pdf", type="primary", width="stretch", key=f"dlb_{k0}")
        if cr is not None:
            mostrar_comparacao(cr, k0, r["nome"])
        bloco_shapefiles(r, k0)
        v = r["volumes"].copy()
        if v["Produto"].str.startswith("Produto").any():
            v = v[v["Produto"] != "P2O5"]                  # com formulação, o volume é o do produto
        v["Área (ha)"] = v["Área (ha)"].round(2)
        v["Dose média (kg/ha)"] = v["Dose média (kg/ha)"].round(0)
        v["Total (t)"] = v["Total (t)"].round(2)
        st.markdown("##### Volumes de produto (pelos mapas de prescrição)")
        st.dataframe(v, hide_index=True, width="stretch")
        with st.expander("Detalhes da krigagem (uso interno — não vai para o book)"):
            st.dataframe(r["qualidade"], hide_index=True, width="stretch")


def fluxo_recomendacao(modo: str = "normal"):
    """Recomendação, interpretação e book a partir dos laudos.

    modo "comparacao": mesmo fluxo, com o laudo MAIS RECENTE; o book fica comparativo (pede o laudo anterior)."""
    cmp = modo == "comparacao"
    pre = "cmp_" if cmp else ""
    arquivos = st.file_uploader("Laudo(s) do ano mais recente" if cmp else "Laudo(s) de análise de solo",
                                type=["xlsx", "xls", "csv"], key=f"{pre}laudos",
                                accept_multiple_files=True,
                                help="Pode anexar vários arquivos. Laudos da mesma fazenda são reconhecidos e podem ser "
                                     "juntados num só (recomendação e book únicos).")
    if not arquivos:
        return


    # --------------------------------------------- leitura e agrupamento por fazenda
    lidos, falhas = [], []
    for arq in arquivos:
        try:
            lidos.append((arq.name, ler_laudo_cache(arq.getvalue(), arq.name)))
        except Exception as e:  # noqa: BLE001
            falhas.append((arq.name, e))
    for nome, e in falhas:
        st.error(f"Não foi possível ler {nome}: {e}")
    if not lidos:
        return

    grupos = list(range(1, len(lidos) + 1))
    if len(lidos) > 1:
        sug = sugerir_grupos([l for _, l in lidos])
        ident = [identificacao(l) for _, l in lidos]
        n_juntos = len(sug) - len(set(sug))
        with st.expander(("🔗 Laudos da mesma fazenda reconhecidos — serão juntados" if n_juntos else
                          "🔗 Juntar laudos da mesma fazenda"), expanded=bool(n_juntos)):
            st.caption("Arquivos com o **mesmo número de grupo** viram um laudo só: uma planilha de recomendação, "
                       "uma interpretação e um book com todos os talhões. O grupo é sugerido pelo nome do produtor "
                       "e da propriedade; altere o número para juntar ou separar.")
            tab_g = st.data_editor(
                pd.DataFrame({"Arquivo": [n for n, _ in lidos], "Produtor": [a for a, _ in ident],
                              "Propriedade": [b for _, b in ident],
                              "Talhões": [len(set(l.dados["talhao"])) for _, l in lidos], "Grupo": sug}),
                hide_index=True, disabled=["Arquivo", "Produtor", "Propriedade", "Talhões"], key=f"{pre}grupos_laudos",
                column_config={"Grupo": st.column_config.NumberColumn("Grupo", min_value=1, max_value=len(lidos),
                                                                      step=1, required=True)})
            grupos = [int(g) for g in tab_g["Grupo"]]

    unidades = []                       # (nome exibido, nome base dos arquivos, laudo)
    for g in dict.fromkeys(grupos):
        membros = [lidos[i] for i in range(len(lidos)) if grupos[i] == g]
        if len(membros) == 1:
            nome, lau = membros[0]
            unidades.append((nome, re.sub(r"\.(xlsx|xls|csv)$", "", nome, flags=re.I), lau))
        else:
            lau, _ = juntar_laudos([l for _, l in membros], [n for n, _ in membros])
            prod, faz = identificacao(lau)
            base = (faz or prod or "Laudos juntados").strip().title()
            unidades.append((f"{len(membros)} laudos juntados · {base}", base, lau))

    for n_arq, (nome_arq, base_nome, laudo) in enumerate(unidades):
        st.divider()
        st.subheader(f"📄 {nome_arq}".replace("_", "\\_"))
        if getattr(laudo, "origens", None):
            st.caption("Arquivos: " + " · ".join(n for n, _ in laudo.origens))
        k0 = f"{pre}{n_arq}_{nome_arq}"

        n_sub = int(laudo.dados["subsuperficial"].sum())
        st.caption(f"{len(laudo.dados) - n_sub} amostras 0-20 cm"
                   + (f" · {n_sub} subsuperficiais (20-40 cm)" if n_sub else "") + f" · aba “{laudo.aba}”")
        for a in laudo.avisos:
            if "subsuperficial" not in a:
                st.warning(a)
        avisos_dados = getattr(laudo, "avisos_dados", None) or []
        if avisos_dados:
            st.info("**Dados ausentes no laudo**\n\n" + "\n".join(f"- {a}" for a in avisos_dados), icon="🔎")

        # --- abertura
        d0 = laudo.dados[~laudo.dados["subsuperficial"]]
        talhoes = list(dict.fromkeys(d0["talhao"]))
        st.markdown("**Marque os talhões que são área de abertura** (dose de calcário × "
                    f"{par.calcario.fator_abertura:g})")
        edit = st.data_editor(
            pd.DataFrame({"Talhão": talhoes, "Abertura": [False] * len(talhoes),
                          "Amostras 0-20": [int((d0["talhao"] == t).sum()) for t in talhoes]}),
            hide_index=True, disabled=["Talhão", "Amostras 0-20"], key=f"ab_{k0}",
            column_config={"Abertura": st.column_config.CheckboxColumn("Área de abertura?")},
        )
        abertura = dict(zip(edit["Talhão"], edit["Abertura"]))

        # --- ajustes (médias pela regra vêm de um cálculo sem ajustes)
        metodos = getattr(laudo, "metodos", {}) or {}
        mehlich = metodos.get("p") == "Mehlich-1"
        if mehlich:
            st.error("**P extraído por Mehlich-1.** A regra de P₂O₅ e as classes de P da equipe foram calibradas para "
                     "P resina: a dose de P **não é calculada** até você confirmar nos ajustes abaixo.")
        base = recomendar(laudo, par, abertura)
        b0 = base[~base[COL_SUB]]
        with st.expander("⚙️ Ajustar doses · volume já comprado · formulação de P · correção de P · S elementar · "
                         "parcelamento de KCl",
                         expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            aj_cal = bloco_ajuste(c1, "Calcário", b0[COL_CAL].mean(), f"{k0}_cal", "calcário")
            s0 = st.session_state.get(f"{k0}_s0", False)
            if s0:
                tmp = recomendar(laudo, par, abertura, Ajustes(gesso_por_s_elementar=True))
                media_g, nome_g = tmp.loc[~tmp[COL_SUB], COL_S0].mean(), "S elementar"
            else:
                media_g, nome_g = b0[COL_GES].mean(), "gesso"
            aj_ges = bloco_ajuste(
                c2, "Gesso / S elementar", media_g, f"{k0}_{'s0aj' if s0 else 'ges'}", nome_g,
                antes=lambda: st.toggle("Substituir por S elementar", key=f"{k0}_s0",
                                        help="S elementar (kg/ha) = 4,1904 × (dose de gesso)^0,3754. Substitui só o "
                                             "FORNECIMENTO DE S: não traz Ca nem melhora o ambiente em profundidade "
                                             "(Ca baixo / Al alto na camada 20–40), funções do gesso."))
            aj_p = bloco_ajuste(c3, "P2O5", b0[COL_P].mean(), f"{k0}_p", "fósforo (em P2O5)")
            with c3:
                aceitar_mehlich = False
                if mehlich:
                    aceitar_mehlich = st.checkbox("Aplicar a regra de P (calibrada para resina) mesmo com P Mehlich-1",
                                                  key=f"{k0}_mehlich")
                formula = st.text_input("Formulação do produto (N-P2O5-K2O)", key=f"{k0}_form",
                                        placeholder="ex.: 11-52-00, 00-46-00, 02-22-00")
                if formula and not nome_produto_p(formula):
                    st.error("Formulação inválida. Use o formato 11-52-00.")
                elif formula:
                    fl = ler_formula(formula)
                    st.caption(f"Produto com {fl[1]:g}% de P2O5 → dose = P2O5 ÷ {fl[1] / 100:g}")
            media_k, nota_k = b0[COL_K].mean(), None
            fl_k = ler_formula(formula) if formula and nome_produto_p(formula) else None
            if fl_k and fl_k[2] > 0:                  # formulação com K2O: KCl já descontado
                tmp_k = recomendar(laudo, par, abertura, Ajustes(p2o5=aj_p, formula_p=formula))
                media_k = tmp_k.loc[~tmp_k[COL_SUB], COL_K].mean()
                nota_k = (f"A formulação {formula} traz {fl_k[2]:g}% de K₂O: o equivalente em KCl foi descontado "
                          f"(média {br(b0[COL_K].mean())} → {br(media_k)} kg/ha).")
            aj_k = bloco_ajuste(c4, "KCl", media_k, f"{k0}_k", "KCl")
            if nota_k:
                c4.caption("🔻 " + nota_k)
            with c4:
                parcelar = st.toggle("Parcelar em 2 aplicações", key=f"{k0}_parc",
                                     help="Indicado para solos arenosos, para reduzir perdas de K. "
                                          "As duas parcelas somam a dose total recomendada.")
                pct1 = 50.0
                if parcelar:
                    pct1 = float(st.slider("1ª aplicação (% da dose)", 10, 90, 50, 5, key=f"{k0}_pct1"))
                    st.caption(f"1ª aplicação: **{pct1:.0f}%** · 2ª aplicação: **{100 - pct1:.0f}%**")

            st.divider()
            pc_dose, pc_form, pc_prod, pc_var = 0.0, "", False, True
            if st.checkbox("**Correção de P** — dose extra de fósforo, além da manutenção", key=f"{k0}_pc",
                           help="Marque só quando o produtor for fazer correção de fósforo. A dose padrão (manutenção) "
                                "não muda: a correção sai em uma coluna própria, “P correção”, e em uma prescrição "
                                "separada no book."):
                q1, q2, q3, q4 = st.columns(4)
                pc_dose = q1.number_input("Dose média de correção (kg/ha)", 0.0, 2000.0, 0.0, 10.0, format="%.0f",
                                          key=f"{k0}_pc_dose")
                pc_prod = q2.radio("A dose informada é de", ["P₂O₅", "produto"], horizontal=True, key=f"{k0}_pc_un",
                                   help="P₂O₅: nutriente. Produto: quilos do fertilizante (precisa da formulação)."
                                   ) == "produto"
                pc_form = q3.text_input("Formulação da correção (N-P2O5-K2O)", key=f"{k0}_pc_form",
                                        placeholder="ex.: 00-18-00, 00-46-00, 11-52-00").strip()
                pc_var = q4.radio("Distribuição", ["Variável", "Uniforme"], horizontal=True, key=f"{k0}_pc_dist",
                                  help="Variável: mais onde o P do solo é menor, seguindo a mesma curva da "
                                       "manutenção; a média fica igual à dose informada. Uniforme: a mesma dose "
                                       "em toda a área.") == "Variável"
                if pc_form and not nome_produto_p(pc_form):
                    st.error("Formulação da correção inválida. Use o formato 00-18-00.")
                    pc_form = ""
                if pc_prod and not pc_form:
                    st.warning("Para informar a dose em produto, preencha a formulação da correção.")
                elif pc_dose > 0:
                    aj_tmp = Ajustes(p_correcao=pc_dose, p_correcao_formula=pc_form, p_correcao_em_produto=pc_prod)
                    p2 = aj_tmp.correcao_p2o5()
                    fl_c = ler_formula(pc_form) if pc_form else None
                    st.caption(f"Correção: média de **{br(p2)} kg/ha de P₂O₅**"
                               + (f" = **{br(p2 / (fl_c[1] / 100))} kg/ha de {pc_form}**" if fl_c else "")
                               + " — somada à manutenção, não a substitui."
                               + (" O K₂O dessa formulação não é descontado do KCl." if fl_c and fl_c[2] > 0 else ""))

        ajustes = Ajustes(calcario=aj_cal, gesso=aj_ges, p2o5=aj_p, kcl=aj_k,
                          gesso_por_s_elementar=s0, formula_p=formula or "",
                          kcl_parcelado=parcelar, kcl_pct_1=pct1, bloquear_p=mehlich and not aceitar_mehlich,
                          p_correcao=pc_dose, p_correcao_formula=pc_form, p_correcao_em_produto=pc_prod,
                          p_correcao_variavel=pc_var)
        if s0:
            alerta = alerta_subsuperficie(laudo)
            if alerta:
                st.warning(alerta)
        res = recomendar(laudo, par, abertura, ajustes)
        rec = res[~res[COL_SUB]]
        doses = colunas_doses(res)
        doses_totais = colunas_doses_totais(res)

        d1, d2 = st.columns(2)
        # arquivos gerados só no clique (não a cada interação na tela)
        d1.download_button("⬇️ Baixar recomendações (Excel)",
                           data=adiado(gerar_excel, res, laudo, par, nome_arq, ajustes),
                           file_name=f"Recomendacao - {base_nome}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary", key=f"dl_{k0}", width="stretch")
        d2.download_button("📊 Baixar interpretação por talhão (Excel)",
                           data=adiado(gerar_excel_interpretacao, res, laudo, nome_arq),
                           file_name=f"Interpretacao - {base_nome}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key=f"dli_{k0}", width="stretch")

        aba_rec, aba_int, aba_book = st.tabs(["📋 Recomendações", "📊 Interpretação por talhão",
                                              "📈 Montar book comparativo" if cmp else "📚 Montar book"])

        with aba_rec:
            cols = st.columns(len(doses_totais))
            for col, nome in zip(cols, doses_totais):
                s = rec[nome].dropna()
                info = res.attrs["ajustes"].get(
                    {COL_CAL: "Calcário", COL_GES: "Gesso", COL_S0: "S elementar", COL_P: "P2O5",
                     COL_K: "KCl"}.get(nome, "Produto P"), {})
                delta = None
                a = info.get("ajuste")
                if a is not None and a.ativo() and info.get("media_regra"):
                    delta = f"{(info['fator'] - 1) * 100:+.0f}% vs. regra"
                col.metric(nome.replace(" (kg/ha)", "") + " · média",
                           f"{br(s.mean())} kg/ha" if len(s) else "—", delta=delta, delta_color="off",
                           help=f"Faixa: {br(s.min())}–{br(s.max())} kg/ha" if len(s) else None)

            if COL_K1 in res:
                st.caption(f"KCl parcelado: 1ª aplicação {pct1:.0f}% (média {br(rec[COL_K1].mean())} kg/ha) + "
                           f"2ª aplicação {100 - pct1:.0f}% (média {br(rec[COL_K1.replace('1ª', '2ª')].mean())} kg/ha)")
            vis = ["N Lab", "Identificação", "Talhão", "Profundidade (cm)", "Abertura"] + doses + ["Observações"]
            st.dataframe(estilo_linhas(res[vis], res[COL_SUB]), hide_index=True, width="stretch",
                         height=min(38 + 35 * len(res), 560))
            if n_sub:
                st.markdown(f'<span class="sub-legenda"></span>Amostras de 20-40 cm: exibidas para '
                            'consulta, mas <b>fora do cálculo e da planilha de saída</b>.',
                            unsafe_allow_html=True)

        with aba_int:
            medias = it.medias_por_talhao(laudo)
            if medias.empty:
                st.info("Sem amostras de 0-20 cm para interpretar.")
            else:
                st.markdown("##### Situação por talhão (camada 0-20 cm)")
                st.image(png_interpretacao("mapa", medias), width="stretch")
                st.markdown("##### Pontos de atenção")
                for t, itens in it.pontos_de_atencao(medias).items():
                    txt = ", ".join(itens) if itens else "sem atributos em classe baixa ou crítica"
                    st.markdown(f"- **{t}** ({medias.set_index('Talhão').loc[t, 'Textura'].lower()}): {txt}")
                st.markdown("##### Panorama da fertilidade")
                st.caption("Média de cada talhão sobre a escala de 8 classes (mais à direita = teor mais alto; "
                           "↓ = atributo em que valores menores são melhores). Para pH, V%, Ca na CTC e B, "
                           "'Muito alto' pode indicar excesso.")
                st.image(png_interpretacao("panorama", medias), width="stretch")
                st.markdown("##### Distribuição das amostras por classe")
                st.image(png_interpretacao("classes", laudo.dados), width="stretch")
                if all(c in medias for c in ("ca", "mg", "k", "hal")):
                    st.markdown("##### Equilíbrio de bases")
                    st.image(png_interpretacao("bases", medias, 100 * par.calcario.alvo_ca, 100 * par.calcario.alvo_mg),
                             width="stretch")
                st.markdown("##### Doses médias recomendadas por talhão")
                st.image(png_interpretacao("doses", sem_attrs(res), tuple(doses_totais)), width="stretch")
                medias_sub = it.medias_por_talhao(laudo, subsuperficial=True)
                if not medias_sub.empty:
                    st.markdown("##### Camada subsuperficial (20-40 cm) — informativa")
                    st.image(png_interpretacao("mapa", medias_sub), width="stretch")

        with aba_book:
            aba_montar_book(laudo, par, abertura, ajustes, k0, nome_arq, modo)


def credencial_earthdata() -> dict | None:
    """Login Earthdata (NASA) para ler o NISAR, vindo dos Secrets: `earthdata_token`, ou `earthdata_user` e
    `earthdata_password`. Sem credencial, a página de água sai com o balanço hídrico e avisa sobre o NISAR."""
    tok, usu, sen = segredo("earthdata_token"), segredo("earthdata_user"), segredo("earthdata_password")
    if tok:
        return {"token": str(tok)}
    if usu and sen:
        return {"usuario": str(usu), "senha": str(sen)}
    return None


def diagnostico_cache(kml: bytes, nome_arq: str, indices: tuple, modulos: tuple, anos_lav: int, anos_term: int,
                      nome: str, progresso=None, par_agua=None):
    from motor.sat.base import area_de_kml
    from motor.sat.diagnostico import executar
    area = area_de_kml(kml, nome_arq, nome, list(indices) or None)
    return executar(area, modulos, anos_lav, anos_term, progresso=progresso, par_agua=par_agua,
                    earthdata=credencial_earthdata())


def aba_satelites():
    """Diagnóstico do talhão por satélite a partir apenas do perímetro (KML/KMZ)."""
    from motor.book import MARCAS
    from motor.geo import ler_kml
    from motor.sat import exportar, relatorio
    from motor.sat.diagnostico import MODULOS
    from motor.sat.lavoura import kml_pontos

    st.markdown("### 🛰️ Diagnóstico do talhão por satélite")
    st.caption("Envie só o **perímetro** do talhão (KML ou KMZ). A plataforma consulta bases públicas e monta a "
               "biografia da área, o comportamento da lavoura, o relevo e a água, o perfil climático, o contexto do "
               "solo, a temperatura de superfície e o vento — com relatório em PDF.")
    arq = st.file_uploader("Perímetro do talhão (KML ou KMZ)", type=["kml", "kmz"], key="sat_kml")
    if not arq:
        return
    try:
        k = ler_kml(arq.getvalue(), arq.name)
    except Exception as e:  # noqa: BLE001
        st.error(f"Não foi possível ler o arquivo: {e}")
        return
    if not k.poligonos:
        st.error("O arquivo não tem polígono de perímetro.")
        return
    indices: tuple = ()
    if len(k.poligonos) > 1:
        from motor.geo import epsg_utm_sirgas, projetar
        c = k.poligonos[0].representative_point()
        ep = epsg_utm_sirgas(c.x, c.y)
        rot = [f"Polígono {i + 1} · {projetar(p, ep).area / 1e4:.1f} ha" for i, p in enumerate(k.poligonos)]
        sel = st.multiselect("O arquivo tem vários polígonos: quais formam o talhão?", rot, default=rot,
                             key="sat_polys")
        if not sel:
            return
        indices = tuple(rot.index(r) for r in sel)
    nome_padrao = re.sub(r"\.(kml|kmz)$", "", arq.name, flags=re.I).replace("_", " ")
    with st.form("form_sat"):
        c1, c2, c3 = st.columns(3)
        marca = c1.radio("Relatório para", list(MARCAS), horizontal=True, index=list(MARCAS).index("Atria"))
        produtor = c2.text_input("Produtor")
        propriedade = c3.text_input("Propriedade")
        c4, c5 = st.columns(2)
        talhao = c4.text_input("Talhão", nome_padrao)
        municipio = c5.text_input("Município (UF)")
        mods = st.multiselect("Análises", list(MODULOS), default=list(MODULOS), format_func=MODULOS.get)
        c6, c7 = st.columns(2)
        anos_lav = c6.select_slider("Anos de imagens Sentinel-2 (lavoura)", [2, 3, 4, 5, 6], value=5,
                                    help="Os mapas de NDVI, NDRE, NDMI e MSAVI2 são a média histórica desses anos.")
        anos_term = c7.select_slider("Anos de imagens Landsat (temperatura)", [2, 3, 4, 5], value=3)
        from motor.sat import agua as ag
        with st.expander("💧 Água no solo — informações opcionais (cultura, solo e irrigação)"):
            st.caption("O perímetro basta para o balanço de chuva e ET₀ dos últimos 90 dias. O **armazenamento na zona "
                       "radicular** só é estimado se você informar cultura, data de semeadura, profundidade radicular e "
                       "CAD. O relatório mostra separadamente o que é dado público e o que foi informado aqui.")
            w1, w2, w3 = st.columns(3)
            cult = w1.selectbox("Cultura", [""] + list(ag.CULTURAS), key="sat_ag_cult",
                                format_func=lambda k: ag.CULTURAS[k][0] if k else "(não informar)")
            semead = w2.date_input("Data de semeadura", value=None, format="DD/MM/YYYY", key="sat_ag_sem")
            ciclo = w3.number_input("Ciclo (dias) — opcional", 0, 400, 0, 5, key="sat_ag_ciclo",
                                    help="0 = usar a duração de referência da FAO-56 para a cultura.")
            w4, w5, w6 = st.columns(3)
            raiz = w4.number_input("Profundidade radicular efetiva (cm)", 0.0, 300.0, 0.0, 5.0, key="sat_ag_raiz",
                                   help="0 = não informar.")
            cad = w5.number_input("CAD (mm de água por cm de solo)", 0.0, 3.0, 0.0, 0.1, format="%.2f",
                                  key="sat_ag_cad",
                                  help="Capacidade de água disponível do solo. 0 = não informar. Referência: "
                                       "arenoso ~0,6–0,9; textura média ~1,0–1,4; argiloso ~1,3–1,8 mm/cm.")
            irrig = w6.selectbox("Irrigação", list(ag.IRRIGACAO), format_func=ag.IRRIGACAO.get, key="sat_ag_irr",
                                 help="“Não informado” não é tratado como “sem irrigação”: o armazenamento sai como "
                                      "cenário sem irrigação, identificado como tal.")
            st.caption("Lâminas de irrigação aplicadas nos últimos 90 dias (só para área irrigada; opcional):")
            lam = st.data_editor(pd.DataFrame({"Data": pd.Series([], dtype="datetime64[ns]"),
                                               "Lâmina (mm)": pd.Series([], dtype=float)}),
                                 num_rows="dynamic", hide_index=True, key="sat_ag_lam",
                                 column_config={"Data": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                                "Lâmina (mm)": st.column_config.NumberColumn(min_value=0.0,
                                                                                             max_value=200.0)})
            st.caption("🛰️ NISAR: " + ("credencial Earthdata configurada — a umidade superficial será lida."
                                       if credencial_earthdata() else
                                       "credencial Earthdata **não configurada** nos Secrets — o relatório informa "
                                       "quantas passagens existem, mas não lê os valores (ver README)."))
        gerar = st.form_submit_button("🛰️ Gerar diagnóstico", type="primary", width="stretch")
    if gerar:
        if not mods:
            st.warning("Escolha ao menos uma análise.")
            return
        barra = st.status("Consultando as bases (pode levar alguns minutos)…", expanded=True)
        try:
            laminas = tuple((pd.Timestamp(d).date(), float(mm)) for d, mm in zip(lam["Data"], lam["Lâmina (mm)"])
                            if pd.notna(d) and pd.notna(mm) and mm > 0)
            par_agua = ag.ParametrosAgua(cultura=cult or "", semeadura=semead, ciclo_dias=int(ciclo) or None,
                                         prof_raiz_cm=float(raiz) or None, cad_mm_cm=float(cad) or None,
                                         irrigacao=irrig, laminas=laminas if irrig == "irrigado" else ())
            diag = diagnostico_cache(arq.getvalue(), arq.name, indices, tuple(mods), anos_lav, anos_term, talhao,
                                     progresso=barra.write, par_agua=par_agua)
            barra.write("Montando o relatório…")
            fundo = relatorio.fundo_satelite(diag.area, segredo("google_maps_key") and "Google (Map Tiles API)"
                                             or "Esri World Imagery", segredo("google_maps_key"))
            dados = relatorio.DadosRelatorio(marca=marca, produtor=produtor, propriedade=propriedade, talhao=talhao,
                                             municipio=municipio, data=pd.Timestamp.today().strftime("%d/%m/%Y"))
            with it.TRAVA_MPL:
                pdf = relatorio.montar(diag, dados, fundo)
            st.session_state["sat_res"] = {"pdf": pdf, "diag": diag, "nome": talhao or nome_padrao}
            barra.update(label="Pronto!", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            barra.update(label="Falhou", state="error")
            st.exception(e)
    r = st.session_state.get("sat_res")
    if not r:
        return
    diag = r["diag"]
    for a in diag.avisos:
        st.caption(f"⚠️ {a}")
    ind = relatorio.indicadores(diag)
    cols = st.columns(4)
    for i, (rot, val, sub) in enumerate(ind):
        cols[i % 4].metric(rot, val, help=sub)
    al = relatorio.alertas(diag)
    if al:
        st.markdown("##### Pontos de atenção")
        for a in al:
            st.markdown(f"- {a}")
    b1, b2, b3 = st.columns(3)
    b1.download_button("📕 Relatório (PDF)", r["pdf"], file_name=f"Diagnostico satelite - {r['nome']}.pdf",
                       mime="application/pdf", type="primary", width="stretch", key="sat_pdf")
    b2.download_button("📊 Tabelas (Excel)", adiado(exportar.excel, diag),
                       file_name=f"Diagnostico satelite - {r['nome']}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                       key="sat_xlsx")
    if diag.lavoura is not None and len(diag.lavoura.pontos):
        b3.download_button("📍 Pontos de visita (KML)", kml_pontos(diag.lavoura.pontos),
                           file_name=f"Pontos prioritarios - {r['nome']}.kml",
                           mime="application/vnd.google-earth.kml+xml", width="stretch", key="sat_kml_pts")
    with st.expander("Tabelas"):
        for nome, df in exportar.tabelas(diag).items():
            st.markdown(f"**{nome}**")
            st.dataframe(df, hide_index=True, width="stretch")


def laudo_unico(arquivos):
    """Lê um ou mais arquivos de laudo e devolve (laudo, nome base); vários arquivos são juntados."""
    lidos = []
    for arq in arquivos:
        try:
            lidos.append((arq.name, ler_laudo_cache(arq.getvalue(), arq.name)))
        except Exception as e:  # noqa: BLE001
            st.error(f"Não foi possível ler {arq.name}: {e}")
    if not lidos:
        return None, ""
    if len(lidos) == 1:
        nome, lau = lidos[0]
        return lau, re.sub(r"\.(xlsx|xls|csv)$", "", nome, flags=re.I)
    lau, _ = juntar_laudos([l for _, l in lidos], [n for n, _ in lidos])
    prod, faz = identificacao(lau)
    return lau, (faz or prod or "Laudos juntados").strip().title()


def excel_sementes(res, resumo, faixas=None) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        resumo.to_excel(w, sheet_name="Resumo", index=False)
        if faixas is not None and len(faixas):
            faixas.to_excel(w, sheet_name="Doses por faixa", index=False)
        res.amostras.to_excel(w, sheet_name="Índice por amostra", index=False)
        pd.DataFrame([{"Atributo": k, "Máximo (=100%)": v} for k, v in res.maximos.items()]).to_excel(
            w, sheet_name="Máximos", index=False)
        for ws in w.sheets.values():
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(max(12, len(str(col[0].value)) + 2), 34)
    return buf.getvalue()


def aba_sementes():
    """Mapa de semeadura em taxa variável: índice de fertilidade (P, M.O., argila, SB) → sementes/m."""
    from motor import sementes as sm
    from motor.book import MARCAS, DadosBook
    from motor import book_sementes
    from motor.prescricao import superficies

    st.markdown("### 🌱 Mapa de semeadura em taxa variável")
    st.caption("Índice de fertilidade = (P% × 5 + M.O.% × 20 + argila% × 40 + SB% × 35) ÷ 100, com cada atributo em "
               "% do maior valor das amostras (sem argila, SB com peso 75). Onde o solo é **mais fértil caem menos "
               "sementes**; a média do mapa fica igual à média comprada. *Regra da equipe em avaliação: deixe faixas "
               "com taxa fixa para comparar na colheita.*")
    arqs = st.file_uploader("Laudo(s) de análise de solo", type=["xlsx", "xls", "csv"], accept_multiple_files=True,
                            key="sem_laudos")
    if not arqs:
        return
    laudo, base_nome = laudo_unico(arqs)
    if laudo is None:
        return
    faltam = [n for c, n in (("p", "P"), ("mo", "M.O."), ("ca", "Ca"), ("mg", "Mg"), ("k", "K"))
              if c not in laudo.dados or laudo.dados[c].notna().sum() == 0]
    if faltam:
        st.error(f"O laudo não tem {', '.join(faltam)} — necessários para o índice de fertilidade.")
        return
    if "argila" not in laudo.dados or sm.maximos(laudo.dados)["argila"] != sm.maximos(laudo.dados)["argila"]:
        st.info("Laudo sem argila medida: a soma de bases entra com peso 75 (regra da equipe).", icon="🔎")
    st.markdown("Anexe os **KML/KMZ de perímetro e de pontos** de cada talhão (mesma regra do book).")
    proj = ligar_kmls(laudo, "sem")
    if proj is None:
        return
    d0 = laudo.dados
    with st.form("form_sem"):
        c1, c2, c3 = st.columns(3)
        marca = c1.radio("Book para", list(MARCAS), horizontal=True, index=list(MARCAS).index("Atria"))
        produtor = c2.text_input("Produtor", next((x for x in d0["proprietario"] if x), "").strip().title())
        propriedade = c3.text_input("Propriedade", next((x for x in d0["propriedade"] if x), "").strip().title())
        c4, c5, c6, c7 = st.columns(4)
        cultura = c4.text_input("Cultura e safra", "Soja 2026/27")
        media = c5.number_input("Sementes/m compradas (média)", 1.0, 40.0, 13.0, 0.1, format="%.1f",
                                help="Média que o produtor comprou para a área; o mapa mantém essa média. "
                                     "Ex.: soja 13, algodão 8, milho 3.")
        espac = c6.number_input("Espaçamento entre linhas (m)", 0.2, 1.0, 0.5, 0.05)
        var = c7.slider("Variação máxima em torno da média (±%)", 5, 40, 20, 5,
                        help="Limita a taxa mais baixa e a mais alta do mapa.")
        c8, c9, c10 = st.columns(3)
        classes = c8.select_slider("Número de doses (faixas) no mapa", list(range(3, 16)), value=8,
                                   help="Quantas doses diferentes o mapa pode ter. O talhão só chega a esse número se "
                                        "a diferença entre a menor e a maior taxa comportar tantos degraus — veja "
                                        "“Degrau entre doses” e “Amplitude”.")
        res_m = c9.select_slider("Tamanho do pixel (m)", [5, 10, 15, 20], value=10)
        unidade = c10.radio("Unidade no shapefile", ["sementes/m", "sementes/ha"], horizontal=True,
                            help="Confira a unidade que o monitor da plantadeira espera.")
        c11, c12 = st.columns(2)
        passo_txt = c11.selectbox("Degrau entre doses (sementes/m)", ["Automático", "0,1", "0,05", "0,02", "0,01"],
                                  help="Menor diferença entre duas doses. Automático: 0,1 para médias a partir de 8 "
                                       "sementes/m (soja), 0,05 entre 4 e 8 e 0,02 abaixo de 4 (milho). Degrau menor = "
                                       "mais doses possíveis quando a taxa varia pouco.")
        amplitude = c12.radio("Amplitude das doses", ["Proporcional ao índice", "Usar toda a faixa (± variação)"],
                              horizontal=True,
                              help="Proporcional: taxa = k ÷ índice de fertilidade; se o índice varia pouco no talhão, "
                                   "as doses ficam próximas (poucas faixas). Toda a faixa: o lugar mais fértil recebe a "
                                   "média − variação máxima e o menos fértil a média + variação máxima — mais contraste "
                                   "e mais doses, mantendo a média comprada.")
        com_indice = st.checkbox("Incluir no PDF a página do índice de fertilidade (como a taxa foi calculada)",
                                 value=False)
        st.caption("Média específica por talhão (deixe em branco para usar a média geral):")
        tab_med = st.data_editor(pd.DataFrame({"Talhão": [t.nome for t in proj.talhoes],
                                               "Sementes/m": [None] * len(proj.talhoes)}).astype({"Sementes/m": float}),
                                 hide_index=True, disabled=["Talhão"], key="sem_medias",
                                 column_config={"Sementes/m": st.column_config.NumberColumn(
                                     "Sementes/m", min_value=1.0, max_value=40.0, step=0.1)})
        gerar = st.form_submit_button("🌱 Gerar mapa de semeadura", type="primary", width="stretch",
                                      disabled=not getattr(proj, "liberado", True))
    if gerar:
        medias = {r["Talhão"]: float(r["Sementes/m"]) for _, r in tab_med.iterrows() if pd.notna(r["Sementes/m"])}
        cfg = sm.ConfigSementes(media=float(media), variacao_max=float(var), classes=int(classes),
                                espacamento=float(espac), unidade_shape=unidade, medias_talhao=medias,
                                passo=0.0 if passo_txt == "Automático" else float(passo_txt.replace(",", ".")),
                                esticar=amplitude.startswith("Usar"))
        barra = st.status("Gerando…", expanded=True)
        try:
            barra.write("Interpolando P, M.O., argila e bases (krigagem)…")
            mapas = superficies(proj, float(res_m), sm.ATRIB_SEMENTES, modo="fiel")   # mesmo padrão dos books
            barra.write("Calculando índice de fertilidade e taxas…")
            res = sm.gerar(mapas, laudo.dados, cfg)
            dados = DadosBook(marca=marca, produtor=produtor, propriedade=propriedade,
                              titulo_capa=("MAPA DE SEMEADURA", f"EM TAXA VARIÁVEL · {cultura}".upper().strip(" ·")))
            st.session_state["sem_res"] = {"mapas": mapas, "epsg": proj.epsg, "cfg": cfg, "res": res, "proj": proj,
                                           "dados": dados, "cultura": cultura, "com_indice": com_indice,
                                           "rec": {m.talhao.nome: m.zonas["Sementes"] for m in mapas
                                                   if "Sementes" in m.zonas},
                                           "manuais": {}, "versao": 1,
                                           "assinatura": assinatura_entradas(laudo, proj, None, None, {}),
                                           "nome": f"{propriedade or base_nome}"}
            barra.update(label="Pronto!", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            barra.update(label="Falhou", state="error")
            st.exception(e)
    r = st.session_state.get("sem_res")
    if not r:
        return
    if r.get("assinatura") != assinatura_entradas(laudo, proj, None, None, {}):
        st.warning("⚠️ **Resultado desatualizado:** o laudo ou os KMLs mudaram depois que este mapa foi gerado. "
                   "Gere novamente antes de baixar.")
    mapas, epsg, cfg = r["mapas"], r["epsg"], r["cfg"]
    for a in r["res"].avisos:
        st.caption(f"⚠️ {a}")
    resumo = sm.tabela_resumo(mapas, cfg)
    st.dataframe(resumo, hide_index=True, width="stretch")
    poucas = resumo[resumo["Faixas"] < cfg.classes]
    if len(poucas) and not r["manuais"]:
        st.caption(f"ℹ️ {len(poucas)} talhão(ões) ficaram com menos de {cfg.classes} doses: a diferença entre a menor e "
                   "a maior taxa não comporta mais degraus. Para ter mais doses, diminua o **degrau entre doses** ou "
                   "use **toda a faixa** em “Amplitude das doses” e gere de novo.")

    # ---- ajuste manual: o usuário vê o que o sistema recomendou e troca as doses que quiser
    with st.expander("✏️ Ajustar as doses à mão (opcional)", expanded=bool(r["manuais"])):
        st.caption("Cada linha é uma dose do mapa. **Recomendado** é o que o sistema calculou; edite **Ajustado** onde "
                   "quiser outra taxa. Os polígonos não mudam — só a dose de cada faixa. O PDF e os shapefiles saem "
                   "com as doses ajustadas, e o PDF registra o que foi alterado.")
        faixas = sm.tabela_faixas(mapas, r["rec"])
        casas = cfg.casas()
        with st.form("form_sem_ajuste"):
            ed = st.data_editor(
                faixas, hide_index=True, width="stretch", key=f"sem_faixas_{r['versao']}",
                disabled=["Talhão", "Área (ha)", "Recomendado (sem/m)"],
                column_config={"Recomendado (sem/m)": st.column_config.NumberColumn(format=f"%.{casas}f"),
                               "Ajustado (sem/m)": st.column_config.NumberColumn(
                                   format=f"%.{casas}f", min_value=0.5, max_value=60.0,
                                   step=10 ** -casas)})
            a1, a2 = st.columns(2)
            aplicar = a1.form_submit_button("Aplicar doses ajustadas", type="primary", width="stretch")
            voltar = a2.form_submit_button("Voltar ao recomendado", width="stretch")
        if aplicar or voltar:
            r["manuais"] = {}
            for m in mapas:
                rec = r["rec"].get(m.talhao.nome)
                if rec is None:
                    continue
                novo = {}
                if aplicar:
                    t = ed[ed["Talhão"] == m.talhao.nome]
                    novo = {float(a): float(b) for a, b in zip(t["Recomendado (sem/m)"], t["Ajustado (sem/m)"])
                            if pd.notna(b)}
                alterados = sm.ajustar_doses(m, rec, novo)
                if alterados:
                    r["manuais"][m.talhao.nome] = alterados
            r["versao"] += 1
            st.rerun()
        if r["manuais"]:
            total_c = sum(cfg.por_ha(cfg.media_de(m.talhao.nome)) * m.zonas["Sementes"].area_ha for m in mapas
                          if "Sementes" in m.zonas)
            total_m = sum(cfg.por_ha(m.zonas["Sementes"].dose_media) * m.zonas["Sementes"].area_ha for m in mapas
                          if "Sementes" in m.zonas)
            dif = 100 * (total_m - total_c) / total_c if total_c else 0.0
            st.info(f"Doses ajustadas à mão em {len(r['manuais'])} talhão(ões). Total de sementes com o ajuste: "
                    f"**{br(total_m / 1e6, 2)} milhões** — {'+' if dif >= 0 else '−'}{abs(dif):.1f}% em relação ao "
                    f"comprado ({br(total_c / 1e6, 2)} milhões).")

    def montar_pdf():
        return book_sementes.montar(r["proj"], mapas, r["dados"], cfg, r["res"], cultura=r["cultura"],
                                    com_indice=r["com_indice"], manuais=r["manuais"])

    b1, b2, b3 = st.columns(3)
    b1.download_button("📕 Mapa de semeadura (PDF)", adiado(montar_pdf), file_name=f"Semeadura - {r['nome']}.pdf",
                       mime="application/pdf", type="primary", width="stretch", key="sem_pdf")
    b2.download_button(f"🗂️ Shapefiles ({cfg.unidade_shape}, .zip)", adiado(sm.zip_sementes, mapas, epsg, cfg),
                       file_name=f"Semeadura - {r['nome']}.zip", mime="application/zip", width="stretch",
                       key="sem_zip")
    b3.download_button("📊 Índice por amostra (Excel)", adiado(excel_sementes, r["res"], resumo, faixas),
                       file_name=f"Semeadura - {r['nome']}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                       key="sem_xlsx")
    ajuda_pastas_monitor()


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=24)
def vigor_cache(wkb_ll: bytes, anos: int):
    """Vigor relativo histórico (NDVI Sentinel-2) de um perímetro. Devolve (V, máscara, xs, ys, anos) ou None."""
    from shapely import wkb

    from motor import amostragem as am
    from motor.geo import epsg_utm_sirgas, projetar
    from motor.sat import lavoura
    from motor.sat.base import Area
    ll = wkb.loads(wkb_ll)
    c = ll.representative_point()
    epsg = epsg_utm_sirgas(c.x, c.y)
    L = lavoura.obter(Area("talhão", ll, projetar(ll, epsg), epsg), anos=anos)
    v = am.vigor_da_lavoura(L)
    return None if v is None else (*v, list(L.vigor_rel))


def aba_amostragem():
    """Pontos de amostragem de solo a partir do perímetro: grade ou zonas de manejo (NDVI histórico)."""
    from motor import amostragem as am
    from motor import book_amostragem
    from motor.book import MARCAS
    from motor.geo import ler_kml, projetar

    st.markdown("### 📍 Pontos de amostragem")
    st.caption("Envie o **perímetro** (KML ou KMZ; um arquivo por talhão ou vários polígonos no mesmo arquivo). "
               "A plataforma monta os pontos em **grade** (você escolhe a densidade) ou por **zonas de manejo** "
               "(pelo histórico de NDVI) e entrega o mapa em PDF e os pontos em KML, GPX, shapefile e Excel. "
               "Os KMLs já saem no padrão `Pontos_TH_1.kml`, prontos para usar depois na aba de recomendação.")
    arqs = st.file_uploader("Perímetro(s) dos talhões (KML ou KMZ)", type=["kml", "kmz"], accept_multiple_files=True,
                            key="am_kml")
    if not arqs:
        return
    try:
        kmls = [ler_kml(a.getvalue(), a.name) for a in arqs]
        talhoes = am.talhoes_de_kmls([k for k in kmls if k.poligonos])
    except Exception as e:  # noqa: BLE001
        st.error(f"Não foi possível ler o(s) arquivo(s): {e}")
        return
    if not talhoes:
        st.error("Os arquivos não têm polígono de perímetro.")
        return
    st.caption("Talhões: " + " · ".join(f"**{t.nome}** {t.area_ha:.1f} ha" for t in talhoes)
               + f" — total {sum(t.area_ha for t in talhoes):.1f} ha")
    metodo = st.radio("Método", ["Grade", "Zonas de manejo (NDVI histórico)"], horizontal=True, key="am_metodo")
    por_zonas = metodo.startswith("Zonas")
    with st.form("form_am"):
        c1, c2, c3 = st.columns(3)
        marca = c1.radio("Mapa para", list(MARCAS), horizontal=True, index=list(MARCAS).index("Atria"))
        propriedade = c2.text_input("Propriedade")
        ha = c3.number_input("Hectares por ponto", 0.25, 50.0, 5.0, 0.25,
                             help="Densidade da amostragem: 5 ha por ponto = célula de 224 × 224 m; "
                                  "2 ha = 141 × 141 m; 1 ha = 100 × 100 m.")
        if por_zonas:
            c4, c5, c6, c7 = st.columns(4)
            n_zonas = c4.select_slider("Zonas", [2, 3, 4, 5], value=3)
            anos = c5.select_slider("Anos de imagens", [3, 4, 5, 6], value=5,
                                    help="Quanto mais anos, mais o mapa mostra o que se repete no talhão e menos o "
                                         "efeito de uma safra só.")
            min_zona = c6.number_input("Mínimo de pontos por zona", 1, 20, 2)
            area_min = c7.number_input("Menor mancha (ha)", 0.25, 20.0, 1.0, 0.25,
                                       help="Manchas menores que isso são incorporadas à zona vizinha.")
            st.caption("As zonas vêm do **vigor relativo** da lavoura: em cada ano agrícola, o NDVI do pico de cada "
                       "ponto comparado com a mediana do talhão; depois, a média dos anos. Divisão em zonas de áreas "
                       "iguais. Requer internet (imagens Sentinel-2) e leva de 1 a 3 minutos por talhão.")
        else:
            c4, c5, c6 = st.columns(3)
            formato = c4.selectbox("Formato", list(am.FORMATOS), format_func=am.FORMATOS.get,
                                   help="Desencontrada (triangular): linhas alternadas deslocadas meia célula — para a "
                                        "mesma densidade, nenhum lugar do talhão fica tão longe de um ponto.")
            posicao = c5.selectbox("Posição do ponto", list(am.POSICOES), format_func=am.POSICOES.get,
                                   help="Aleatória: sorteia o ponto dentro de cada célula (evita coincidir sempre com "
                                        "a mesma linha de plantio ou de tráfego).")
            alinhar = c6.selectbox("Direção da malha", ["Lado mais comprido do talhão", "Norte–Sul", "Outro ângulo"])
            angulo_txt = st.number_input("Ângulo da malha (graus a partir do Leste, sentido anti-horário)", 0.0, 179.0,
                                         0.0, 1.0, help="Só é usado com 'Outro ângulo'. Ex.: direção das linhas de "
                                                        "plantio.")
        c8, c9, c10 = st.columns(3)
        borda = c8.number_input("Distância mínima da borda (m)", 0.0, 100.0, 20.0, 5.0,
                                help="Evita pontos em cabeceiras, carreadores e bordaduras.")
        num_talhao = c9.checkbox("Numerar por talhão", value=True,
                                 help="Marcado: cada talhão tem pontos 1, 2, 3… (padrão dos laudos por talhão). "
                                      "Desmarcado: numeração corrida na fazenda.")
        fundo_img = c10.checkbox("Imagem de satélite no mapa", value=True)
        gerar = st.form_submit_button("📍 Gerar pontos", type="primary", width="stretch")
    if gerar:
        barra = st.status("Gerando…", expanded=True)
        try:
            if por_zonas:
                vigores, anos_usados = [], []
                for t in talhoes:
                    barra.write(f"{t.nome}: consultando imagens Sentinel-2 ({anos} anos)…")
                    v = vigor_cache(projetar(t.poligono, t.epsg, inverso=True).wkb, int(anos))
                    if v is None:
                        raise RuntimeError(f"{t.nome}: não há imagens suficientes sem nuvens (ou sem internet) para "
                                           "montar as zonas. Use a grade ou tente mais anos.")
                    vigores.append(v[:4])
                    anos_usados = v[4]
                plano = am.plano_zonas(talhoes, vigores, int(n_zonas), float(ha), int(min_zona), float(area_min),
                                       float(borda), num_talhao, anos=anos_usados)
                grade = None
            else:
                ang = None if alinhar.startswith("Lado") else (90.0 if alinhar.startswith("Norte") else
                                                               float(angulo_txt))
                grade = {"ha_por_ponto": float(ha), "formato": formato, "angulo": ang, "posicao": posicao,
                         "margem_borda": float(borda), "numerar_por_talhao": num_talhao}
                plano = am.plano_grade(talhoes, **grade)
            sat = None
            if fundo_img:
                barra.write("Baixando imagem de satélite…")
                chave = segredo("google_maps_key")
                sat = book_amostragem.fundo_satelite(plano, "Google (Map Tiles API)" if chave else "Esri World Imagery",
                                                     chave)
            st.session_state["am_res"] = {"plano": plano, "sat": sat, "marca": marca, "propriedade": propriedade,
                                          "nome": propriedade or "Amostragem", "grade": grade, "talhoes": talhoes,
                                          "desloc": [0.0, 0.0], "giro": 0.0, "versao": 1, "recusas": [],
                                          "num_talhao": num_talhao,
                                          "arquivos": tuple((a.name, a.size) for a in arqs)}
            barra.update(label="Pronto!", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            barra.update(label="Falhou", state="error")
            st.error(str(e)) if isinstance(e, (RuntimeError, ValueError)) else st.exception(e)
    r = st.session_state.get("am_res")
    if not r:
        return
    if r.get("arquivos") != tuple((a.name, a.size) for a in arqs):
        st.warning("⚠️ **Resultado desatualizado:** os perímetros mudaram depois que estes pontos foram gerados. "
                   "Gere novamente antes de baixar.")

    def refazer_malha():
        """Regera a grade com o deslocamento e o giro guardados (edições manuais anteriores são descartadas)."""
        r["plano"] = am.plano_grade(r["talhoes"], **r["grade"], deslocamento=tuple(r["desloc"]), giro=r["giro"])
        r["versao"] += 1

    plano = r["plano"]
    st.markdown("#### 🗺️ Prévia e ajustes")
    st.caption("Confira a distribuição antes de baixar. **Mover ponto:** arraste o ponto. "
               + ("**Mover malha:** arraste o mapa e todos os pontos andam juntos. " if r["grade"] else "")
               + "**Incluir / Remover:** clique no mapa ou no ponto. As alterações só valem depois de **✔ Aplicar** — "
                 "e os arquivos abaixo passam a sair com elas.")
    from motor import editor_mapa
    res = editor_mapa.componente()(data={**am.dados_do_mapa(plano), "versao": r["versao"], "malha": bool(r["grade"])},
                                   key="am_editor", on_edicao_change=lambda: None)
    ed = getattr(res, "edicao", None)
    if ed and ed.get("versao") == r["versao"]:
        malha = [float(v) for v in (ed.get("malha") or [0, 0])]
        manuais = bool(ed.get("mover") or ed.get("remover") or ed.get("adicionar"))
        r["recusas"] = []
        if r["grade"] and (abs(malha[0]) >= 0.5 or abs(malha[1]) >= 0.5):
            r["desloc"] = [r["desloc"][0] + malha[0], r["desloc"][1] + malha[1]]
            refazer_malha()
            if manuais:
                r["recusas"].append("A malha foi refeita na nova posição; os ajustes de pontos feitos no mesmo passo "
                                    "não foram aplicados — refaça-os agora, sobre a malha nova.")
        else:
            r["recusas"] = am.editar_pontos(plano, ed.get("mover"), ed.get("remover"), ed.get("adicionar"),
                                            r["num_talhao"])
            r["versao"] += 1
        st.rerun()
    for msg in r.get("recusas", []):
        st.warning(msg)
    if r["grade"]:
        with st.expander("Posição da malha (valores exatos)"):
            with st.form("form_am_malha"):
                g1, g2, g3 = st.columns(3)
                d_l = g1.number_input("Deslocar para leste (m)", -2000.0, 2000.0, float(round(r["desloc"][0])), 5.0,
                                      help="Negativo = para oeste.")
                d_n = g2.number_input("Deslocar para norte (m)", -2000.0, 2000.0, float(round(r["desloc"][1])), 5.0,
                                      help="Negativo = para sul.")
                giro = g3.number_input("Girar a malha (graus)", -90.0, 90.0, float(r["giro"]), 1.0,
                                       help="Positivo = sentido anti-horário, a partir da direção atual.")
                h1, h2 = st.columns(2)
                aplicar = h1.form_submit_button("Aplicar posição", width="stretch")
                zerar = h2.form_submit_button("Voltar à malha automática", width="stretch")
            st.caption("A malha é refeita na nova posição: as células são recortadas de novo pelo perímetro e os "
                       "ajustes manuais de pontos são descartados.")
            if aplicar or zerar:
                r["desloc"], r["giro"] = ([0.0, 0.0], 0.0) if zerar else ([float(d_l), float(d_n)], float(giro))
                r["recusas"] = []
                refazer_malha()
                st.rerun()
    m1, m2, m3 = st.columns(3)
    m1.metric("Pontos", len(plano.pontos))
    m2.metric("Área", f"{plano.area_ha:.1f} ha".replace(".", ","))
    m3.metric("Densidade", f"{plano.ha_por_ponto:.2f} ha/ponto".replace(".", ","))
    st.caption(plano.descricao)
    for a in plano.avisos:
        st.caption(f"⚠️ {a}")
    col = ["Talhão"] + (["Zona"] if plano.zonas else [])
    st.dataframe(plano.pontos.groupby(col, sort=False).size().rename("Pontos").reset_index(), hide_index=True,
                 width="stretch")
    b1, b2, b3 = st.columns(3)
    b1.download_button("📕 Mapa de amostragem (PDF)",
                       adiado(book_amostragem.montar, plano, r["marca"], r["propriedade"], r["sat"]),
                       file_name=f"Amostragem - {r['nome']}.pdf", mime="application/pdf", type="primary",
                       width="stretch", key="am_pdf")
    b2.download_button("📍 Pontos (KML, GPX, shapefile · .zip)", adiado(am.zip_amostragem, plano),
                       file_name=f"Amostragem - {r['nome']}.zip", mime="application/zip", width="stretch",
                       key="am_zip")
    b3.download_button("📊 Coordenadas (Excel)", adiado(am.excel_pontos, plano),
                       file_name=f"Amostragem - {r['nome']}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                       key="am_xlsx")
    with st.expander("Ver coordenadas"):
        st.dataframe(am.tabela_coordenadas(plano), hide_index=True, width="stretch")


def em_desenvolvimento(icone: str, titulo: str, texto: str) -> None:
    st.markdown(f"### {icone} {titulo}")
    st.info("**Em desenvolvimento =)**", icon="🚧")
    st.caption(texto)


# ------------------------------------------------------------------ abas
aba_rec, aba_comp, aba_sat, aba_sem, aba_am = st.tabs(["🧪 Recomendação e book", "📈 Comparações", "🛰️ Satélites",
                                                       "🌱 Mapa de Sementes", "📍 Amostragem"])
with aba_comp:
    st.markdown("### 📈 Book comparativo entre anos")
    st.caption("Envie o laudo **mais recente** (é dele que saem a recomendação e as prescrições). Na aba "
               "**📈 Montar book comparativo**, anexe os KMLs e o laudo do **ano anterior**: o book mostra os mapas "
               "dos dois anos lado a lado, com as mesmas legendas, as áreas por classe e as médias de cada ano, onde "
               "mudou e os volumes de produto de um ano contra o outro.")
    fluxo_recomendacao("comparacao")
with aba_sat:
    aba_satelites()
with aba_sem:
    aba_sementes()
with aba_am:
    aba_amostragem()
with aba_rec:
    fluxo_recomendacao()
