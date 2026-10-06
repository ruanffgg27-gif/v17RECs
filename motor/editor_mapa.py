"""Editor interativo dos pontos de amostragem (mapa Leaflet dentro do app).

O usuário vê a prévia sobre a imagem de satélite e pode: arrastar um ponto, arrastar a malha inteira, incluir e
remover pontos. As alterações ficam pendentes no navegador até clicar em "Aplicar"; aí voltam para o Python,
que valida (ponto dentro do talhão), refaz a malha ou a numeração e redesenha.

O Leaflet (BSD-2) vai junto com o app em assets/leaflet — não depende de CDN.
"""
from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "leaflet"

HTML = """
<div class="am-editor">
  <div class="am-barra">
    <button type="button" data-modo="mover" class="am-ativo" title="Arraste um ponto para outro lugar do talhão">✋ Mover ponto</button>
    <button type="button" data-modo="malha" title="Arraste em qualquer lugar do mapa para deslocar todos os pontos juntos">⬌ Mover malha</button>
    <button type="button" data-modo="adicionar" title="Clique no mapa para incluir um ponto">＋ Incluir</button>
    <button type="button" data-modo="remover" title="Clique num ponto para removê-lo (clique de novo para manter)">✕ Remover</button>
    <span class="am-espaco"></span>
    <button type="button" data-acao="fundo" title="Liga/desliga a imagem de satélite">🛰️ Satélite</button>
    <button type="button" data-acao="desfazer" title="Descarta as alterações ainda não aplicadas">↺ Desfazer</button>
    <button type="button" data-acao="aplicar" class="am-aplicar" disabled>✔ Aplicar</button>
  </div>
  <div class="am-mapa"></div>
  <div class="am-rodape">Arraste um ponto para reposicionar. Nada muda nos arquivos até você clicar em “Aplicar”.</div>
</div>
"""

CSS = """
.am-editor { border: 2px solid var(--tinta, #1D2A22); border-radius: 10px; overflow: hidden;
  box-shadow: 6px 6px 0 var(--sombra2, #DCEAD6); background: var(--ficha, #FBF8EE); }
.am-barra { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px; border-bottom: 2px solid var(--tinta, #1D2A22); }
.am-espaco { flex: 1 1 12px; }
.am-barra button { font: 700 13px 'Rubik', 'Segoe UI', sans-serif; padding: 6px 10px; border-radius: 7px; cursor: pointer;
  border: 2px solid var(--tinta, #1D2A22); background: var(--cartao, #FFFDF6); color: var(--texto, #1D2A22);
  box-shadow: 3px 3px 0 var(--sombra, #1D2A22); }
.am-barra button:hover:not(:disabled) { background: var(--destaque, #E9B23A); color: var(--destaque-txt, #1D2A22); }
.am-barra button.am-ativo { background: var(--aba-ativa, #104A2A); color: var(--aba-ativa-txt, #fff);
  box-shadow: 1px 1px 0 var(--sombra, #1D2A22); transform: translate(2px, 2px); }
.am-barra button.am-aplicar:not(:disabled) { background: var(--primario, #E25A10); color: var(--primario-txt, #fff); }
.am-barra button:disabled { opacity: .5; cursor: default; box-shadow: 2px 2px 0 var(--inativo, #9AA39C); }
.am-barra button[hidden] { display: none; }
.am-mapa { height: 600px; background: #26302A; }
.am-mapa.am-modo-malha { cursor: move; }
.am-mapa.am-modo-adicionar { cursor: crosshair; }
.am-rodape { padding: 7px 10px; font: 500 13px 'IBM Plex Sans', 'Segoe UI', sans-serif; color: var(--texto, #1D2A22);
  border-top: 2px solid var(--tinta, #1D2A22); min-height: 20px; }
.am-rodape b { color: var(--primario, #E25A10); }
.am-pt { width: 22px; height: 22px; border-radius: 50%; background: #E0203A; border: 2px solid #fff;
  box-shadow: 0 0 0 1px #000, 0 1px 4px rgba(0,0,0,.6); color: #fff; font: 700 10px/18px 'Segoe UI', Arial, sans-serif;
  text-align: center; box-sizing: border-box; }
.am-pt.am-denso { width: 16px; height: 16px; font-size: 8px; line-height: 12px; }
.am-pt.am-movido { background: #FFE23D; color: #111; }
.am-pt.am-novo { background: #1FD1FF; color: #111; }
.am-pt.am-fora { background: #555; opacity: .55; text-decoration: line-through; }
.am-rotulo { background: transparent; border: 0; box-shadow: none; color: #fff; font: 800 14px 'Rubik', 'Segoe UI', sans-serif;
  text-shadow: 0 0 3px #000, 0 0 3px #000; }
.am-rotulo::before { display: none; }
"""

JS = r"""
const ESTADOS = new WeakMap();
const M_LAT = 110574.0;

function resumo(e) {
  const p = [];
  const nm = Object.keys(e.mover).length;
  if (e.malha[0] || e.malha[1]) {
    const f = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(0);
    p.push('malha ' + f(e.malha[0]) + ' m (leste) e ' + f(e.malha[1]) + ' m (norte)');
  }
  if (nm) p.push(nm + (nm === 1 ? ' ponto movido' : ' pontos movidos'));
  if (e.remover.size) p.push(e.remover.size + ' a remover');
  const na = e.adicionar.filter(Boolean).length;
  if (na) p.push(na + ' a incluir');
  return p;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const L = globalThis.L;
  const raiz = parentElement.querySelector('.am-editor');
  if (!raiz || !L || !data) return;
  const elMapa = raiz.querySelector('.am-mapa');
  const rodape = raiz.querySelector('.am-rodape');
  const btAplicar = raiz.querySelector('[data-acao="aplicar"]');
  let st = ESTADOS.get(raiz);

  if (!st) {
    const mapa = L.map(elMapa, { zoomControl: true, attributionControl: true, zoomSnap: 0.25, maxZoom: 20 });
    const fundo = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 20, maxNativeZoom: 18, attribution: 'Imagem: Esri, Maxar, Earthstar Geographics' });
    fundo.addTo(mapa);
    st = { mapa, fundo, comFundo: true, modo: 'mover', camadas: L.layerGroup().addTo(mapa), marcadores: [],
           novos: [], versao: null, assinatura: '', ed: null, arrasto: null, data: null };
    ESTADOS.set(raiz, st);

    raiz.querySelectorAll('[data-modo]').forEach((b) => {
      b.addEventListener('click', () => st.definirModo(b.dataset.modo));
    });
    raiz.querySelector('[data-acao="fundo"]').addEventListener('click', () => {
      st.comFundo = !st.comFundo;
      if (st.comFundo) st.fundo.addTo(st.mapa); else st.mapa.removeLayer(st.fundo);
    });
    raiz.querySelector('[data-acao="desfazer"]').addEventListener('click', () => st.desenhar(true));
    btAplicar.addEventListener('click', () => {
      const e = st.ed;
      if (!resumo(e).length) return;
      btAplicar.disabled = true;
      rodape.textContent = 'Aplicando…';
      st.enviar('edicao', { versao: st.versao, mover: e.mover, remover: Array.from(e.remover),
                            adicionar: e.adicionar.filter(Boolean), malha: e.malha, r: Date.now() });
    });

    // arrastar a malha inteira
    const ponto = (ev) => st.mapa.mouseEventToLatLng(ev);
    elMapa.addEventListener('pointerdown', (ev) => {
      if (st.modo !== 'malha' || ev.button > 0) return;
      st.arrasto = { ini: ponto(ev), base: st.ed.malha.slice() };
      elMapa.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    });
    elMapa.addEventListener('pointermove', (ev) => {
      if (!st.arrasto) return;
      const a = ponto(ev), i = st.arrasto.ini;
      const cosLat = Math.cos(i.lat * Math.PI / 180);
      st.ed.malha = [st.arrasto.base[0] + (a.lng - i.lng) * 111320 * cosLat,
                     st.arrasto.base[1] + (a.lat - i.lat) * M_LAT];
      st.posicionar();
      st.atualizarRodape();
    });
    const soltar = () => { st.arrasto = null; };
    elMapa.addEventListener('pointerup', soltar);
    elMapa.addEventListener('pointercancel', soltar);

    st.mapa.on('click', (ev) => {
      if (st.modo !== 'adicionar') return;
      st.ed.adicionar.push([ev.latlng.lng, ev.latlng.lat]);
      const k = st.ed.adicionar.length - 1;
      const m = L.marker(ev.latlng, { draggable: true, icon: st.icone('+', 'am-novo') }).addTo(st.camadas);
      m.on('dragend', () => { const p = m.getLatLng(); st.ed.adicionar[k] = [p.lng, p.lat]; });
      m.on('click', () => {
        if (st.modo !== 'remover') return;
        st.ed.adicionar[k] = null; st.camadas.removeLayer(m); st.atualizarRodape();
      });
      st.novos.push(m);
      st.atualizarRodape();
    });

    st.icone = (txt, extra) => L.divIcon({
      className: '', iconSize: st.denso ? [16, 16] : [22, 22],
      html: '<div class="am-pt ' + (st.denso ? 'am-denso ' : '') + (extra || '') + '">' + txt + '</div>' });

    st.definirModo = (modo) => {
      if (modo === 'malha' && !st.data.malha) modo = 'mover';
      st.modo = modo;
      raiz.querySelectorAll('[data-modo]').forEach((b) => b.classList.toggle('am-ativo', b.dataset.modo === modo));
      elMapa.classList.toggle('am-modo-malha', modo === 'malha');
      elMapa.classList.toggle('am-modo-adicionar', modo === 'adicionar');
      if (modo === 'malha') st.mapa.dragging.disable(); else st.mapa.dragging.enable();
      st.marcadores.forEach((m) => { if (modo === 'mover') m.dragging.enable(); else m.dragging.disable(); });
      st.atualizarRodape();
    };

    st.posicionar = () => {            // aplica o deslocamento pendente da malha a todos os pontos (prévia)
      const [de, dn] = st.ed.malha;
      st.marcadores.forEach((m) => {
        const p = m._amOrig;
        const mv = st.ed.mover[m._amI];
        const lat = mv ? mv[1] : p.lat, lon = mv ? mv[0] : p.lon;
        m.setLatLng([lat + dn / M_LAT, lon + de / (111320 * Math.cos(lat * Math.PI / 180))]);
      });
    };

    st.atualizarRodape = () => {
      const r = resumo(st.ed).filter(Boolean);
      const nAdd = st.ed.adicionar.filter(Boolean).length;
      st.ed._limpo = !(r.length && (nAdd || st.ed.remover.size || Object.keys(st.ed.mover).length ||
                                    st.ed.malha[0] || st.ed.malha[1]));
      btAplicar.disabled = st.ed._limpo;
      const dicas = { mover: 'Arraste um ponto para reposicionar.',
                      malha: 'Arraste em qualquer lugar do mapa: todos os pontos andam juntos.',
                      adicionar: 'Clique no mapa para incluir um ponto.',
                      remover: 'Clique num ponto para removê-lo (clique de novo para manter).' };
      if (st.ed._limpo) { rodape.textContent = dicas[st.modo] + ' Nada muda nos arquivos até você clicar em “Aplicar”.'; return; }
      rodape.innerHTML = '<b>Pendente:</b> ' + r.join(' · ') + ' — clique em <b>✔ Aplicar</b>.' +
        ((st.ed.malha[0] || st.ed.malha[1]) && st.data.malha ? ' A malha é refeita na nova posição (pontos podem entrar ou sair pelas bordas).' : '');
    };

    st.desenhar = (manterVista) => {
      const d = st.data;
      st.camadas.clearLayers();
      st.marcadores = []; st.novos = [];
      st.ed = { mover: {}, remover: new Set(), adicionar: [], malha: [0, 0], _limpo: true };
      st.denso = d.pontos.length > 220;
      const limites = [];
      (d.zonas || []).forEach((z) => z.poligonos.forEach((pol) => {
        L.polygon(pol, { color: '#ffffff', weight: 1, fillColor: z.cor, fillOpacity: 0.45, interactive: false }).addTo(st.camadas);
      }));
      (d.celulas || []).forEach((pol) => {
        L.polygon(pol, { color: '#ffffff', weight: 1, opacity: 0.75, fill: false, interactive: false }).addTo(st.camadas);
      });
      d.talhoes.forEach((t) => t.poligonos.forEach((pol) => {
        const g = L.polygon(pol, { color: '#FFE23D', weight: 3, fill: false, interactive: false }).addTo(st.camadas);
        limites.push(g.getBounds());
        if (d.talhoes.length > 1) {
          L.marker(g.getBounds().getCenter(), { interactive: false, opacity: 0.9,
            icon: L.divIcon({ className: 'am-rotulo', html: t.nome, iconSize: [70, 18] }) }).addTo(st.camadas);
        }
      }));
      d.pontos.forEach((p) => {
        const m = L.marker([p.lat, p.lon], { draggable: st.modo === 'mover', icon: st.icone(p.n, ''), title:
          p.t + ' · ponto ' + p.n + (p.z ? ' · ' + p.z : '') }).addTo(st.camadas);
        m._amI = p.i; m._amOrig = p; m._amN = p.n;
        m.on('dragend', () => {
          const q = m.getLatLng();
          st.ed.mover[p.i] = [q.lng, q.lat];
          m.setIcon(st.icone(p.n, 'am-movido'));
          st.atualizarRodape();
        });
        m.on('click', () => {
          if (st.modo !== 'remover') return;
          if (st.ed.remover.has(p.i)) { st.ed.remover.delete(p.i); m.setIcon(st.icone(p.n, st.ed.mover[p.i] ? 'am-movido' : '')); }
          else { st.ed.remover.add(p.i); m.setIcon(st.icone(p.n, 'am-fora')); }
          st.atualizarRodape();
        });
        st.marcadores.push(m);
      });
      if (!manterVista && limites.length) {
        let b = limites[0];
        limites.slice(1).forEach((x) => { b = b.extend(x); });
        st.mapa.fitBounds(b.pad(0.06));
      }
      raiz.querySelector('[data-modo="malha"]').hidden = !d.malha;
      st.definirModo(st.modo);
    };
    setTimeout(() => st.mapa.invalidateSize(), 150);
  }

  st.enviar = setTriggerValue;
  const assinatura = JSON.stringify(data.talhoes.map((t) => t.nome + t.poligonos.length));
  if (st.versao !== data.versao) {
    st.data = data;
    st.versao = data.versao;
    const mesmaArea = st.assinatura === assinatura;
    st.assinatura = assinatura;
    st.desenhar(mesmaArea);
    setTimeout(() => st.mapa.invalidateSize(), 50);
  }
}
"""


def componente():
    """Registra o componente (uma vez por execução do script) e devolve a função de montagem."""
    import streamlit as st
    js = (ASSETS / "leaflet.js").read_text(encoding="utf-8").split("//# sourceMappingURL")[0]
    css = (ASSETS / "leaflet.css").read_text(encoding="utf-8")
    return st.components.v2.component("editor_amostragem", html=HTML, css=css + CSS, js=js + "\n" + JS,
                                      isolate_styles=False)
