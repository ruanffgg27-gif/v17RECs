# Recomendação de calcário, gesso, P2O5 e KCl a partir de laudos de solo

Plataforma simples: o usuário **anexa o laudo em Excel** e baixa um **Excel com as recomendações** por amostra.

```
laudo.xlsx ──► leitor (acha colunas e unidades) ──► motor determinístico (regras) ──► Excel de saída
                                    ▲                          ▲
                              interface Streamlit ── ajustes do calcário (CaO, MgO, PRNT) e limites
```

## Estrutura

| Arquivo | O que faz |
|---|---|
| `app.py` | Interface Streamlit (upload, marcar talhões de abertura, baixar Excel) |
| `motor/parametros.py` | **Todos os coeficientes e limites** num só lugar. Editar aqui muda o padrão |
| `motor/regras.py` | As quatro regras, em funções puras e testadas |
| `motor/leitor.py` | Lê o laudo: acha a linha de títulos, reconhece colunas por nome, lê a linha de unidades e converte |
| `motor/processar.py` | Aplica as regras a todas as amostras |
| `motor/saida.py` | Monta o Excel de recomendações |
| `motor/interpretacao.py` | Classes de teores (8 classes), médias por talhão e gráficos |
| `motor/saida_interpretacao.py` | Monta o Excel de interpretação (extra) |
| `assets/` | Logos, ícones e banner; `assets/book/` fotos das capas e seta de norte; `assets/fontes/` Montserrat (licença OFL) |
| `motor/geo.py` | Lê KML/KMZ, identifica o talhão pelo nome do arquivo e liga os pontos às amostras do laudo |
| `motor/krigagem.py` | Krigagem ordinária com semivariograma automático e tratamento de pontos atípicos |
| `motor/prescricao.py` | Superfícies por atributo, doses por pixel, zonas de manejo e shapefiles |
| `motor/externos.py` | Satélite, altitude (SRTM) e clima (NASA POWER) — consultados na hora, com internet |
| `motor/ndvi.py` | NDVI Sentinel-2 dos últimos 12 meses (pico de vigor e série por talhão) |
| `motor/juntar.py` | Reconhece e junta laudos (arquivos) da mesma fazenda |
| `motor/textura.py` | Reconhece dados ausentes no laudo e estima a argila pela CTC |
| `motor/sat/` | Módulo Satélites: fontes (`fontes.py`), análises (`historico`, `lavoura`, `relevo`, `clima`, `vento`, `solo`, `termico`), execução (`diagnostico.py`), relatório (`relatorio.py`) e planilha (`exportar.py`) |
| `motor/ilustracoes.py` | Ilustrações autorais das capas (geral, fertilidade e prescrição), desenhadas em código |
| `motor/book.py` | Monta o book em PDF (A4) para Atria, Protecplan ou Ativa |
| `motor/comparacao.py` e `motor/book_comparacao.py` | Book comparativo entre dois laudos (ano anterior × recente) |
| `motor/sementes.py` e `motor/book_sementes.py` | Mapa de semeadura em taxa variável (índice de fertilidade → sementes/m) |
| `tests/` | Testes (`pytest`): regras conferidas à mão, mapas/shapefiles e os casos da revisão técnica (v13) |
| `exemplos/laudo_exemplo.xlsx` | Laudo fictício para testar |

## Regras implementadas

Unidades padrão: Ca, Mg, K, H+Al e Al em mmolc/dm³ · P e S em mg/dm³ · argila em g/kg.

**Calcário**
- CTC\* = Ca + Mg + H+Al + 5,6
- Dose = máx{ 85·(0,55·CTC\* − Ca) + 165·Al ; 345·(0,194·CTC\* − Mg) + 165·Al }
- Limites de 400 a 3.800 kg/ha. Em área de abertura, a dose já limitada é multiplicada por 1,8 (faixa de 720 a 6.840). A ordem pode ser invertida na interface.
- Correção quando o calcário não é o de referência (33% CaO, 14,9% MgO, PRNT 82):
  - critério Ca × (33 / CaO) × (82 / PRNT)
  - critério Mg × (14,9 / MgO) × (82 / PRNT)
  - termo do Al × (82 / PRNT)

**Gesso**
- Dose = ((S alvo − S) × 1000/75) × argila / 100
- S alvo: 35 mg/dm³ com argila < 200 g/kg; 25 com argila de 200 a 400; 20 com argila > 400
- Mínimo de 300 kg/ha (aplicado também quando o S já está acima do alvo, com aviso). Máximo de 1.800 kg/ha, com aviso.

**P2O5**
- 141,84 × P^−0,218, limitado a 55–105 kg/ha

**KCl (60% K2O)**
- 164,66 × K^−0,269 (K em mmolc/dm³), com mínimo de 100 kg/ha
- Se a formulação de P tiver K2O (ex.: 00-30-10), o K2O que ela fornece é descontado do KCl:
  KCl = KCl pela regra − (dose do produto × %K2O ÷ 100) ÷ 0,60 (nunca abaixo de zero). Vale para a planilha,
  para os mapas de prescrição, para o parcelamento e para os volumes. A planilha mostra "KCl pela regra",
  "K2O via produto P" e "KCl equivalente descontado".

**Camada 20-40 cm**: amostras cuja profundidade começa em 20 cm ou mais aparecem em vermelho na tela, mas não entram no cálculo nem na planilha de saída. Elas aparecem só na interpretação, como camada informativa.

**Ajustes de dose** (na tela, por laudo), aplicados à coluna inteira mantendo as proporções entre amostras:
- **Ajuste (%)**: aumenta ou reduz todas as doses.
- **Cliente já comprou**: informe a média (kg/ha) que o volume comprado permite; a coluna é escalada para chegar a essa média. Tem prioridade sobre o ajuste percentual.
- **Gesso → S elementar**: S elementar = 4,1904 × Gesso^0,3754.
- **Parcelamento do KCl**: divide a dose final de KCl em 2 aplicações (ex.: 60% + 40%). A 1ª parcela é arredondada e a 2ª recebe o restante, então as duas somam exatamente a dose total.
- **Formulação de P** (ex.: 11-52-00): cria a coluna do produto = P2O5 ÷ (%P2O5/100), além do N e K2O que o produto fornece (na memória de cálculo).

Arredondamento: calcário e gesso para múltiplos de 10 kg/ha, KCl de 5, P2O5 de 1 (pode ser alterado em `parametros.py`).

## Abas do app

- **📍 Amostragem** — pontos de amostragem em grade ou por zonas de manejo, a partir do perímetro.

- **🧪 Recomendação e book** — o fluxo completo (laudos → recomendações, interpretação, mapas, shapefiles e book).
- **🛰️ Satélites** — diagnóstico do talhão só com o perímetro (KML/KMZ), com relatório em PDF (ver abaixo).
- **📈 Comparações** — mesmo fluxo da recomendação com o laudo mais recente; o book fica comparativo (ver abaixo).
- **🌱 Mapa de Sementes** — semeadura em taxa variável a partir de P, M.O., argila e soma de bases (ver abaixo).

Granulometria: argila, areia e silte são lidas em g/kg (usadas assim nas contas do gesso) e **exibidas em %**
(÷ 10) nos mapas do book e na interpretação.

## Módulo Satélites (diagnóstico do talhão)

Entrada: apenas o perímetro do talhão (KML/KMZ). Saídas: relatório em PDF no estilo dos books, planilha com as
tabelas e KML com os pontos prioritários de visita. Cada frente é independente (se uma fonte falhar, as outras
seguem e o relatório avisa) e todas as fontes são públicas e gratuitas:

| Frente | Fonte | O que entrega |
|---|---|---|
| Histórico | MapBiomas Coleção 11 (1985–2025; reserva Coleção 10) e PRODES/INPE | "Biografia do talhão": uso ano a ano, ano provável de abertura, fases de pastagem/lavoura, água superficial, desmatamento anual |
| Lavoura | Sentinel-2 L2A (Earth Search/AWS), 1 imagem/mês nos últimos N anos | Série de NDVI; NDVI, NDRE, NDMI e MSAVI2 no pico de cada ano agrícola; estabilidade do vigor; anomalia recente; uniformidade (CV e % de área); baixa cobertura recorrente; pontos prioritários com coordenadas |
| Relevo e água | Copernicus DEM GLO-30 (AWS; reserva SRTM) | Altitude e curvas de nível; declividade em classes Embrapa (ha); posição na paisagem (TPI); escoamento D8 com o entorno; saídas de água do talhão; perfis; suscetibilidade relativa à erosão (LS × cobertura) |
| Clima | CHIRPS 2.0 via ClimateSERV (pedidos em blocos de 10 anos; reserva NASA POWER) e NASA POWER | Chuva mensal e anual desde 1981; safra atual × histórico; frequência de veranicos (≥ 10 e ≥ 15 dias) por mês; início e fim típicos das chuvas; dias de calor (≥ 35 °C) e frio (≤ 5 °C, ≤ 2 °C) |
| Vento | NASA POWER horário (últimos 5 anos, hora local; reserva: diário) | Rosa dos ventos por classe de velocidade; meses mais ventosos; calmaria; janelas de pulverização por mês e hora (vento 3–10 km/h, < 30 °C, UR > 55%) — regional, igual para todo o talhão |
| Solo | SoilGrids 2.0 (ISRIC), 250 m | Só granulometria (argila, silte, areia 0–60 cm) e grupo textural na camada 0–30 cm, com faixa provável de 90% — só contexto (C, CTC, pH, N e densidade foram retirados: ficavam longe da realidade das lavouras) |
| Temperatura | Landsat 8/9 Coleção 2 (Planetary Computer) | Mapa da temperatura de superfície relativa (média das passagens, suavizada ~45 m e interpolada em 10 m, com isotermas), amplitude e área ≥ 1 °C mais quente |

Limitações importantes (também escritas no relatório): baixo vigor não identifica a causa; o DEM é de superfície
(30 m) e não substitui topografia; clima e solo vêm de grades regionais (não são medições no talhão) e o SoilGrids
não serve para recomendar corretivos ou adubos; temperatura alta sozinha não comprova falta de água; a
evapotranspiração não é estimada nesta versão. O tempo de processamento depende das fontes (tipicamente 2–5 min).

## Novidades da v17

- **Correção de P (opcional).** Em "Ajustar doses" há uma caixa "Correção de P": o usuário informa a dose média
  de correção (em P₂O₅ ou em produto) e, se quiser, a formulação. A manutenção não muda; a correção sai em coluna
  própria ("P correção" e, com formulação, "Correção NN-PP-KK") e em prescrição própria no book e nos shapefiles
  (sigla `PCOR`). Distribuição **variável** (padrão: mais onde o P do solo é menor, pela mesma curva da manutenção
  sem os limites, com a média igual à informada) ou **uniforme**. O K₂O da formulação de correção não é
  descontado do KCl.
- **Escolha das prescrições do book.** No formulário do book, "Prescrições que entram no book e nos shapefiles"
  (calagem, gessagem, fósforo, correção de P, potássio). O que for retirado some das páginas, da tabela de volumes
  e dos shapefiles; os mapas de fertilidade continuam completos.
- **Mapa de sementes.** Até 15 doses; **degrau entre doses** configurável (automático: 0,1 / 0,05 / 0,02 conforme
  a média — antes era fixo em 0,1, o que limitava culturas de poucas sementes/m a 3 ou 4 doses); **amplitude**
  "toda a faixa" (o lugar mais fértil recebe média − variação e o menos fértil média + variação); e **ajuste
  manual**: tabela com a dose recomendada e a ajustada por faixa — PDF e shapefiles saem com o ajuste, e o PDF
  registra o que foi alterado.
- **Estética.** Títulos e subtítulos das prescrições se ajustam à moldura (reduzem a fonte e, no limite, encurtam
  pelo meio); barra de escala na cor da moldura; "Pontos de atenção" em texto corrido separado por ";", sempre
  acima da moldura; nomes longos de talhão abreviados nas tabelas estreitas; legenda do MSAVI2 invertida.
- **Satélites — página "Água no solo"** (`motor/sat/agua.py` e `motor/sat/nisar.py`): ver abaixo.

### Água no solo

Só o perímetro é obrigatório. O relatório separa **dados públicos** de **informações do usuário**.

- **Balanço hídrico (90 dias):** chuva CHIRPS (dias ainda não publicados: NASA POWER) e ET₀ Penman-Monteith FAO-56
  com dados da NASA POWER (Hargreaves-Samani nos dias sem radiação, umidade ou vento). Dia sem dado não vira zero.
  Sem parâmetros do usuário saem chuva, ET₀ e o saldo P − ET₀, apresentado como indicador climático — não como
  água disponível.
- **Armazenamento na zona radicular:** só com cultura, data de semeadura, profundidade radicular e CAD. Balanço
  diário FAO-56 (Kc único por fase, fração p de esgotamento, Ks de estresse), com 90 dias de aquecimento e três
  condições iniciais (a faixa mostra o quanto o resultado depende do chute inicial). Os coeficientes por cultura
  estão em `CULTURAS` (`motor/sat/agua.py`) — são valores genéricos da FAO-56, **confira antes de usar em
  recomendação de irrigação**. Irrigação não informada sai como "cenário sem irrigação", identificado como tal.
- **Umidade superficial NISAR:** produto oficial L3 SME2 (200 m, EASE-Grid 2.0), coleção provisória
  `NISAR_L3_SME2_PROVISIONAL_V1` da ASF DAAC/NASA, com a máscara de qualidade do próprio produto. Mostra a
  observação válida mais recente (m³/m³), data, passagem, resolução, cobertura válida, pixels de borda e a evolução
  nas passagens dos últimos 90 dias. Não é extrapolada para o perfil, não corrige o balanço e não é substituída por
  NDMI. Sem observação: "Sem observação válida disponível".
  - **Credencial:** a busca no catálogo é aberta, mas ler os arquivos exige login Earthdata (gratuito, em
    https://urs.earthdata.nasa.gov). Coloque nos Secrets `earthdata_token` (perfil → *Generate Token*) ou
    `earthdata_user` e `earthdata_password`; no primeiro uso, aceite o aplicativo de dados da ASF na sua conta.
  - **Situação em 06/10/2026 (fontes oficiais):** produto provisório, ainda não totalmente calibrado; versão
    validada prevista para o 4º trimestre de 2026 — quando sair, troque `COLECAO` e `NOME_CURTO` em
    `motor/sat/nisar.py`. O valor de preenchimento, os bits de `surfaceQualityFlag` e os nomes dos vetores de
    coordenadas não puderam ser conferidos na especificação: o código lê o que estiver no arquivo.
  - **Não testado contra o servidor real** (o ambiente de desenvolvimento não tem acesso nem credencial): a leitura
    foi validada com arquivos de teste no layout documentado. O primeiro uso real deve ser conferido.

## Novidades da v16

- **Modo escuro de verdade.** O app agora tem dois temas (`.streamlit/config.toml`: `[theme.light]` e
  `[theme.dark]`) e o estilo do `app.py` usa uma paleta para cada um. No escuro: fundo quase preto, texto claro,
  contornos em verde fluorescente, sombras e botão principal em rosa-choque, destaques em amarelo; a marca ganha
  uma etiqueta clara para continuar legível. O tema segue o sistema do usuário e pode ser trocado a qualquer
  momento no menu ⋮ do app (System / Light / Dark). A troca é feita no navegador (CSS `light-dark()`), então
  o modo claro continua disponível mesmo com o computador em modo escuro (corrigido na v16.1).
- **Amostragem com prévia e edição no mapa** (`motor/editor_mapa.py`, Leaflet embutido em `assets/leaflet`):
  depois de gerar, os pontos aparecem sobre a imagem de satélite e dá para **arrastar um ponto**, **arrastar a
  malha inteira**, **incluir** e **remover** pontos. As alterações ficam pendentes até "✔ Aplicar"; ponto solto
  fora do talhão é recusado. PDF, KML, GPX, shapefile e Excel saem sempre com o estado atual.
- **Posição da malha com valores exatos:** deslocar para leste/norte (m) e girar (graus). A malha é refeita na
  nova posição (células recortadas de novo pelo perímetro).

## Novidades da v15

- **Mapas de fertilidade fiéis às amostras (novo padrão).** Comparando books feitos no SMS com os da v14.1
  (Sítio Irma Dina, 13 pontos; Fazenda Brasão, 185 pontos), a diferença vinha do semivariograma ajustado na área
  inteira: quando a malha não mostra estrutura espacial nessa escala, a krigagem "estatística" puxa o mapa para
  a média (P, K, %K, argila do TH 01 da Brasão ficavam quase lisos) e ainda limitava picos isolados. O modo
  **fiel** usa krigagem ordinária local — esférico, alcance = 4 × distância média entre pontos vizinhos, pepita
  de 5% filtrada — e reproduz os mapas do SMS: a diferença de área por classe em relação ao SMS caiu de 16% para
  8% (Irma Dina) e de 12,5% para 5,7% (Brasão), na média dos atributos. Contrapartida, medida por validação
  cruzada: o erro ao prever um ponto retirado é 6–16% maior que no modo estatístico — manchas em torno de um
  único ponto mostram o que foi medido ali, não uma tendência comprovada.
- **Prescrições no mesmo padrão dos mapas (v15.1).** Por padrão, as doses são calculadas sobre as mesmas
  superfícies "fiéis" dos mapas de fertilidade, e o mapa de semeadura também. Desmarcando "Prescrições com a
  mesma interpolação dos mapas", as doses voltam à interpolação estatística da v14.1 (zonas mais largas). As
  regras de zona (largura mínima de 30 m, área mínima de 0,5 ha, fechamento do volume) não mudaram.
- **Faixas sólidas em contornos vetoriais:** as divisas entre classes são curvas lisas em qualquer ampliação
  (sem "escadinha" de pixel nem pontinhos em cima das amostras).
- **H+Al, Al e m%:** quanto menor, melhor — a faixa mais baixa passa a ser "Excelente" (nunca "Muito alto") na
  condição média, no panorama, na área por classe e no Excel de interpretação. As cores dos mapas não mudam.
- **Nova aba "📍 Amostragem":** pontos de amostragem a partir do perímetro — em grade (quadrada ou desencontrada,
  ponto no centro ou sorteado na célula, malha alinhada ao talhão) ou por zonas de manejo (vigor relativo
  histórico, NDVI Sentinel-2). Saídas: mapa em PDF, KML por talhão (`Pontos_TH_1.kml`, nome do ponto = número),
  GPX, shapefile e Excel. Detalhes em `motor/amostragem.py`.
- **Barra lateral** em verde claro liso (sem listras).

## Novidades da v14.1

- **Uma paleta para cada índice de vegetação:** NDVI em vermelho→verde, NDRE em rosa→roxo, NDMI em marrom→azul e
  MSAVI2 em roxo→amarelo, para deixar claro que cada mapa mede uma coisa diferente (a legenda diz como ler cada um).
- **Nada passa da moldura no diagnóstico por satélite:** os gráficos ficam dentro de limites fixos e todo texto
  é conferido antes de a página ser salva (reduz a fonte e, em último caso, encurta com reticências).
- **Mapa de semeadura em PDF simples:** capa própria (`assets/book/capa_sementes.jpg`), um mapa por talhão e o
  resumo. A página do índice de fertilidade é opcional (caixa "Incluir a página do índice de fertilidade").

## Novidades da v14

- **Visual retrô (anos 80/90):** fundo de papel, barra lateral em "formulário contínuo", abas como divisórias de
  fichário e tudo o que é clicável (botões, caixas de envio de arquivo, campos) com contorno de tinta e sombra
  dura — os botões "afundam" ao clicar. O tema fica em `.streamlit/config.toml` e no bloco de estilo do `app.py`.
- **Informações técnicas do book:** listas longas de talhões quebram em linhas e os nomes longos da tabela são
  encurtados pelo meio (mantendo o número), sem passar da moldura.
- **Prescrições agrupadas por produto:** calagem de todos os talhões, depois gessagem (ou S elementar), fósforo e
  potássio (parcelas na sequência); o sumário aponta para cada produto.
- **Satélites — página "Índices de vegetação":** NDVI, NDRE, NDMI e MSAVI2 lado a lado, como **média histórica**
  do pico de cada ano agrícola (padrão: 5 anos). A legenda de cada mapa vai do percentil 2 ao 98 do próprio
  talhão, com os valores reais escritos (mínimo, média, máximo) — as diferenças internas aparecem mesmo quando o
  NDVI está saturado. A página de NDVI do book de fertilidade usa a mesma lógica (classes ajustadas à área).

## Confiabilidade das prescrições (v13, após revisão técnica)

- **Profundidade validada:** início, fim e unidade (cm ou m: "0,20–0,40 m" = 20–40 cm). 0–20 entra nas contas;
  camadas que começam em ≥ 20 cm são subsuperficiais; outras camadas (0–10, 10–20, 0–40…) ficam **fora**, com
  aviso. Sem coluna de profundidade (ou toda em branco), as amostras são tratadas como 0–20 com aviso.
- **Método analítico:** P Mehlich-1 e pH em água são reconhecidos. Com Mehlich-1, a dose de P não é calculada até
  a confirmação nos ajustes (a regra e as classes são de P resina); o book mostra o método reconhecido ou
  "não informado"; o Excel registra versão, domínio das regras e métodos na aba Parâmetros.
- **Requisitos por produto:** faltou S (ou argila) num talhão → só o gesso daquele talhão fica de fora; os demais
  produtos seguem. Dose zero continua zero ("não aplicar"), distinta de "sem dado".
- **Volume comprado preservado:** com média-alvo, as doses das zonas são recalibradas depois da suavização, da
  classificação e da fusão de manchas (mantendo o arredondamento operacional).
- **Parcelas do KCl:** 1ª e 2ª parcelas usam a mesma partição da dose total (2ª = total − 1ª): fecham polígono a
  polígono.
- **Conferência final:** cada shapefile é relido após a exportação — cobertura do perímetro, sobreposições, doses
  válidas, volume (zonas × shapefile) e fechamento das parcelas. Resumo no app e `CONFERENCIA.txt` no zip.
- **Resultado desatualizado:** book e shapefiles guardam uma assinatura das entradas (laudo, KMLs, parâmetros,
  ajustes, abertura, blocos); se algo muda depois, o app avisa para gerar de novo.
- **Krigagem sem estrutura espacial:** quando a validação cruzada não supera a média, o atributo é marcado
  ("estrutura espacial não comprovada") e você escolhe entre mapa exploratório e valor médio uniforme; a limitação
  de valores extremos isolados pode ser desligada.
- **S elementar × gesso:** o S elementar substitui só o fornecimento de S; se a camada 20–40 indica gesso como
  condicionador (Ca ≤ 5, Al ≥ 5 mmolc/dm³ ou m ≥ 20%), o app avisa.
- **"Muito alto" não é "melhor":** para pH, V%, Ca na CTC e B, passar de Excelente para Muito alto não conta como
  melhora na comparação; os textos do panorama foram ajustados.
- **Semeadura:** identificada como regra da equipe em avaliação (recomenda faixas com taxa fixa para comparar).
- **Engenharia:** versões das bibliotecas com faixas testadas (`requirements.txt`), testes automáticos no GitHub
  (`.github/workflows/testes.yml`, aba Actions) e teste que confirma que planilha e mapa usam as mesmas regras.

## Book comparativo (aba "📈 Comparações")

Envie o laudo **mais recente**; na aba **📈 Montar book comparativo**, os KMLs (perímetros e pontos) e o laudo do
**ano anterior**. Os pontos do ano anterior são opcionais: sem envio, valem os mesmos pontos do ano recente, na mesma
ordem; se a amostragem mudou, envie só os KML de pontos daquele ano (os perímetros são sempre os do ano recente).

- O laudo anterior é interpolado **na mesma grade** do recente (mesmo perímetro e pixel): mapas, áreas por classe e
  médias ficam diretamente comparáveis, pixel a pixel.
- Página de cada atributo: dois mapas com a mesma legenda — **lado a lado** para áreas mais altas que largas, **um
  sobre o outro** para áreas largas (escolhe o arranjo em que o mapa fica maior) —, média e classe de cada ano, a
  variação da média, a tabela de área por classe de cada ano (ha, % e variação em pontos percentuais) e as barras
  "para onde foi a área" (transições de classe pixel a pixel).
- Páginas de síntese no lugar do diagnóstico/panorama: **Evolução da fertilidade** (média, classe e variação de cada
  atributo, verde = melhorou na escala de interpretação), **Evolução por talhão**, **Área por classe** (ano anterior
  sobre ano recente) e **Onde mudou** (mapas de diferença de pH, V%, Ca, Mg, K e P).
- Prescrições: **só as do laudo mais recente**. A tabela final de volumes traz, por talhão e produto, o total do ano
  anterior (mesma regra e parâmetros, sem o ajuste de volume comprado) contra o prescrito agora.
- Downloads: book comparativo (PDF), comparação (Excel: evolução, médias por talhão e volumes) e shapefiles.

## Mapa de semeadura (aba "🌱 Mapa de Sementes")

1. Cada atributo vira % do maior valor das amostras (regra de três): P, M.O., argila e soma de bases (Ca+Mg+K).
2. Índice de fertilidade = (P% × 5 + M.O.% × 20 + argila% × 40 + SB% × 35) ÷ 100. Sem argila medida (inclusive
   quando ela foi estimada pela CTC), a soma de bases recebe peso 75.
3. Taxa = k ÷ índice (proporção inversa: onde é mais fértil caem menos sementes), limitada a ± a variação máxima
   (padrão 20%) em torno da média comprada, com k ajustado para a média do talhão ficar **igual à média comprada**.
4. O mapa é dividido em até 8 faixas aplicáveis (mesmas regras das zonas de prescrição) e as faixas são
   recalibradas para manter a média (passo de 0,1 semente/m).

Entradas: sementes/m compradas (padrão 13), espaçamento entre linhas (para sementes/ha), variação máxima, número de
faixas, média específica por talhão (opcional) e unidade do shapefile (sementes/m ou sementes/ha). Saídas: PDF do
mapa de semeadura (capa, um mapa por talhão e resumo; índice de fertilidade opcional), shapefiles `SEM_T1…` e Excel com a conta amostra a
amostra.

## Shapefiles: nomes curtos e pastas por monitor

Os arquivos saem com nomes de até 9 caracteres, sem espaços (`CAL_T4` = calcário do talhão 4; siglas CAL, GES, SEL,
FOS, KCL, KC1/KC2, SEM, P2O5), um por produto dentro da pasta do talhão, e o zip traz um `LEIA-ME.txt`. O ícone
📁 **Pastas por marca de monitor** (passe o mouse) lembra onde colocar os arquivos no pen drive: John Deere `Rx/`;
Stara `Dados/Mapas/`; Ag Leader e Trimble `AgGPS/Prescriptions/`; Raven `rxMaps/`.

Cores das prescrições: sempre a partir do verde escuro na menor dose (2 doses → verde escuro e verde claro;
3 → + amarelo; 4 → + laranja…), contando só as doses que aparecem no mapa.

## Laudos sem granulometria ou sem micronutrientes

O leitor identifica os atributos que não vieram no laudo (sem coluna ou coluna vazia) e mostra um aviso 🔎.

- **Argila ausente** (usada só no gesso): estimada pela CTC a pH 7 com a equação ajustada em 2.085 amostras da
  base da equipe: **Argila (g/kg) = −186 + 6,57 × CTC (mmolc/dm³)** (R² = 0,78; erro típico ≈ 100 g/kg; a classe
  de argila do gesso é acertada em ~75% dos casos), limitada a 60–750 g/kg. Se a CTC não vier, usa-se
  Ca + Mg + H+Al + K. Se o laudo tiver argila em parte das amostras, a relação é recalibrada com elas
  (regressão local, ou correção do viés da equação regional). Cada valor estimado fica marcado na planilha
  (coluna *Argila - origem* e observação), na interpretação (textura "(est.)") e no book (metodologia; o mapa
  de argila sai do book quando a maioria das amostras é estimada).
- **Micronutrientes, areia ou silte ausentes:** ficam fora da interpretação e do book, sem erro.

## Blocos de aplicação (unir talhões na prescrição)

Na aba do book, em **🧩 Unir talhões na prescrição**, escreva o mesmo nome de bloco (ex.: *Bloco A*, *Pivô 1*)
nos talhões que devem sair juntos — ou marque *Todos os talhões num bloco só*. Os talhões de um bloco
recebem **as mesmas doses de zona** (calculadas com a faixa de doses do bloco inteiro) e saem num **shapefile
único por produto** (um registro por dose). No book, a prescrição do bloco mostra todos os talhões dele numa
página. Mapas de fertilidade e volumes por talhão não mudam.

## Vários laudos da mesma fazenda

Anexe todos os arquivos de uma vez. O app compara produtor e propriedade (ignorando acentos, "Fazenda/Faz.",
maiúsculas etc.) e sugere quais arquivos são da mesma fazenda (quadro 🔗). Arquivos com o mesmo número de grupo
viram **um laudo só**: uma planilha, uma interpretação e um book com todos os talhões. Dá para mudar o grupo
à mão. Na planilha, a coluna *Arquivo* mostra a origem de cada amostra e cada arquivo tem sua aba de laudo
original. Talhões com o mesmo nome em arquivos diferentes ganham a letra do arquivo (`1 (B)`). No book, se os
talhões de um arquivo estiverem a mais de 5 km dos demais, o app avisa.

## O Excel de saída

1. **Recomendações** – uma linha por amostra, com as 4 doses e observações
2. **Resumo por talhão** – média, mínimo e máximo de cada dose
3. **Memória de cálculo** – valores usados, resultado bruto de cada equação e o critério que definiu a dose
4. **Parâmetros** – calcário, coeficientes e limites usados, colunas lidas e avisos (garante rastreabilidade)
5. **Laudo original** – cópia do que foi enviado (somente as linhas de 0-20 cm)

## O Excel de interpretação (extra)

Situação média de cada talhão classificada nas 8 classes da tabela de legendas da equipe (Crítico → Muito alto), com as mesmas cores:
1. **Diagnóstico 0-20 cm** – médias coloridas por classe e pontos de atenção por talhão
2. **Camada 20-40 cm** – o mesmo para a camada subsuperficial, se houver
3. **Gráficos** – mapa de classes e doses médias por talhão
4. **Classes (lista)** – formato longo, para filtrar
5. **Faixas de referência** – os limites de cada classe (editáveis em `motor/interpretacao.py`)

## Book de fertilidade e prescrições (aba "📚 Montar book")

1. Anexe o laudo e ajuste as recomendações normalmente (abertura, ajustes, parcelamento etc.).
2. Na aba **Montar book**, anexe os KML/KMZ de **perímetro** e de **pontos** de cada talhão.
   O talhão é lido do nome do arquivo (`Perimetro_TH_1.kml`, `Pontos_TH_2.kmz`…) e pode ser corrigido na tabela.
3. Ligação laudo ↔ pontos (v13): pelo **número** — número da identificação da amostra ("AMOSTRA 07" → 7) ×
   número do nome do ponto no KML ("P07", "7" → 7); a ordem das linhas no laudo não importa. Numeração corrida
   (amostras 31–60 × pontos 1–30, mesmo deslocamento) é aceita. Sem números nas amostras, liga pela ordem.
   A tabela **🔗 Conferência amostra × ponto** mostra cada par; quantidades diferentes, números sem par ou
   números que não coincidem **travam a geração** até você marcar que conferiu. Amostras de 20-40 cm herdam o
   ponto da amostra de mesmo número.
4. Escolha a marca (Atria, Protecplan ou Ativa), preencha município e data da coleta e clique em **Gerar**.

Saídas: **book em PDF** e **.zip com os shapefiles** de prescrição (uma pasta por talhão; WGS84; campo
`Taxa_Dest_`, como o padrão usado hoje; com arquivo .prj). Depois de gerar, escolha **quais produtos** entram
no .zip.

**Fósforo:** informe no formulário do book a **formulação comprada** (ex.: 11-52-00) — vem preenchida com a
da aba de ajustes. O mapa e o shapefile saem em dose do produto. Sem formulação, o mapa mostra P₂O₅
(nutriente, um "00-100-00" que não existe) e o shapefile de P₂O₅ fica bloqueado (pode ser liberado só para
conversão manual). **KCl parcelado:** também definido no formulário; o book traz um mapa por aplicação.

Como os mapas são feitos:
- Todos os atributos são interpolados numa grade de 10 m por **krigagem ordinária** (padronizado para todos
  os talhões e atributos). O semivariograma é ajustado automaticamente (esférico, exponencial ou gaussiano,
  por validação cruzada); quando a malha amostral não detecta a estrutura espacial (alcance menor que a
  distância entre pontos ou efeito pepita puro), usa-se um semivariograma padrão (esférico, pepita 15%,
  alcance de 2,5× a distância entre pontos). Os detalhes de cada mapa ficam no app, em
  "Detalhes da krigagem" (uso interno).
- **Pontos atípicos:** valores além de 3 intervalos interquartis são limitados; e um ponto muito diferente dos
  6 vizinhos (resíduo > 3 desvios robustos) é trazido para a faixa dos vizinhos. Isso evita os "alvos"
  (círculos concêntricos de dose) que um único ponto cria. O valor do laudo e a dose da planilha não mudam.
- As doses são calculadas **em cada pixel pelas mesmas regras da planilha** (e com os mesmos ajustes);
  depois são agrupadas em até 6 zonas com **largura mínima de 30 m** e **área mínima de 0,5 ha**.
- As cores dos mapas de fertilidade seguem as 8 classes da tabela de legendas da equipe (faixas sólidas, 1 cor por classe;
  há opção de transição suave); a legenda traz a área de cada classe e a média de cada talhão, com unidade.
- Fazendas com muitos talhões: talhões próximos (< 2,5 km) ficam na mesma página; grupos distantes ganham páginas próprias.

Dados externos (precisam de internet no servidor; se falharem, o book sai sem aquela parte e avisa):
- Satélite: Google (Map Tiles API, precisa de `google_maps_key` nos Secrets), Esri World Imagery ou
  Sentinel-2 cloudless (EOX, CC BY 4.0). Se a fonte escolhida falhar, o app tenta a Esri. Tiles baixados em paralelo.
- Altitude: SRTM por tiles de terreno (Terrain Tiles, AWS Open Data) — poucas requisições, valor em cada pixel;
  reserva: OpenTopoData/Open-Meteo por pontos (no máximo ~800 pontos).
- Clima: normais mensais da NASA POWER.
- **NDVI** (opcional, marcado por padrão): Sentinel-2 L2A (Copernicus) via Earth Search, sem chave. Uma data por mês
  nos últimos 12 meses, nuvens removidas pela máscara SCL; página com o **pico de vigor** (NDVI máximo) em
  7 classes fixas, a **evolução do NDVI** e o pico médio/CV de cada talhão. Soma cerca de 20–40 s.

Capas (v10): artes da equipe em `assets/book/capa_geral.jpg`, `capa_fertilidade.jpg` e `capa_prescricao.jpg`
(página inteira; o título fica na área do céu e o produtor/propriedade num cartão na capa geral). Para trocar
uma arte, substitua o arquivo mantendo o nome (formato retrato, proporção próxima de A4, céu claro no topo) e,
se o céu ficar mais alto ou mais baixo, ajuste `CAPAS` em `motor/book.py`. Se o arquivo faltar, o book usa as
ilustrações desenhadas em código (`motor/ilustracoes.py`): ilustrações autorais (paisagem com talhões, pivô, amostragem georreferenciada, drone
e satélite na capa geral; perfil do solo, trado, troca de cátions, agregados e vidraria na capa de fertilidade;
talhão em zonas de dose, trator com distribuidor a lanço de taxa variável e GNSS na capa de prescrição), nas
cores de cada marca.

Desempenho: a leitura do laudo, as figuras da interpretação e os Excel são guardados em cache/gerados só no
clique; a validação cruzada da krigagem usa a fórmula fechada (uma inversão por modelo). Um book de 12 talhões
(~1.800 ha) leva ~25 s de processamento, mais o tempo das consultas externas.

## Rodar no computador

```bash
pip install -r requirements.txt
streamlit run app.py          # abre em http://localhost:8501
pytest                        # roda os testes
```

## Hospedagem gratuita (Streamlit Community Cloud)

1. Crie uma conta no GitHub e um repositório novo; envie esta pasta para ele (pode usar o botão *"Add file → Upload files"* no site).
2. Acesse **share.streamlit.io** e entre com a conta do GitHub.
3. Clique em **Create app**, escolha o repositório, branch `main` e arquivo `app.py`, e depois **Deploy**.
4. Em poucos minutos você recebe um endereço `https://<nome>.streamlit.app` para compartilhar com os colegas.
5. **Senha e música:** no painel do app, abra *Settings → Secrets* e cole (veja `.streamlit/secrets.toml.exemplo`):
   ```
   senha = "sua-senha"
   playlist = "https://www.youtube.com/playlist?list=..."
   ```
   A playlist pode ser do YouTube ou do Spotify. O botão 🎵 na barra lateral liga e desliga o player. No Spotify, só quem estiver logado ouve as faixas completas; no YouTube, todos ouvem.

Observações:
- Os laudos são processados em memória e **não ficam gravados** no servidor. O repositório tem apenas código e um laudo fictício; não suba laudos reais.
- Apps gratuitos "dormem" depois de alguns dias sem uso. O primeiro acesso depois disso leva cerca de 30 s para acordar.
- Alternativas gratuitas, se preferir: Hugging Face Spaces (modelo "Streamlit") ou Render.

## Como alterar regras

- **Garantias do calcário, limites e S alvo:** na barra lateral do app (vale só para aquela sessão).
- **Mudança permanente:** edite `motor/parametros.py`, rode `pytest` e envie ao GitHub. O app atualiza sozinho.
- **Colunas com outros nomes:** acrescente o nome em `ALIASES`, no `motor/leitor.py` (sem acento, minúsculo, sem espaços).
