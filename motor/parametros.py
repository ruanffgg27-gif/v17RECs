"""Parâmetros do motor de recomendação.

Todos os números que definem as recomendações estão aqui, num só lugar.
Alterar um valor aqui muda o padrão da plataforma para todos os usuários;
na interface, parte deles pode ser ajustada por sessão (garantias do calcário,
limites etc.) sem mexer no código.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Calcario:
    # --- Modelo base (ajustado sobre 2197 recomendações históricas) ---
    # Dose = máx{ a·(pCa·CTC* − Ca) + c·Al ; b·(pMg·CTC* − Mg) + c·Al }
    # CTC* = Ca + Mg + H+Al + k_ctc     (tudo em mmolc/dm³)
    a_ca: float = 85.0          # kg/ha de calcário por mmolc/dm³ de Ca a elevar
    alvo_ca: float = 0.55       # saturação-alvo de Ca na CTC (fração)
    b_mg: float = 345.0         # kg/ha de calcário por mmolc/dm³ de Mg a elevar
    alvo_mg: float = 0.194      # saturação-alvo de Mg na CTC (fração)
    k_ctc: float = 5.6          # constante somada à CTC (≈ K médio no ajuste)
    usar_k_medido: bool = False  # True: usa o K do laudo no lugar de k_ctc
    c_al: float = 165.0         # kg/ha por mmolc/dm³ de Al³⁺

    # --- Calcário de referência do modelo ---
    ref_cao: float = 33.0
    ref_mgo: float = 14.9
    ref_prnt: float = 82.0

    # --- Calcário que será usado (editável na interface) ---
    cao: float = 33.0
    mgo: float = 14.9
    prnt: float = 82.0

    # --- Regras operacionais ---
    dose_min: float = 400.0
    dose_max: float = 3800.0
    fator_abertura: float = 1.8
    # False: limita 400–3800 e DEPOIS multiplica por 1,8 (abertura: 720–6840)
    # True: multiplica por 1,8 e depois limita (abertura nunca passa de 3800)
    abertura_antes_dos_limites: bool = False


@dataclass
class Gesso:
    # Dose = ((S_alvo − S_medido)·1000/75)·Argila(g/kg)/100
    # Faixas de argila (g/kg) → S almejado (mg/dm³)
    limite_arenoso: float = 200.0   # argila < 200  → s_alvo_arenoso
    limite_medio: float = 400.0     # 200 ≤ argila ≤ 400 → s_alvo_medio; > 400 → s_alvo_argiloso
    s_alvo_arenoso: float = 35.0
    s_alvo_medio: float = 25.0
    s_alvo_argiloso: float = 20.0
    dose_min: float = 300.0         # mínimo operacional
    dose_max: float = 1800.0        # acima disso a dose é limitada e sinalizada


@dataclass
class Fosforo:
    # P2O5 (kg/ha) = coef · P_resina^expoente
    coef: float = 141.84
    expoente: float = -0.218
    dose_min: float = 55.0
    dose_max: float = 105.0


@dataclass
class Potassio:
    # KCl (kg/ha, 60% K2O) = coef · K^expoente   (K em mmolc/dm³)
    coef: float = 164.66
    expoente: float = -0.269
    dose_min: float = 100.0
    dose_max: float | None = None   # sem teto definido
    k2o_kcl: float = 60.0           # % de K2O no KCl (00-00-60): K2O da formulação de P é descontado do KCl


@dataclass
class EnxofreElementar:
    # Substituição do gesso: S elementar (kg/ha) = coef · Gesso^expoente
    coef: float = 4.1904
    expoente: float = 0.3754


@dataclass
class Arredondamento:
    calcario: int = 10   # arredonda para múltiplos de 10 kg/ha
    gesso: int = 10
    s_elementar: int = 1
    p2o5: int = 1
    produto_p: int = 5
    kcl: int = 5


@dataclass
class AjusteProduto:
    """Ajuste da coluna inteira de um produto, mantendo as proporções entre amostras.

    - pct: aumenta/diminui todas as doses em X% (ex.: 10 → +10%).
    - media_alvo: se preenchido, escala a coluna para que a média fique igual a
      esse valor (ex.: média do que o cliente já comprou). Tem prioridade sobre pct.
    """
    pct: float = 0.0
    media_alvo: float | None = None

    def ativo(self) -> bool:
        return self.media_alvo is not None or abs(self.pct) > 1e-9


@dataclass
class Ajustes:
    calcario: AjusteProduto = field(default_factory=AjusteProduto)
    gesso: AjusteProduto = field(default_factory=AjusteProduto)   # vale para S elementar se ativado
    p2o5: AjusteProduto = field(default_factory=AjusteProduto)
    kcl: AjusteProduto = field(default_factory=AjusteProduto)
    gesso_por_s_elementar: bool = False
    formula_p: str = ""          # ex.: "11-52-00" → coluna extra com a dose do produto
    kcl_parcelado: bool = False  # divide o KCl em 2 aplicações
    kcl_pct_1: float = 50.0      # % da dose total na 1ª aplicação (a 2ª recebe o restante)
    bloquear_p: bool = False     # P por Mehlich-1 sem confirmação: a regra (P resina) não é aplicada
    # Correção de P (opcional; o padrão é só a manutenção): dose MÉDIA extra informada pelo usuário
    p_correcao: float = 0.0            # kg/ha (de P2O5, ou do produto se p_correcao_em_produto)
    p_correcao_formula: str = ""       # formulação usada na correção, ex.: "00-18-00"
    p_correcao_em_produto: bool = False
    p_correcao_variavel: bool = True   # True: mais onde o P do solo é menor (média = informada); False: uniforme

    def correcao_p2o5(self) -> float:
        """Dose média de correção em kg/ha de P2O5 (0 = sem correção)."""
        from .regras import ler_formula
        d = float(self.p_correcao or 0)
        if d <= 0:
            return 0.0
        if self.p_correcao_em_produto:
            f = ler_formula(self.p_correcao_formula)
            return d * f[1] / 100 if f and f[1] > 0 else 0.0
        return d


@dataclass
class Parametros:
    calcario: Calcario = field(default_factory=Calcario)
    gesso: Gesso = field(default_factory=Gesso)
    fosforo: Fosforo = field(default_factory=Fosforo)
    potassio: Potassio = field(default_factory=Potassio)
    enxofre: EnxofreElementar = field(default_factory=EnxofreElementar)
    arredondamento: Arredondamento = field(default_factory=Arredondamento)

    def como_tabela(self) -> list[tuple[str, str, object]]:
        """Lista (grupo, parâmetro, valor) para registrar no Excel de saída."""
        linhas = []
        for grupo, obj in asdict(self).items():
            for nome, valor in obj.items():
                linhas.append((grupo, nome, valor))
        return linhas
