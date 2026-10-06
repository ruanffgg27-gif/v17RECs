"""Motor determinístico de recomendação de corretivos e fertilizantes."""
VERSAO = "17.0"   # muda a chave do cache do app a cada versão (evita objetos antigos em cache)
from .leitor import ler_laudo, Laudo
from .parametros import Ajustes, AjusteProduto, Parametros
from .processar import recomendar, resumo_por_talhao
from .saida import gerar_excel
from .saida_interpretacao import gerar_excel_interpretacao

__all__ = ["ler_laudo", "Laudo", "Ajustes", "AjusteProduto", "Parametros", "recomendar",
           "resumo_por_talhao", "gerar_excel", "gerar_excel_interpretacao"]
