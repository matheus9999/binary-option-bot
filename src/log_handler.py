
from datetime import datetime, timedelta
import pytz
from dateutil import tz

def format_prediction(direction: str) -> str:
    """Formata a direção da previsão com ícones."""
    if direction.upper() in ['CALL', 'BUY', 'SUBIR']:
        return "subir ⬆️"
    elif direction.upper() in ['PUT', 'SELL', 'CAIR']:
        return "descer ⬇️"
    return direction

def get_pattern_messages(patterns: dict) -> list:
    """Gera mensagens descritivas para os padrões encontrados."""
    messages = []
    
    # Mapeia o nome da feature para a tupla (Nome do Padrão, Mensagem Bullish, Mensagem Bearish)
    pattern_map = {
        "engulfing": ("Engolfo", "Engolfo de Alta", "Engolfo de Baixa"),
        "pin_bar": ("Pin Bar", "Pin Bar de Alta (Martelo)", "Pin Bar de Baixa (Estrela Cadente)"),
        "marubozu": ("Marubozu", "Marubozu de Alta", "Marubozu de Baixa"),
    }

    for key, value in patterns.items():
        if key in pattern_map and value != 0:
            name, bullish_msg, bearish_msg = pattern_map[key]
            if value == 1:
                messages.append(bullish_msg)
            elif value == -1:
                messages.append(bearish_msg)
        elif key == "inside_bar" and value == 1:
            messages.append("Inside Bar")
            
    if not messages:
        messages.append("Nenhum padrão claro identificado, decisão baseada em indicadores.")
        
    return messages

def log_operation(user: str, ativo: str, expiracao: str, hora_entrada: datetime, previsao: str, confianca: float, padroes: dict):
    """
    Imprime um log de operação formatado e unificado.

    Args:
        user: O nome do usuário que esta executando o robô.
        ativo: O par de moedas (ex: "EURUSD-OTC").
        expiracao: O timeframe (ex: "M1", "M5").
        hora_entrada: O horário de início da vela de análise (esperado em UTC).
        previsao: A previsão do modelo ("SUBIR" ou "CAIR").
        confianca: A confiança do modelo na previsão.
        padroes: Um dicionário com os padrões encontrados na vela.
    """
    # Define o fuso horário para o fuso local da máquina
    local_timezone = tz.tzlocal()
    
    # Garante que a hora de entrada está ciente do fuso (assume UTC se for ingênua)
    if hora_entrada.tzinfo is None:
        hora_entrada = pytz.utc.localize(hora_entrada)
    
    # Converte para o fuso horário local da máquina
    hora_entrada_local = hora_entrada.astimezone(local_timezone)

    # Extrai o número do intervalo de tempo (ex: "5m" -> 5)
    try:
        interval_minutes = int(''.join(filter(str.isdigit, expiracao)))
    except (ValueError, TypeError):
        interval_minutes = 1 # Padrão de 1 minuto se não conseguir extrair

    hora_expiracao_local = hora_entrada_local + timedelta(minutes=interval_minutes)

    formatted_previsao = format_prediction(previsao)
    pattern_messages = get_pattern_messages(padroes)
    
    log_message = f"""
==================================
✅ OPERAÇÃO {user} ✅
📈 ATIVO: {ativo}
⏰ EXPIRAÇÃO: {expiracao.replace("m", "M")}
⏱️ ENTRADA: {hora_entrada_local.strftime('%H:%M:%S')} — EXPIRAÇÃO: {hora_expiracao_local.strftime('%H:%M:%S')}
📊 PREVISÃO: O Robô acredita que o valor irá {formatted_previsao}
🤖 CONFIANÇA (Confiança que o robô tem da previsão): {confianca:.2f}%
📢 MOTIVO: Foram encontrados os seguintes padrõas no gráfico:
"""
    
    for msg in pattern_messages:
        log_message += f"    - {msg}\n"
        
    print(log_message)
