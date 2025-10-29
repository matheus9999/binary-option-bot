import os
import schedule
import time
from dotenv import load_dotenv
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Ignora os avisos de depreciação da biblioteca 'ta' e avisos de métricas do sklearn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.data_handler import fetch_data
from src.feature_engineering import create_features
from src.model_trainer import train_model
from src.predictor import make_prediction

# --- 1. CONFIGURAÇÃO INICIAL ---
load_dotenv()

# Carrega as variáveis de ambiente com valores padrão para segurança
TICKER = os.getenv("TICKER", "EURUSD=X")
TICKER_PERIOD = os.getenv("TICKER_PERIOD", "1mo")
TICKER_INTERVAL = os.getenv("TICKER_INTERVAL", "5m")
LAGS = int(os.getenv("LAGS", 5))
BOT_SCHEDULE_IN_MINUTE = int(os.getenv("BOT_SCHEDULE_IN_MINUTE", 15))

def job():
    """Função principal do bot que executa todo o pipeline."""
    print("\n=========================================")
    print(f"=== INICIANDO EXECUÇÃO DO BOT: {time.ctime()} ===")
    print("=========================================")
    print(f"Ativo: {TICKER} | Intervalo: {TICKER_INTERVAL} | Período: {TICKER_PERIOD}")

    try:
        # --- ETAPA 1: COLETA DE DADOS ---
        df_raw = fetch_data(TICKER, TICKER_PERIOD, TICKER_INTERVAL)

        # --- ETAPA 2: ENGENHARIA DE FEATURES ---
        df_featured = create_features(df_raw.copy(), lags=LAGS)

        # --- ETAPA 3: TREINAMENTO E AVALIAÇÃO ---
        # Em um cenário real, o treino pode não ser executado a cada vez.
        # Poderia ser feito diariamente ou semanalmente.
        train_model(df_featured.copy())

        # --- ETAPA 4: PREVISÃO ---
        # A previsão usa o último dado disponível após a criação de features.
        make_prediction(df_featured.copy())

    except ValueError as e:
        print(f"\nERRO: {e}")
        print("A execução foi interrompida.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
        # Em um ambiente de produção, seria bom ter um log mais detalhado.

    print("\n=== EXECUÇÃO FINALIZADA ===")


if __name__ == "__main__":
    # Executa o trabalho uma vez imediatamente ao iniciar o script
    job()

    # Agenda a execução para o futuro
    print(f"\nAgendando a próxima execução para daqui a {BOT_SCHEDULE_IN_MINUTE} minutos.")
    schedule.every(BOT_SCHEDULE_IN_MINUTE).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)