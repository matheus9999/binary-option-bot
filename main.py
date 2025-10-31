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
USER = os.getenv("USER_BOT", "@FERNANDINHO TRADER")
TICKER = os.getenv("TICKER", "BTC")
INTERVAL = os.getenv("INTERVAL", "1min")
LAGS = int(os.getenv("LAGS", 5))
BOT_SCHEDULE_IN_MINUTE = int(os.getenv("BOT_SCHEDULE_IN_MINUTE", 1))

def job():
    """Função principal do bot que executa todo o pipeline."""

    try:
        # --- ETAPA 1: COLETA DE DADOS ---
        df_raw = fetch_data(TICKER, INTERVAL)

        # --- ETAPA 2: ENGENHARIA DE FEATURES ---
        df_featured = create_features(df_raw.copy(), lags=LAGS)

        # --- ETAPA 3: TREINAMENTO E AVALIAÇÃO ---
        train_model(df_featured.copy())

        # --- ETAPA 4: PREVISÃO ---
        make_prediction(df_featured.copy(), user=USER, ticker=TICKER, interval=INTERVAL)

    except ValueError as e:
        print(f"\nERRO: {e}")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")


if __name__ == "__main__":
    # Executa o trabalho uma vez imediatamente ao iniciar o script
    job()

    # Agenda a execução para o futuro
    schedule.every(BOT_SCHEDULE_IN_MINUTE).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)