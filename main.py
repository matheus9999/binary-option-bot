import os
import schedule
import time
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier # O modelo de ML
from sklearn.metrics import accuracy_score # Para ver a precisão
from dotenv import load_dotenv # Carregar variaveis de ambiente


# --- 1. CARREGAR AS VARIÁVEIS DE AMBIENTE ---

load_dotenv()

env_time = os.getenv("BOT_SCHEDULE_IN_MINUTE")
env_ticker = os.getenv("TICKER")
env_ticker_period = os.getenv("TICKER_PERIOD")
env_ticker_interval = os.getenv("TICKER_INTERVAL")
env_lags = os.getenv("LAGS")

def run_bot():
    # --- INICIANDO ---

    print("\n=== INICIANDO EXECUÇÃO DO BOT ===\n")
    print(f"Ticket: {env_ticker}")
    print(f"Período: {env_ticker_period}")
    print(f"Intervalo: {env_ticker_interval}")
    print(f"Qtd. de vela para analise: {env_lags}")
    print(f"Tempo de execução do robô: a cada {env_time} minuto")

    # --- 2. COLETA DE DADOS ---

    ticker = yf.Ticker(env_ticker)
    df = ticker.history(period=env_ticker_period, interval=env_ticker_interval)

    # Vamos focar apenas no preço de fechamento (Close)
    df = df[['Close']]

    # --- 3. ENGENHARIA DE FEATURES (Criando "Pistas" para o Modelo) ---

    # Queremos que o modelo olhe para "lags" (preços anteriores) para tomar uma decisão.
    # Vamos criar features baseadas nos retornos percentuais de minutos anteriores.
    # (Retorno = (preço_agora - preço_anterior) / preço_anterior)

    lags = int(env_lags)

    # Criamos colunas 'lag_1', 'lag_2', ... 'lag_5'
    # lag_1 = retorno do minuto anterior
    # lag_2 = retorno de 2 minutos atrás
    # etc.
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].pct_change(periods=i)

    # Vamos adicionar também uma Média Móvel (Moving Average)
    df['MA_10m'] = df['Close'].rolling(window=10).mean()
    # Feature: O preço atual está acima ou abaixo da média dos últimos 10 min?
    df['feature_MA'] = np.where(df['Close'] > df['MA_10m'], 1, -1)


    # --- 4. CRIAÇÃO DO ALVO (O que queremos prever) ---

    # Queremos prever se o PRÓXIMO minuto (t+1) vai subir ou cair
    # Usamos .shift(-1) para trazer o preço do futuro para a linha atual
    df['preco_futuro'] = df['Close'].shift(-1)

    # Nosso "Alvo" (y):
    # 1 se o preço futuro for MAIOR que o preço atual (Sobe)
    # 0 se o preço futuro for MENOR ou IGUAL ao preço atual (Cai/Fica igual)
    df['Alvo'] = np.where(df['preco_futuro'] > df['Close'], 1, 0)


    # --- 5. LIMPEZA E PREPARAÇÃO DOS DADOS ---

    # A criação de lags e médias móveis gera valores "NaN" (Nulos) no início.
    # Vamos remover todas as linhas que contenham qualquer NaN.
    df = df.dropna()

    # Separar os dados em:
    # X (Features): As "pistas" que o modelo usa (os lags)
    # y (Target): O que queremos prever (o "Alvo")

    # Nossas features são as colunas de lag e a feature da Média Móvel
    features_list = [f'lag_{i}' for i in range(1, lags + 1)] + ['feature_MA']
    X = df[features_list]

    y = df['Alvo']

    # --- 6. DIVISÃO DE TREINO E TESTE ---

    # IMPORTANTE: Para séries temporais, NUNCA podemos embaralhar (shuffle=False).
    # Temos que treinar com o passado para prever o futuro.
    # Vamos usar 80% dos dados para treinar e 20% para testar.
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Verificando se temos dados suficientes
    if len(X_train) == 0 or len(X_test) == 0:
        print("\nNão há dados suficientes para treinar/testar após a limpeza.")
        print(f"Total de linhas válidas: {len(df)}")
    else:
        print(f"\nAmostra de treino: {len(X_train)}")
        print(f"Amostra de teste: {len(X_test)}")

        # --- 7. TREINAMENTO DO MODELO ---
        
        # RandomForest é um bom modelo inicial (um conjunto de "árvores de decisão")
        # n_estimators = número de árvores
        # min_samples_leaf = mínimo de amostras para tomar uma decisão (evita overfitting)
        # random_state = para resultados reprodutíveis
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
        
        print("\nIniciando treinamento do modelo...")
        model.fit(X_train, y_train)
        
        # --- 8. AVALIAÇÃO (BÁSICA) ---
        
        # Fazendo previsões nos dados de teste (dados que o modelo nunca viu)
        y_pred = model.predict(X_test)
        
        # Qual a precisão (accuracy)?
        # (Quantas vezes ele acertou se ia subir ou cair)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n--- Avaliação do Modelo (em dados de teste) ---")
        print(f"Precisão (Accuracy): {accuracy * 100:.2f}%")
        
        # Uma precisão de 50% é o mesmo que jogar uma moeda.
        # No mercado, 52-54% já é considerado bom (mas pode não ser lucrativo!)
        
        # Comparando com um modelo "ingênuo" (que sempre chuta "Sobe")
        # Isso nos diz se o modelo é melhor que um chute aleatório
        print(f"Distribuição real Sobe/Cai (Teste): {np.mean(y_test) * 100:.2f}% de 'Sobe'")
        

        # --- 9. FAZENDO A "PRÓXIMA" PREVISÃO ---
        
        # Para prever o próximo minuto REAL, precisamos das features MAIS RECENTES
        # No nosso caso, é a última linha dos nossos dados 'X'
        
        X_recente = X.iloc[[-1]] # Pega a última linha do DataFrame X
        
        print("\n--- Previsão para o Próximo Minuto ---")
        print("Usando as seguintes features (dados mais recentes):")
        print(X_recente)
        
        proxima_previsao = model.predict(X_recente)
        proxima_probabilidade = model.predict_proba(X_recente)
        
        previsao = "SUBIR" if proxima_previsao[0] == 1 else "CAIR"

        print(f"\nPREVISÃO: O modelo acha que o preço vai {previsao} no próximo minuto.")
        print(f"(Confiança: {proxima_probabilidade[0][1]*100:.2f}%)")
        

# Chamamos a função diretamente para a primeira execução
run_bot()

# Agendamos a função para rodar a cada minuto programado
schedule.every(int(env_time)).minutes.do(run_bot)

while True:
    # Verifica e executa as tarefas agendadas que estão prontas
    schedule.run_pending()
    # Pausa o programa por 1 segundo para não consumir 100% da CPU
    time.sleep(1)