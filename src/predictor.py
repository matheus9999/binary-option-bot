
import pandas as pd
import joblib
from datetime import datetime
import pytz
from src.log_handler import log_operation

def make_prediction(df_full: pd.DataFrame, user: str, ticker: str, interval: str):
    """
    Carrega o modelo treinado, faz uma previsão e registra a operação.

    Args:
        df_full: DataFrame com todas as features calculadas.
        user: O nome do usuário que esta executando o robô.
        ticker: O ativo sendo analisado (ex: "EURUSD=X").
        interval: O intervalo de tempo da vela (ex: "5m").
    """
    # 1. Carregar o modelo
    model = joblib.load("artifacts/trained_model.joblib")

    # 2. Selecionar as mesmas features usadas no treino
    excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'price_future', 'target']
    features = [col for col in df_full.columns if col not in excluded_cols]
    
    # 3. Pegar a última linha (os dados mais recentes)
    X_recent = df_full.iloc[[-1]]
    X_for_prediction = X_recent[features]

    # 4. Fazer a previsão e obter probabilidades
    prediction_result = model.predict(X_for_prediction)[0]
    probabilities = model.predict_proba(X_for_prediction)[0]

    direction = "SUBIR" if prediction_result == 1 else "CAIR"
    confidence = probabilities[prediction_result] * 100
    
    # 5. Coletar dados para o log
    # ATENÇÃO: Usando a hora atual da predição em UTC, não a hora da vela de dados.
    # Isso garante que o log reflita a hora real da operação.
    hora_entrada = datetime.now(pytz.utc)
    
    pattern_cols = ['engulfing', 'pin_bar', 'inside_bar', 'marubozu']
    padroes_encontrados = {col: X_recent[col].iloc[0] for col in pattern_cols if col in X_recent}

    # 6. Chamar o log unificado
    log_operation(
        user=user,
        ativo=ticker,
        expiracao=interval,
        hora_entrada=hora_entrada,
        previsao=direction,
        confianca=confidence,
        padroes=padroes_encontrados
    )
