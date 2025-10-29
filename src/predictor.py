
import pandas as pd
import joblib

def make_prediction(df_full: pd.DataFrame):
    """
    Carrega o modelo treinado e faz uma previsão na vela mais recente.

    Args:
        df_full: O DataFrame completo com todas as features calculadas.
    """
    print("--- 4. Fazendo a Próxima Previsão ---")

    # 1. Carregar o modelo
    model = joblib.load("artifacts/trained_model.joblib")

    # 2. Selecionar as mesmas features usadas no treino
    excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'price_future', 'target']
    features = [col for col in df_full.columns if col not in excluded_cols]
    
    # 3. Pegar a última linha (os dados mais recentes)
    X_recent = df_full[features].iloc[[-1]]

    print("\nUsando as seguintes features (dados mais recentes):")
    print(X_recent.T.head(10)) # Mostra as 10 primeiras features dos dados recentes

    # 4. Fazer a previsão e obter probabilidades
    prediction = model.predict(X_recent)[0]
    probabilities = model.predict_proba(X_recent)[0]

    direction = "SUBIR" if prediction == 1 else "CAIR"
    confidence = probabilities[prediction] * 100

    print(f"\nPREVISÃO: O modelo prevê que o preço vai {direction}.")
    print(f"Confiança: {confidence:.2f}%")
