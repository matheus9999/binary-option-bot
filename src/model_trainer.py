
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Para salvar o modelo
import os

def train_model(df: pd.DataFrame):
    """
    Treina, avalia e salva o modelo de machine learning.

    Args:
        df: DataFrame completo com features e alvo.

    Returns:
        O modelo treinado.
    """

    # 1. Seleção de Features
    # Excluímos colunas que não são features ou que dão a resposta ao modelo
    excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'price_future', 'target']
    features = [col for col in df.columns if col not in excluded_cols]
    X = df[features]
    y = df['target']

    # 2. Divisão de Treino e Teste (sem embaralhar para séries temporais)
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Não há dados suficientes para treinar/testar.")

    # 3. Treinamento
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 4. Avaliação Completa
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. Importância das Features (O "motivo" da previsão)
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    # 6. Salvar o modelo treinado
    model_path = "artifacts/trained_model.joblib"
    # Garante que o diretório de artefatos exista
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model
