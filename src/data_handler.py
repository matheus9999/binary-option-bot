
import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Busca dados históricos de um ticker usando a biblioteca yfinance.

    Args:
        ticker: O símbolo do ativo (ex: "EURUSD=X").
        period: O período dos dados (ex: "1mo", "1y").
        interval: O intervalo entre as velas (ex: "5m", "1h").

    Returns:
        Um DataFrame do pandas com os dados OHLCV.
    """
    print(f"\n--- 1. Coletando Dados para {ticker} ---")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    print(f"Dados coletados: {len(df)} velas.")
    
    if df.empty:
        raise ValueError("Não foram encontrados dados para o ticker/período especificado.")

    return df
