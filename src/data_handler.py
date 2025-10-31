

import os
import pandas as pd
from twelvedata import TDClient

def fetch_data(ticker: str, interval: str) -> pd.DataFrame:
    """
    Busca dados históricos de um par de moedas usando a biblioteca twelvedata.

    Args:
        ticker: O símbolo do ativo (ex: "BTC/BRL").
        interval: O intervalo entre as velas (ex: "1min").

    Returns:
        Um DataFrame do pandas com os dados OHLCV.
    """
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        print("ALERTA: Chave da API da Twelve Data não encontrada na variável de ambiente TWELVE_DATA_API_KEY.")
        print("Para uso real, obtenha uma chave gratuita em: https://twelvedata.com/apikey")
        raise ValueError("API Key da Twelve Data não configurada.")

    # Mapeia o intervalo para o formato da Twelve Data (ex: "5m" -> "5min")
    interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}
    if interval not in interval_map:
        raise ValueError(f"Intervalo '{interval}' não suportado. Use um de: {list(interval_map.keys())}")

    td = TDClient(apikey=api_key)

    # Busca os dados da série temporal, garantindo que o fuso horário seja UTC
    ts = td.time_series(
        symbol=ticker,
        interval=interval_map[interval],
        outputsize=5000,  # Solicita o máximo de pontos de dados possível
        timezone="UTC"  # Garante que os dados venham em UTC
    )

    df = ts.as_pandas()

    if df is None or df.empty:
        raise ValueError("Não foram encontrados dados para o ticker/intervalo especificado na Twelve Data.")

    # A biblioteca já retorna com índice de datetime e em ordem ascendente.
    # Apenas renomeamos as colunas para o padrão do projeto (letras maiúsculas).
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    return df
