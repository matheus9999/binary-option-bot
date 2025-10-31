import pandas as pd
import numpy as np
from ta.momentum import rsi
from ta.trend import ema_indicator

def is_engulfing(df: pd.DataFrame) -> pd.Series:
    """Identifica padrão de Engolfo (Bullish e Bearish)."""
    close_1 = df['Close'].shift(1)
    open_1 = df['Open'].shift(1)
    close = df['Close']
    open = df['Open']

    bullish_engulfing = (close_1 < open_1) & (close > open) & (close > open_1) & (open < close_1)
    bearish_engulfing = (close_1 > open_1) & (close < open) & (close < open_1) & (open > close_1)

    return np.select([bullish_engulfing, bearish_engulfing], [1, -1], default=0)

def is_pin_bar(df: pd.DataFrame) -> pd.Series:
    """Identifica Pin Bar (Martelo e Estrela Cadente)."""
    body = abs(df['Open'] - df['Close'])
    wick = df['High'] - df['Low']
    
    # Martelo (Bullish)
    is_hammer = (df['High'] - df['Close']) > 2 * body
    is_hammer &= (df['Open'] - df['Low']) < body
    
    # Estrela Cadente (Bearish)
    is_shooting_star = (df['Open'] - df['Low']) > 2 * body
    is_shooting_star &= (df['High'] - df['Close']) < body

    return np.select([is_hammer, is_shooting_star], [1, -1], default=0)

def is_inside_bar(df: pd.DataFrame) -> pd.Series:
    """Identifica Inside Bar."""
    high_1 = df['High'].shift(1)
    low_1 = df['Low'].shift(1)
    high = df['High']
    low = df['Low']

    is_inside = (high < high_1) & (low > low_1)
    return is_inside.astype(int)

def is_marubozu(df: pd.DataFrame) -> pd.Series:
    """Identifica Marubozu (Bullish e Bearish)."""
    body = abs(df['Open'] - df['Close'])
    wick = df['High'] - df['Low']
    
    is_bullish_marubozu = (df['Close'] > df['Open']) & (body / wick > 0.95)
    is_bearish_marubozu = (df['Open'] > df['Close']) & (body / wick > 0.95)

    return np.select([is_bullish_marubozu, is_bearish_marubozu], [1, -1], default=0)


def create_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Cria features de análise técnica e o alvo para o modelo de ML.

    Args:
        df: DataFrame com dados OHLCV.
        lags: O número de períodos passados para usar como features (lags).

    Returns:
        DataFrame com as features e o alvo.
    """
    # 1. Features de Lag
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].pct_change(periods=i)

    # 2. Indicadores Técnicos da Estratégia
    df['ema_5'] = ema_indicator(df['Close'], window=5)
    df['ema_15'] = ema_indicator(df['Close'], window=15)
    df['ema_100'] = ema_indicator(df['Close'], window=100)
    df['rsi'] = rsi(df['Close'], window=14)

    # 3. Padrões de Candlestick
    df['engulfing'] = is_engulfing(df)
    df['pin_bar'] = is_pin_bar(df)
    df['inside_bar'] = is_inside_bar(df)
    df['marubozu'] = is_marubozu(df)

    # 4. Criação do Alvo (Target)
    df['price_future'] = df['Close'].shift(-1)
    df['target'] = np.where(df['price_future'] > df['Close'], 1, 0)

    # 5. Limpeza Final
    df = df.dropna()
    
    return df