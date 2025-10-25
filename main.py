import os
import time
import schedule
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

env_ticker = os.getenv("TICKER")
env_period = os.getenv("TICKER_PERIOD")
env_interval = os.getenv("TICKER_INTERVAL")
env_last_count = os.getenv("TICKER_LAST_COUNT")

def main():
    ticker = yf.Ticker(env_ticker)
    ticker_history = ticker.history(period=env_period, interval=env_interval)
    print(f"\n--- {env_ticker} ---")
    print(ticker_history.tail(int(env_last_count)))

schedule.every(10).seconds.do(main)

print("========= ROBO INICIADO =========")

while True:
    schedule.run_pending()
    time.sleep(1)