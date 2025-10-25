import time
import schedule


def main():
    print("Tarefa executada a cada 10s.")

schedule.every(10).seconds.do(main)

print("========= ROBO INICIADO =========")

while True:
    schedule.run_pending()
    time.sleep(1)