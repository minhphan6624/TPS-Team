from datetime import datetime

def log(message):
    # Print current time only and the message
    print(f'[{datetime.now().strftime("%H:%M:%S")}]: {message}')