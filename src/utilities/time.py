from datetime import datetime, timedelta

from utilities import logger

def round_to_nearest_15_minutes(time_str):
    formats = ["%H:%M", "%I:%M %p"]  # 24-hour and 12-hour formats

    for fmt in formats:
        try:
            # Try parsing with the current format
            time_obj = datetime.strptime(time_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unable to parse time string: {time_str}")

    # Round the time to the nearest 15 minutes
    rounded_minute = (time_obj.minute // 15) * 15
    time_obj = time_obj.replace(minute=rounded_minute, second=0, microsecond=0)

    return time_obj.strftime("%H:%M")

def get_day_of_week(date):
    # Parse the date string
    date = datetime.strptime(date, "%d/%m/%Y")
    return date.strftime("%A")

def format_date_to_words(date_time):
    # Parse the date string
    date = datetime.strptime(date_time, "%d/%m/%Y %H:%M")
    
    # Get the day with ordinal suffix
    day = ordinal(date.day)
    
    # Format the rest of the date and combine
    return f"{day} {date.strftime('%h, %I:%M %p')}"

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"