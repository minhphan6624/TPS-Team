from datetime import datetime, timedelta


def round_to_15(minutes):
    # Round the minutes to the nearest 15
    return 15 * round(minutes / 15)


def round_to_nearest_15_minutes(time_str):
    # Convert 12-hour time to 24-hour format
    time_obj = datetime.strptime(time_str, "%I:%M %p")

    # Get the rounded minutes
    rounded_minutes = round_to_15(time_obj.minute)

    # Adjust for the rounded minutes possibly overflowing into the next hour
    if rounded_minutes == 60:
        time_obj += timedelta(hours=1)
        rounded_minutes = 0

    # Return the formatted time
    return time_obj.strftime(f"%H:{rounded_minutes:02d}")
