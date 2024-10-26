from datetime import datetime, timedelta

from utilities import logger


def round_to_15(minutes):
    # Round the minutes to the nearest 15
    return 15 * round(minutes / 15)


from datetime import datetime, timedelta


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


def get_day_month_and_time(date_time):
    # Parse the date string
    date = datetime.strptime(date_time, "%d/%m/%Y %H:%M")
    return date.strftime("%d/%m %H:%M")


def validate_date_time(date_time):
    formats = [
        "%d/%m/%Y %H:%M",  # Already in the correct format
        "%Y-%m-%d %H:%M:%S",  # ISO format
        "%m/%d/%Y %H:%M",  # US format
        "%d-%m-%Y %H:%M",  # Dash-separated format
        "%d/%m/%Y %H:%M:%S",  # With seconds
        "%Y-%m-%dT%H:%M:%S",  # ISO format with T separator
        "%d/%m/%y %H:%M",  # Full year format
        "%m/%d/%y %I:%M %p",  # US format with 2-digit year and AM/PM
        "%m/%d/%Y %I:%M %p",  # US format with 4-digit year and AM/PM
    ]

    for fmt in formats:
        try:
            # Try to parse the date string
            date_obj = datetime.strptime(date_time, fmt)
            # If successful, return the date in the desired format
            return date_obj.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date string: {date_time}")


def get_date_time_index(df, date_time):
    # validate the date time
    date_time = validate_date_time(date_time)
    # Create a dictionary to map the datetime to the index
    datetime_to_index = {date_time: i for i, date_time in enumerate(df["15 Minutes"])}
    # split into date and time
    datetime_value = datetime.strptime(date_time, "%d/%m/%Y %H:%M")
    date = datetime_value.day
    time = datetime_value.time().strftime("%H:%M")

    index = 0
    # find the index for the input datetime
    for date_str in datetime_to_index:
        # Parse each date in the array
        date_striped = datetime.strptime(date_str, "%d/%m/%Y %H:%M")
        # Check if day and time match
        if date_striped.day == date and date_striped.time().strftime("%H:%M") == time:
            # Set the index
            index = datetime_to_index[date_str]

    if index == 0:
        logger.log(
            "Not enough historical data for the given time. Predicting for the next or previous day."
        )
        new_date = date
        if date == 31:
            new_date -= 1
        else:
            new_date += 1
        for date_str in datetime_to_index:
            # Parse each date in the array
            date_striped = datetime.strptime(date_str, "%d/%m/%Y %H:%M")
            # Check if day and time match
            if (
                date_striped.day == new_date
                and date_striped.time().strftime("%H:%M") == time
            ):
                # Set the index
                index = datetime_to_index[date_str]

    return index