from datetime import datetime, timedelta

from utilities import logger


def round_to_15(minutes):
    # Round the minutes to the nearest 15
    return 15 * round(minutes / 15)


def round_to_nearest_15_minutes(time_str):
    # Convert 12-hour time to 24-hour format
    time_obj = datetime.strptime(time_str, "%I:%M")

    # Get the rounded minutes
    rounded_minutes = round_to_15(time_obj.minute)

    # Adjust for the rounded minutes possibly overflowing into the next hour
    if rounded_minutes == 60:
        time_obj += timedelta(hours=1)
        rounded_minutes = 0

    # Return the formatted time
    return time_obj.strftime(f"%H:{rounded_minutes:02d}")

def validate_date_time(date_time):
    formats = [
        "%d/%m/%Y %H:%M",  # Already in the correct format
        "%Y-%m-%d %H:%M:%S",  # ISO format
        "%m/%d/%Y %H:%M",  # US format
        "%d-%m-%Y %H:%M",  # Dash-separated format
        "%d/%m/%Y %H:%M:%S",  # With seconds
        "%Y-%m-%dT%H:%M:%S",  # ISO format with T separator
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
    datetime_to_index = { date_time: i for i, date_time in enumerate(df["15 Minutes"]) }
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
        logger.log("Not enough historical data for the given time. Predicting for the next or previous day.")
        new_date = date
        if date == 31:
            new_date -= 1
        else:
            new_date += 1
        for date_str in datetime_to_index:
            # Parse each date in the array
            date_striped = datetime.strptime(date_str, "%d/%m/%Y %H:%M")
            # Check if day and time match
            if date_striped.day == new_date and date_striped.time().strftime("%H:%M") == time:
                # Set the index
                index = datetime_to_index[date_str]

    return index
