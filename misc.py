from datetime import timedelta

def convert_to_timedelta(time_str):
    hours, minutes, seconds = map(int, time_str.split('.'))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)
