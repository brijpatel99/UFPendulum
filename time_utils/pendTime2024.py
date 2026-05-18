from datetime import datetime

def is_valid_date(date_str, format_str):
    try:
        # Attempt to parse the string with the given format
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        # If it fails, the format is incorrect or the date is invalid
        return False

def pendTime2024(time=datetime.now(), mute=False):
    """
    Converts time into corresponding pendulum time (seconds since pendulum birthday in this case "2024-05-07 00:00:00")

    Args:
        time (string):  needs to be of the following format: "%Y-%m-%d %H:%M:%S"
                        takes string, converts to datetime and finds pendulum time at this time
             (empty):   uses the current time from datetime.now()

        mute (boolean): if True, don't print pendulum time
                        if False, print pendulum time 
    """

    # time string format
    fmt = "%Y-%m-%d %H:%M:%S"

    # time since pendulum birthday
    pendBirthday2024 = "2024-05-07 00:00:00" #pendBirthday = "2014-05-07 00:00:00"

    # current time or time inputted
    currTime = datetime.now()

    # if input time is already datetime, skip checks
    if isinstance(time, datetime):
        pass
    # if input time is string, check if it matches format
    elif is_valid_date(time, fmt):
        currTime = datetime.strptime(time,fmt)
    # if neither, print error
    else:
        #error
        print("INVALID TIME STRING")
    
    # calculate total seconds since pendulum birthday
    pendTime = (currTime - datetime.strptime(pendBirthday2024,fmt)).total_seconds()

    # if not mute, print pendulum time
    if not mute:
        print('Pendulum Time: ', pendTime, ' s')
    return pendTime