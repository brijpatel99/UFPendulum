from datetime import datetime

def pendTimeLTPDA(pendTime:float, mute:bool=True):
    """
    Converts pendulum time (seconds since pendulum birthday in this case "2024-05-07 00:00:00")
    to LTPDA date number (seconds since 1970-01-01 00:00:00.000 UTC)
    Args:
        time (float): usually product of pendTime2024()

        mute (boolean): if True, don't print LTPDA time
                        if False, print LTPDA time 
    """

    # time string format
    fmt = "%Y-%m-%d %H:%M:%S"

    # time since pendulum birthday
    pendBirthday2024 = "2024-05-07 00:00:00" #pendBirthday = "2014-05-07 00:00:00"

    # calculate total seconds since pendulum birthday
    LTPDATime = pendTime + (datetime.strptime(pendBirthday2024,fmt).toordinal() 
                         - datetime.strptime("1970-01-01 00:00:00",fmt).toordinal())*86400

    # if not mute, print pendulum time
    if not mute:
        print('LTPDA Time: ', LTPDATime, ' s')
    return LTPDATime