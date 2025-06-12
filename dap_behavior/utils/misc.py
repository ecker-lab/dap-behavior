from datetime import datetime


def timestamp(mode="long"):
    if mode == "short":
        return datetime.now().strftime("%y%m%d-%H%M")
    elif mode == "short_seconds":
        return datetime.now().strftime("%y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y-%m-%dT%H-%M-%S%Z")

