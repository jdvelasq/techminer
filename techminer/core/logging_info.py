import datetime


def logging_info(msg):
    print(
        "{} - INFO - {}".format(
            datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), msg
        )
    )

