from datetime import timedelta
import os


class Time_it():

    def __init__(self) -> None:
        pass

    def time_it(self, tic, toc, path, run):
        elapsed_time = toc - tic
        formatted_time = str(timedelta(seconds=elapsed_time))

        path = os.path.join(path, "rendering_times.txt")
        file_exists = os.path.exists(path)

        times_txt = open(path, "a")

        if not file_exists:
            times_txt.write(" {} {}\n".format("Task/Label".ljust(16), "Time"))
            times_txt.write(" {} {}\n".format(str(run).ljust(16), str(formatted_time).ljust(16)))

        else:
            times_txt.write(" {} {}\n".format(str(run).ljust(16), str(formatted_time).ljust(16)))

        times_txt.close()

    def write_to_txt(self, path, label, message):
        path = os.path.join(path, "rendering_times.txt")
        times_txt = open(path, "a")
        times_txt.write(" {} {}\n".format(str(label).ljust(16), str(message).ljust(16)))
        times_txt.close()
