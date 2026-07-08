# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
import numpy as np
import subprocess
from multiprocessing import Queue, Process

PARALLELPROCESSDATA = None
def launchParallelProcess(f, *a, **k):
    global PARALLELPROCESSDATA

    PARALLELPROCESSDATA = [f, a, k]

    from multiprocessing import Process
    p = Process(target=_launchParallelProcess, args=tuple([]))
    p.start()
    PARALLELPROCESSDATA = None
    return p


def _launchParallelProcess():
    global PARALLELPROCESSDATA
    [f, a, k] = PARALLELPROCESSDATA
    try:
        f(*a, **k)
    except Exception as e:
        print(
            "Exception in worker during forking:\n%s" %
            (traceback.format_exc()))
        raise e

def parallel_workers(CPUs, jobs, callback):
    if len(jobs) < CPUs:
        print("warning: fewer jobs compared to CPUs. truncating number of CPUs.")
        CPUs = len(jobs)

    q = Queue()

    for j in jobs:
        launchParallelProcess(worker, callback, q, j)

    number_of_active_workers = len(jobs)

    while number_of_active_workers > 0:
        next_result = q.get()
        if next_result['finished']:
            number_of_active_workers -= 1
        else:
            yield next_result["value"]


def worker(callback, q, *arguments):
    for return_value in callback(*arguments):
        q.put({"value": return_value, "finished": False})
    q.put({"value": None, "finished": True})


if __name__ == "__main__":
    jobs = [9, 87534, 2298582173]

    def hunt_for_square_numbers(starting_index):
        for i in range(starting_index, starting_index + 1000):
            if int(int(i**0.5)**2) == i:
                print("worker has discovered that", i, "is a square number")
                yield i

    for a_square_number in parallel_workers(3, jobs, hunt_for_square_numbers):
        print("the head has received that", a_square_number, "is a square number")
