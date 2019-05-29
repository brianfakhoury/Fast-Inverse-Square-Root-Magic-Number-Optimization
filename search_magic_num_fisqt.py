#!/usr/bin/env python3

import struct
import multiprocessing
import time
import sys

num_cpu = multiprocessing.cpu_count()

# github.com/ajcr/ajcr.github.io/blob/master/_posts/2016-04-01-fast-inverse-square-root-python.md


def struct_isqrt(number, magic=0x5f3759df):
    threehalfs = 1.5
    x2 = number * 0.5
    y = number

    packed_y = struct.pack('f', y)
    i = struct.unpack('i', packed_y)[0]  # treat float's bytes as int
    i = magic - (i >> 1)            # arithmetic with magic number
    packed_i = struct.pack('i', i)
    y = struct.unpack('f', packed_i)[0]  # treat int's bytes as float

    y = y * (threehalfs - (x2 * y * y))  # Newton's method
    return y


def test(val, set_range, index_ret, best_vals, progress):
    name = multiprocessing.current_process().name
    # print("Starting Process:", name)
    actual = val ** -0.5
    best_magic = 0x0
    best_diff = 1000
    prog = 0
    for curr_magic in set_range:
        if curr_magic % 1000000 == 0:
            progress[index_ret] = prog
        ans = struct_isqrt(val, curr_magic)
        if abs(ans - actual) < best_diff:
            best_magic = curr_magic
            best_diff = abs(ans - actual)
        prog += 1
    progress[index_ret] = prog
    best_vals[index_ret] = best_magic


def progressBar(value, endvalue, bar_length=20):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rProgress: [{0}] {1}%".format(
        arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def run_tests(val, int_size=2147483647, reduced_search_space=True):
    # print("Detected", num_cpu, "Cores. Creating", num_cpu, "Workers.")
    best_vals = multiprocessing.Array('i', num_cpu)
    progress = multiprocessing.Array('i', num_cpu)
    processes = []
    if reduced_search_space:
        j = 0
        for i in range(int(951.75 * num_cpu), int(952.75 * num_cpu)):
            processes.append(multiprocessing.Process(target=test, args=(val,
                                                                        range(i * int_size // (1280 * num_cpu), (i + 1) * int_size // (1280 * num_cpu)), j, best_vals, progress)))
            processes[j].start()
            j += 1
    else:
        for i in range(num_cpu):
            processes.append(multiprocessing.Process(target=test, args=(val,
                                                                        range(i * int_size // num_cpu, (i + 1) * int_size // num_cpu), i, best_vals, progress)))
            processes[i].start()
    # curr_prog = sum(progress)
    # if reduced_search_space:
    #     int_size = int_size // 1280
    # while(curr_prog < int_size):
    #     progressBar(curr_prog, int_size)
    #     time.sleep(0.5)
    #     curr_prog = sum(progress)
    # progressBar(100, 100)
    # print()
    for i in range(num_cpu):
        processes[i].join()
        # print("Process-" + str(i), "Ended.")
    bestNum = 0
    actual = val ** -0.5
    for num in best_vals:
        if abs(struct_isqrt(val, num) - actual) < abs(struct_isqrt(val, bestNum) - actual):
            bestNum = num
    print("Detected Optimal Magic Number For Val =", val, ":", hex(bestNum))
    return bestNum


if __name__ == '__main__':
    ans = run_tests(56)
    print("Detected Optimal Magic Number For Val=:", hex(ans))
