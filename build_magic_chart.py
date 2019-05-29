#!/usr/bin/env python3
from search_magic_num_fisqt import run_tests
import os

f = open("best_values.txt", "a+")

completed = os.path.getsize("best_values.txt") // 11
val = (completed + 1) * 0.1
print("Completed:", completed)
print("Starting val:", val)

while val <= 100:
    print("Running test on val=", val)
    ans = run_tests(val)
    f.write(str(ans) + ",")
    val += 0.1

f.close()
