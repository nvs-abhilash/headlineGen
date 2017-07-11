#!/usr/bin/env python
import subprocess
with open("output.txt", "w+") as output:

    i = 1
    while i < 51:
        fName = '{:0>3}.txt'.format(i)
        subprocess.call(["python", "./reuters_experiment.py", fName], stdout=output);
        i += 1