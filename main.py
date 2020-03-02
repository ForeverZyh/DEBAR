# call the analysis_main.py
import sys
import os
import subprocess

from parse.specified_ranges import SpecifiedRanges

unbounded_weight = False
unbounded_input = False
interpreter_path = sys.executable
print("Running at: ", interpreter_path)
result_filename = "results.txt"

open(result_filename, 'w').close()

for model in SpecifiedRanges.models:
    if not unbounded_weight and not unbounded_input:
        os.system("(time %s ./analysis_main.py /newdisk/Yuhao/dnn_test/%s.pbtxt) >> %s 2>&1" % (interpreter_path, model, result_filename))
    elif unbounded_weight:
        os.system("(time %s ./analysis_main.py /newdisk/Yuhao/dnn_test/%s.pbtxt unbounded_weight) >> %s 2>&1" % (interpreter_path, model, result_filename))
    elif unbounded_input:
        os.system("(time %s ./analysis_main.py /newdisk/Yuhao/dnn_test/%s.pbtxt unbounded_input) >> %s 2>&1" % (interpreter_path, model, result_filename))
        
lines = open(result_filename).readlines()
times = []
sats = []
unsats = []
for line in lines:
    if line.find("user") != -1:
        user_time = float(line[:line.find("user")])
        line = line[line.find("user") + 5:]
        sys_time = float(line[:line.find("system")])
        times.append(user_time + sys_time)
    if line.find("sat") != -1 and len(line) > 10:
        splits = line.split()
        sats.append(int(splits[3]))
        unsats.append(int(splits[5]))

print(sats, unsats, times)
