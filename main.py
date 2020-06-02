# call the analysis_main.py
import sys
import os
import time

from parse.specified_ranges import SpecifiedRanges

path = "/tmp"
try:
    assert len(sys.argv) == 2
    path = sys.argv[1]

except:
    print(
        "Please run 'python main.py PATH_TO_DATASETS'.\nAborted...")
    exit(1)

unbounded_weight = False
unbounded_input = False
interpreter_path = sys.executable
print("Running at: ", interpreter_path)
result_filename = "results.txt"

open(result_filename, 'w').close()

times = {}
for model in SpecifiedRanges.models:
    t0 = time.time()
    print("Running %s" % model)
    if not unbounded_weight and not unbounded_input:
        os.system(
            "(%s ./analysis_main.py %s/%s.pbtxt) >> %s 2>&1" % (interpreter_path, path, model, result_filename))
    elif unbounded_weight:
        os.system("(%s ./analysis_main.py %s/%s.pbtxt unbounded_weight) >> %s 2>&1" % (
            interpreter_path, path, model, result_filename))
    elif unbounded_input:
        os.system("(%s ./analysis_main.py %s/%s.pbtxt unbounded_input) >> %s 2>&1" % (
            interpreter_path, path, model, result_filename))
    times[model] = time.time() - t0

lines = open(result_filename).readlines()
# times = []
info = {}
for line in lines:
    # if line.find("user") != -1:
    #     user_time = float(line[:line.find("user")])
    #     line = line[line.find("user") + 5:]
    #     sys_time = float(line[:line.find("system")])
    #     times.append(user_time + sys_time)
    if line.find("warnings") != -1 and len(line) > 10:
        splits = line.split()
        model_name = splits[0]
        info[model_name] = line.strip()

for model in SpecifiedRanges.models:
    if model in info:
        print(info[model] + "\t in time: %.2f" % times[model])
    else:
        print("Runtime error when running %s." % model)
