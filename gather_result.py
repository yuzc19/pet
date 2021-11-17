import numpy as np


task_name = input()
final_result_test = np.zeros(5)
for i, seed in enumerate([13, 21, 42, 87, 100]):
    result_dir = "result/{}-{}/".format(task_name, seed)
    with open(result_dir + "result_test.txt", "r") as f:
        final_result_test[i] = float(f.readlines()[1].split()[1])

s = "mean +- std: %.1f (%.1f) (median %.1f)" % (
    final_result_test.mean() * 100,
    final_result_test.std() * 100,
    np.median(final_result_test) * 100,
)
print(s)
