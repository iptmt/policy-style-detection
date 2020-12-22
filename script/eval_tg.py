import re, sys
from fio import read, write

ds = sys.argv[1]
def replace(text):
    return re.sub(r"\[.*?\]", "<PH>", text)

tmp = "/home/zaracs/workspace/policy-style-detection/tmp/"
base = "/home/zaracs/workspace/baselines/tagger-generator/tag-and-generate-data-prep/data/"
f0 = "entagged_parallel.test.en.P_0"
f1 = "entagged_parallel.test.en.P_9"

f0_ = "entagged_parallel.test.tagged.P_0"
f1_ = "entagged_parallel.test.tagged.P_9"

tt0 = [x + "\t" + replace(x_) + "\t" + "0" for x, x_ in zip(read(base + f0), read(base + f0_))]
tt1 = [x + "\t" + replace(x_) + "\t" + "1" for x, x_ in zip(read(base + f1), read(base + f1_))]

write(tmp + f"{ds}.test.tfidf", tt0 + tt1)