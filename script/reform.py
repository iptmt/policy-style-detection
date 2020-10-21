import os
from fio import read, write

lines_0 = [(l.strip(), 1) for l in read("../data/yelp/style.test.0")]
lines_1 = [(l.strip(), 0) for l in read("../data/yelp/style.test.1")]

lines = [line + "\t" + line + "\t" + str(lb) for line, lb in lines_0 + lines_1]

write("../out/copy/yelp.test.tsf", lines)