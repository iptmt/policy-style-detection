import sys
from fio import read, write


def reorg(file_list, out, order): # order: [0, 2, 1]
    multi_lines = [read(f) for f in file_list]
    line_tuples = list(zip(*multi_lines))
    line_tuples = [[e for ele in line_tuple for e in ele.strip().split("\t")] for line_tuple in line_tuples]
    re_order_tuples = [[tp[order[i]].strip() for i in range(len(order))] for tp in line_tuples]
    lines = ["\t".join(tp) for tp in re_order_tuples]
    write(out, lines)

def append(inp, out, label):
    lines = read(inp)
    lines = [l.strip() + "\t" + str(label) for l in lines]
    write(out, lines)

def check(inp, out):
    lines = read(inp)
    lines = [l.split("\t") for l in lines]
    new_lines = []
    for line in lines:
        if len(line) < 3:
            line = [line[0], ".", line[1]]
        new_lines.append("\t".join(line))
    write(out, new_lines)



if __name__ == "__main__":
    model = "tg"
    data = "gyafc"

    reorg([f"../data/gyafc/style.test.0", f"../out/{model}/{data}_test.tsf"], f"../out/{model}/{data}_test.tsf", [0, 1])
    append(f"../out/{model}/{data}_test.tsf", f"../out/{model}/{data}_test.tsf", "1")
    # check(f"../out/{model}/{data}_test.tsf", f"../out/{model}/{data}_test.tsf")