import csv


def read(file_name):
    with open(file_name, 'r') as f:
        lines = [line.strip() for line in f]
        return list(filter(lambda x: x, lines))

def write(file_name, obj):
    with open(file_name, "w+") as f:
        for x in obj:
            f.write(x.strip() + "\n")

def read_csv(file_name):
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count, outputs = 0, []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            outputs.append(row)
            line_count += 1
        return outputs
    
def split_lines(file_name, delimiter):
    return [line.split(delimiter) for line in read(file_name)]