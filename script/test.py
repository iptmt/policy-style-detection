from fio import read, write

topic = "caae"

lines = read(f"../out/{topic}/gyafc_test.tsf")

lines = [line + "\t" + "1" for line in lines]
write(f"../out/{topic}/gyafc_test.tsf", lines)