import os

IN_TXT1 = [
    "synthetic2_BW0_list.txt",
    "synthetic2_BW10_list.txt",
    "synthetic2_BW20_list.txt",
    "synthetic2_BW30_list.txt",
    "synthetic2_BW40_list.txt",
]


def read_lines(path):
    """读文件，去空行，统一分隔符"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


for filename in IN_TXT1:
    name = os.path.splitext(filename)[0].replace("_list", "")
    OUT_TXT = os.path.join("train_data_list", f"train+normal+{name}.txt")

    lines = (
        read_lines("train_data_list/train.txt")
        + read_lines("normal_list.txt")
        + read_lines(filename)
    )

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"写入 {OUT_TXT}: {len(lines)} 条")