import os

IN_TXT1 = [
    "synthetic2_BW0_list.txt",
    "synthetic2_BW10_list.txt",
    "synthetic2_BW20_list.txt",
    "synthetic2_BW30_list.txt",
    "synthetic2_BW40_list.txt",
]

OUT_DIR = "train_data_list"
os.makedirs(OUT_DIR, exist_ok=True)

def read_lines(path):
    """读文件，去空行，统一分隔符"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]

for filename in IN_TXT1:
    name = os.path.splitext(os.path.basename(filename))[0].replace("_list", "")
    out_txt = os.path.join(OUT_DIR, f"train+normal+{name}.txt")

    # 用 set 收集 normal 路径（自动去重）
    normal_set = set()
    for line in read_lines(filename):
        base = os.path.basename(line)
        normal_id = base.split("_")[1].split(".")[0]
        normal_set.add(f"../Datasets/USDatasetsClean/normal/{normal_id}.png")

    # 用 set 合并所有行去重
    all_lines = set()
    all_lines.update(read_lines(os.path.join(OUT_DIR, "train.txt")))
    all_lines.update(read_lines(filename))
    all_lines.update(normal_set)

    # 排序保证输出稳定
    lines = sorted(all_lines)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"写入 {out_txt}: {len(lines)} 条（去重后）")

