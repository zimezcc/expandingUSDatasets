import os
import random

DIR_benign = "../Datasets/USDatasetsClean/benign"
DIR_malignant = "../Datasets/USDatasetsClean/malignant"

TRAIN_FILE = "train_data_list/train.txt"
TEST_FILE = "test.txt"

random.seed(42)  # 保证可复现，不需要可删掉

def collect_names(directory):
    """递归收集目录下所有文件名（含相对路径，便于区分来源）"""
    names = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            names.append(path)
    return names

def split_7_3(names):
    random.shuffle(names)
    idx = int(len(names) * 0.7)
    return names[:idx], names[idx:]


# 分别收集
benign_names = collect_names(DIR_benign)
malignant_names = collect_names(DIR_malignant)

print(f"benign    : {len(benign_names)} 个文件")
print(f"malignant : {len(malignant_names)} 个文件")

# 分别按 7:3 划分
benign_train, benign_test = split_7_3(benign_names)
malignant_train, malignant_test = split_7_3(malignant_names)

# 合并
train_names = benign_train + malignant_train
test_names = benign_test + malignant_test

# 写入文件
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for name in train_names:
        f.write(name + "\n")

with open(TEST_FILE, "w", encoding="utf-8") as f:
    for name in test_names:
        f.write(name + "\n")

# 统计
print("-" * 40)
print(f"benign    -> train {len(benign_train)}, test {len(benign_test)}")
print(f"malignant -> train {len(malignant_train)}, test {len(malignant_test)}")
print(f"总计      -> train.txt {len(train_names)} 条, test.txt {len(test_names)} 条")