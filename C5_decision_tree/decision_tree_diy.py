import math
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# =========================================================
# 1. 构造一个简单的猫狗数据集
# =========================================================
dataset = [
    {"ear_shape": "pointy", "sound": "meow", "tail": "long",  "likes_water": "no",  "label": "cat"},
    {"ear_shape": "pointy", "sound": "meow", "tail": "long",  "likes_water": "no",  "label": "cat"},
    {"ear_shape": "pointy", "sound": "meow", "tail": "short", "likes_water": "no",  "label": "cat"},
    {"ear_shape": "pointy", "sound": "meow", "tail": "long",  "likes_water": "yes", "label": "cat"},
    {"ear_shape": "pointy", "sound": "meow", "tail": "short", "likes_water": "no",  "label": "cat"},
    {"ear_shape": "floppy", "sound": "bark", "tail": "long",  "likes_water": "yes", "label": "dog"},
    {"ear_shape": "floppy", "sound": "bark", "tail": "long",  "likes_water": "yes", "label": "dog"},
    {"ear_shape": "floppy", "sound": "bark", "tail": "short", "likes_water": "yes", "label": "dog"},
    {"ear_shape": "floppy", "sound": "bark", "tail": "long",  "likes_water": "no",  "label": "dog"},
    {"ear_shape": "floppy", "sound": "bark", "tail": "short", "likes_water": "yes", "label": "dog"},
]

features = ["ear_shape", "sound", "tail", "likes_water"]


# =========================================================
# 2. 计算熵
# =========================================================
def entropy(data):
    """
    计算数据集标签的熵
    H(S) = - sum(p_i * log2(p_i))
    """
    labels = [row["label"] for row in data]

    count = Counter(labels)
    total = len(labels)

    ent = 0.0
    for c in count.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


# =========================================================
# 3. 按某个特征切分数据
# =========================================================
def split_dataset(data, feature):

    # print("feature: =========== ", feature)
    groups = {}                     # 空字典
    
    for row in data:                # 遍历每个样本
        key = row[feature]          # 获取特征值（如 "sound"）
        
        if key not in groups:       # 如果这个特征值还没出现过
            groups[key] = []        # 先创建空列表
        
        groups[key].append(row)     # 把当前样本加进去
    
    return groups

# =========================================================
# 4. 计算信息增益
# =========================================================

def information_gain(data, feature):

#     """
#     IG(S, A) = H(S) - sum( |Sv|/|S| * H(Sv) )
#     """

    # 1. 分裂前的混乱程度
    before_split_entropy = entropy(data)
    
    print("data: ============", data)

    # 2. 按特征值分组
    groups = split_dataset(data, feature)

    # print("group: =============", groups)

    total_samples = len(data)

    # print("=========== total samples ============ ", total_samples)
    
    # 3. 计算分裂后的平均混乱程度
    after_split_entropy = 0
    for group_name, group_data in groups.items():  # 更明确的命名

        print(f"group_name: {group_name}, group_data: {group_data}")

        group_weight = len(group_data) / total_samples
        group_entropy = entropy(group_data)
        after_split_entropy += group_weight * group_entropy
        
        # 可以打印查看每个子集的情况
        print(f"  特征值 '{group_name}': 权重={group_weight:.2f}, 熵={group_entropy:.3f}")
    
    # 4. 信息增益 = 混乱减少量
    info_gain = before_split_entropy - after_split_entropy
    
    print(f"特征 '{feature}': 分裂前熵={before_split_entropy:.3f}, "
          f"分裂后熵={after_split_entropy:.3f}, 信息增益={info_gain:.3f}")
    
    return info_gain

# =========================================================
# 5. 选择最优特征
# =========================================================
def best_feature(data, feature_list):
    best_f = None
    best_ig = -1

    for f in feature_list:
        ig = information_gain(data, f)
        if ig > best_ig:
            best_ig = ig
            best_f = f

    return best_f, best_ig


# =========================================================
# 6. 判断是否纯节点
# =========================================================
def all_same_label(data):
    labels = [row["label"] for row in data]
    return len(set(labels)) == 1


# =========================================================
# 7. 返回多数类
# =========================================================
def majority_label(data):
    labels = [row["label"] for row in data]
    return Counter(labels).most_common(1)[0][0]


# =========================================================
# 8. 手写构建决策树（ID3风格）
# =========================================================
def build_tree(data, feature_list):
    # 情况1：全属于同一类
    if all_same_label(data):
        return {"label": data[0]["label"]}

    # 情况2：没有特征可分了
    if len(feature_list) == 0:
        return {"label": majority_label(data)}

    # 选信息增益最大的特征
    best_f, best_ig = best_feature(data, feature_list)

    tree = {
        "feature": best_f,
        "info_gain": best_ig,
        "children": {},
        "majority_label": majority_label(data)
    }

    groups = split_dataset(data, best_f)
    remaining_features = [f for f in feature_list if f != best_f]

    for feature_value, subset in groups.items():
        tree["children"][feature_value] = build_tree(subset, remaining_features)

    return tree


# =========================================================
# 9. 打印树结构
# =========================================================
def print_tree(tree, indent=""):
    if "label" in tree:
        print(indent + "-> Predict:", tree["label"])
        return

    print(indent + f"[Feature: {tree['feature']}, Info Gain: {tree['info_gain']:.4f}]")
    for value, subtree in tree["children"].items():
        print(indent + f"  If {tree['feature']} == {value}:")
        print_tree(subtree, indent + "    ")


# =========================================================
# 10. 用树做预测
# =========================================================
def predict(tree, sample):
    """
    如果遇到训练中没见过的特征值，就返回该节点多数类
    """
    if "label" in tree:
        return tree["label"]

    feature = tree["feature"]
    value = sample.get(feature)

    if value in tree["children"]:
        return predict(tree["children"][value], sample)
    else:
        return tree["majority_label"]


# =========================================================
# 11. 计算所有特征的信息增益
# =========================================================
print("===== 数据集整体熵 =====")
base_ent = entropy(dataset)
print(f"Entropy(S) = {base_ent:.4f}\n")

print("===== 各特征的信息增益 =====")
ig_values = {}
for f in features:
    ig = information_gain(dataset, f)
    ig_values[f] = ig
    print(f"{f:12s} -> Information Gain = {ig:.4f}")

# =========================================================
# 12. 画信息增益柱状图
# =========================================================
plt.figure(figsize=(8, 5))
plt.bar(ig_values.keys(), ig_values.values())
plt.ylabel("Information Gain")
plt.title("Information Gain of Each Feature (Cat vs Dog)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# =========================================================
# 13. 构建决策树
# =========================================================
print("\n===== 构建出来的决策树 =====")
tree = build_tree(dataset, features)
print_tree(tree)

# =========================================================
# 14. 测试一个新动物
# =========================================================
new_animal = {
    "ear_shape": "pointy",
    "sound": "meow",
    "tail": "long",
    "likes_water": "no"
}

result = predict(tree, new_animal)

print("\n===== 新动物预测 =====")
print("新动物特征：", new_animal)
print("预测结果：", result)