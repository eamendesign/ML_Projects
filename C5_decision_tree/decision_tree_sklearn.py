import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 1) 读取真实数据集
iris = load_iris()
X = iris.data
y = iris.target

# 2) 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3) 训练决策树
clf = DecisionTreeClassifier(
    criterion="entropy",   # 用信息熵
    max_depth=3,
    random_state=42
)
clf.fit(X_train, y_train)

# 4) 预测
y_pred = clf.predict(X_test)

# 5) 评估
acc = accuracy_score(y_test, y_pred)
print("Accuracy =", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6) 画树
plt.figure(figsize=(10, 5))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.title("Decision Tree on Iris Dataset")
plt.show()