import numpy as np
import matplotlib.pyplot as plt

# 生成数据
n0 = 50
X0 = np.random.randn(n0, 2)                    # 类别0，中心 (0,0)
X1 = np.random.randn(n0, 2) + np.array([3, 3]) # 类别1，中心 (3,3)

# 创建画布
plt.figure(figsize=(8, 6))

# 绘制类别0（蓝色圆点）
plt.scatter(X0[:, 0], X0[:, 1], 
            c='blue',           # 颜色
            marker='o',         # 形状：圆点
            label='Class 0',    # 图例标签
            alpha=0.7)          # 透明度

# 绘制类别1（红色三角）
plt.scatter(X1[:, 0], X1[:, 1], 
            c='red', 
            marker='^',         # 形状：三角形
            label='Class 1',
            alpha=0.7)

# 添加中心点标记（可选）
plt.scatter(0, 0, c='black', marker='x', s=100, label='Center 0')
plt.scatter(3, 3, c='black', marker='x', s=100, label='Center 1')

# 图表装饰
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Two Classes Generated from Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')  # 使坐标轴比例相等，更真实反映分布

# 显示图形
plt.show()