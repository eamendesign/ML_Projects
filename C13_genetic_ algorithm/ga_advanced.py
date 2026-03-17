import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ---------------------- 定义目标函数 ----------------------
x = np.linspace(-15, 15, 300)
y = np.piecewise(
    x,
    [x <= -5, (-5 < x) & (x < 5), (5 <= x) & (x < 12), x >= 12],
    [
        lambda x: 0.2*(x + 10)**2 + 8,
        lambda x: 0.5*(x)**2 - 5,
        lambda x: 0.1*(x - 8)**2 + 4,
        lambda x: 0.5*(x - 12) + 8
    ]
)

def y_cal(x):
    """根据分段函数计算y值"""
    if x <= -5:
        return 0.2*(x + 10)**2 + 8
    elif x < 5:
        return 0.5*(x)**2 - 5
    elif x < 12:
        return 0.1*(x - 8)**2 + 4
    else:
        return 0.5*(x - 12) + 8

# ---------------------- 遗传算法参数 ----------------------
individuals = 10      # 种群规模
generations = 10      # 迭代代数
survive_ind = 4       # 每代保留的最优个体数
mutation_rate = 0.1   # 变异概率

# ---------------------- 初始化种群 ----------------------
x_pos = [30 * random.random() - 15 for _ in range(individuals)]
all_generations = [x_pos.copy()]  # 用于保存每代位置，用于动画

# ---------------------- 遗传算法迭代 ----------------------
for g in range(generations):
    # 计算每个个体的目标函数值
    x_pos_y = [y_cal(xi) for xi in x_pos]
    # 为简便直接将目标函数值进行排序（升序），而不是适应度函数（降序）
    idx = np.argsort(x_pos_y)
    # 保留前survive_ind个个体
    x_pos = [x_pos[i] for i in idx][:survive_ind]

    # 交叉操作：生成新个体填充种群
    while len(x_pos) < individuals:
        father = random.choice(x_pos[:survive_ind])
        mother = random.choice(x_pos[:survive_ind])
        # 简单加权平均生成新个体
        alpha = random.random()
        new_x = alpha * father + (1 - alpha) * mother
        x_pos.append(new_x)

    # 变异操作
    for i in range(survive_ind, individuals):
        if random.random() < mutation_rate:
            x_pos[i] += random.random() - 0.5  # ±0.5的小幅随机变异

    # 保存当前代个体位置
    all_generations.append(x_pos.copy())

# ---------------------- 可视化与动画 ----------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, color="blue", linewidth=2, label="目标函数")
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_xlim([-15, 15])
ax.set_ylim([-10, 25])
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(fontsize=12)

# 动态点初始化
points, = ax.plot([], [], 'ro', markersize=6)

def update(frame):
    """更新动画帧"""
    gen_x = all_generations[frame]
    gen_y = [y_cal(xi) for xi in gen_x]
    points.set_data(gen_x, gen_y)
    ax.set_title(f"遗传算法迭代: 第 {frame+1} 代", fontsize=16)
    return points,

ani = FuncAnimation(fig, update, frames=len(all_generations), interval=300, blit=False)

# 保存动画
# ani.save('遗传算法动画.mp4', writer='ffmpeg', fps=5)

plt.show()