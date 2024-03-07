import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

# 示例数据点
# data_points = [
#     (0.55, 0.73, 0.66),
#     (0.30, 0.48, 1.34),
#     (0.37, 0.52, 0.23),
#     (0.8, 0.7, 1.21),
#     (0.39, 0.68, 1.55),
#     (0.83, 0.57, 1.85),
#     (0.56, 0.51, 1.36),
#     (0.64, 0.54, 1.39),
#     (0.63, 0.72, 1.19),
#     (0.39, 0.54, 0.10)
# ]
data_points = [
    (0.55, 0.73, 0.66),
    (0.30, 0.48, 1.34),
    (0.37, 0.52, 0.23),
    (0.8, 0.7, 1.21),
    (0.39, 0.68, 1.55),
#     (0.83, 0.57, 1.85),
    (0.56, 0.51, 1.36),
#     (0.64, 0.54, 1.39),
#     (0.63, 0.72, 1.19),
    (0.39, 0.54, 0.10)
]

# 分解数据点
friction = [point[0] for point in data_points]
restitution = [point[1] for point in data_points]
loss = [point[2] for point in data_points]

# 创建3D图表
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')


###################################################################
# 创建网格数据
xi = np.linspace(0.3, 0.8, 200)
yi = np.linspace(0.5, 0.8, 200)
zi = griddata((friction, restitution), loss, (xi[None, :], yi[:, None]), method='cubic')

# 绘制曲面
xig, yig = np.meshgrid(xi, yi)
surf = ax.plot_surface(xig, yig, zi, cmap=cm.jet, alpha=0.5, linewidth=0, antialiased=True)

###################################################################
# 绘制原始数据点，使用 'x' 形状
ax.scatter(friction, restitution, loss, color="black", marker='x', s=50)

# 特别标出起点
ax.scatter(friction[0], restitution[0], loss[0], color="black", marker='x', s=200)  # 用绿色标出起点

# 绘制连接点的线
for i in range(len(friction)-1):
    ax.plot([friction[i], friction[i+1]], [restitution[i], restitution[i+1]], [loss[i], loss[i+1]], color="red")

# 在最后两个点之间添加箭头
x_arrow = np.linspace(friction[-2], friction[-1], num=2)
y_arrow = np.linspace(restitution[-2], restitution[-1], num=2)
z_arrow = np.linspace(loss[-2], loss[-1], num=2)
ax.quiver(x_arrow[0], y_arrow[0], z_arrow[0], x_arrow[1]-x_arrow[0], y_arrow[1]-y_arrow[0], z_arrow[1]-z_arrow[0], 
          color="red", arrow_length_ratio=0.15)


###################################################################
# 设置坐标轴标签和字体大小
ax.set_xlabel('Friction', fontsize=18)
ax.set_ylabel('Restitution', fontsize=18)
# 设置 Z 轴标签为竖直方向
ax.set_zlabel('Loss', fontsize=18, rotation=90)

# 设置坐标轴刻度
ax.set_xticks([0.3, 0.5, 0.7])
ax.set_yticks([0.5, 0.7])
ax.set_zticks([0.0, 0.5, 1.0, 1.5])


# Adjust the view angle
ax.view_init(elev=30)  # elev = elevation angle, azim = azimuth angle


# 设置坐标轴刻度的字体大小
ax.tick_params(axis='x', labelsize=16)  # X轴刻度字体大小
ax.tick_params(axis='y', labelsize=16)  # Y轴刻度字体大小
ax.tick_params(axis='z', labelsize=16)  # Z轴刻度字体大小

# 调整刻度标签与坐标轴的距离
ax.tick_params(axis='x', pad=1)  # X轴刻度标签距离
ax.tick_params(axis='y', pad=1)  # Y轴刻度标签距离
ax.tick_params(axis='z', pad=1)  # Z轴刻度标签距离

# 显示图表
plt.show()

