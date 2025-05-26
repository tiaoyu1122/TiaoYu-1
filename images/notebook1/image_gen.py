import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置 matplotlib 支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def gen_image_1():
    # 定义数据
    x = [2, 5, 8, 10, 12, 15, 18, 20, 22, 25]
    y = [200, 212, 225, 238, 241, 257, 269, 275, 278, 285]

    # 绘制散点图
    plt.scatter(x, y)

    # 进行线性回归
    coefficients = np.polyfit(x, y, 1)
    poly = np.poly1d(coefficients)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    # 打印线性回归公式到图片中
    formula = f'$Y = {coefficients[0]:.2f}X + {coefficients[1]:.2f}$'
    plt.text(x_fit[55], y_fit[50], formula, fontsize=12, color='red')

    # 绘制预测直线
    plt.plot(x_fit, y_fit, color='red')

    # 设置X轴和Y轴的标题，使用指定字体
    plt.xlabel('Y')
    plt.ylabel('X')
    # 设置X轴范围为0-45
    plt.xlim(0, 45)

    # 保存图片为png格式
    plt.savefig('images/notebook1/image_1.png')

    # 显示图形
    plt.show()


# 新增函数
def gen_image_2():
    # 定义数据
    x = [2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 27, 30, 32, 35, 37, 40]
    y = [200, 212, 225, 238, 241, 257, 269, 275, 278, 285, 281, 277, 270, 260, 255, 245]

    # 绘制散点图
    plt.scatter(x, y)

    # 进行二次曲线回归
    coefficients = np.polyfit(x, y, 2)
    poly = np.poly1d(coefficients)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    # 打印二次曲线回归公式到图片中
    formula = f'$Y = {coefficients[0]:.2f}X^2 + {coefficients[1]:.2f}X + {coefficients[2]:.2f}$'
    plt.text(x_fit[30], y_fit[20], formula, fontsize=12, color='red')

    # 绘制预测曲线
    plt.plot(x_fit, y_fit, color='red')

    # 设置X轴和Y轴的标题，使用指定字体
    plt.xlabel('Y')
    plt.ylabel('X')
    # 设置X轴范围为0-45
    plt.xlim(0, 45)

    # 保存图片为png格式
    plt.savefig('images/notebook1/image_2.png')

    # 显示图形
    plt.show()


# 新增函数，使用梯度下降方法求解参数 w 和 b
def gradient_descent():
    # 初始化参数
    w = 0
    b = 0
    # 学习率
    learning_rate = 0.001
    # 最大迭代次数
    max_iterations = 10000
    # 迭代次数
    iteration = 0

    while iteration <= max_iterations:
        # 计算损失函数
        loss = (200 - 2 * w - b) ** 2 + (212 - 5 * w - b) ** 2 + (225 - 8 * w - b) ** 2

        # 计算梯度
        dw = -2 * (2 * (200 - 2 * w - b) + 5 * (212 - 5 * w - b) + 8 * (225 - 8 * w - b))
        db = -2 * ((200 - 2 * w - b) + (212 - 5 * w - b) + (225 - 8 * w - b))

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #更新后的损失
        loss_f = (200 - 2 * w - b) ** 2 + (212 - 5 * w - b) ** 2 + (225 - 8 * w - b) ** 2
        # 每隔 1000 步输出一次
        if (iteration % 1000 == 0) or (iteration < 10) or (iteration == 999) or (iteration >= 9999):
            print(f'Iteration {iteration}, Loss: {loss:.3f}, Gradient dw: {dw:.3f}, db: {db:.3f}, w: {w:.3f}, b: {b:.3f}, Loss_f:{loss_f:.3f}')


        iteration += 1

    return w, b


if __name__ == '__main__':
    # gen_image_1()
    # gen_image_2()
    w, b = gradient_descent()
    print(f'Final w: {w}, Final b: {b}')
