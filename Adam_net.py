import torch
'''
    使用PyTorch的优化器来更新一个张量x的值，使其预测值y_pre = 5 * x接近于某个目标值y_label
'''
# 假设a, b, c, d是已知的常数参数
a = 5.0
b = 2.0
c = 1.5
d = 0.5
x = torch.tensor([2.0], device='cuda', requires_grad=True)  # 类比于高斯椭球属性
# 定义优化器，直接传递x作为参数
optimizer = torch.optim.Adam([x], lr=0.01, eps=1e-15)

y_label = torch.tensor([6.0], device='cuda')  # 相当于真实图片标签


for i in range(1, 100000):
    '''
    这段代码的目的是使用PyTorch的优化器来调整张量x的值
    使得通过某个线性关系（这里是y_pre = 5 * x）得到的预测值y_pre尽可能接近于目标值y_label。
    这个过程是一个优化问题，其中x是优化变量，损失函数是预测值和目标值之间的差异。
    这里的y_pre = 5 * x并不是固定的，因为我们希望通过调整x的值来最小化损失函数。
    虽然数学上可以直接解出x = y_label / 5，但在机器学习中，我们通常处理的是更复杂的问题，
    其中目标函数可能不是简单的线性关系，而且可能有多个变量需要同时优化。===>参考3DGS建立的数学模型
    '''
    # y_pre = 5 * x  # render()方法进行渲染 三维椭球到二维图片的方法
    y_pre = a * torch.exp(b * x) + c * x ** d + torch.log(x + 1)
    # 计算损失，这里使用MSE损失
    loss = (y_pre - y_label) ** 2  # 计算损失

    loss.backward()  # 反向传播

    with torch.no_grad():
        optimizer.step()  # 优化器步骤:step()方法应用梯度下降（或其他优化算法），根据计算得到的梯度更新模型的参数
        optimizer.zero_grad()  # 清零梯度，为下一次迭代做准备

    # 打印中间结果
    if i % 2 == 0:  # 每两步打印一次结果
        print(f"Iteration {i}: x = {x.item()}, loss = {loss.item()}")
