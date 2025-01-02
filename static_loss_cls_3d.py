import torch

# 假设 sample_features 和 features 是之前定义的张量
sample_features = torch.tensor([[1, 2, 3], [4, 5, 6]])
features = torch.tensor([[1, 2, 3], [7, 8, 9], [9, 8, 7], [6, 5, 4]])

# 确保张量是浮点数类型
sample_features = sample_features.float()
features = features.float()

# 计算 sample_features 和 features 之间的成对距离
dists = torch.cdist(sample_features, features)

# 获取每个样本点的 top-3 最近邻居的索引
_, neighbor_indices_tensor = dists.topk(3, largest=False)  # Get top-k smallest distances

print(neighbor_indices_tensor)

'''
`torch.cdist` 函数计算两个点集之间的成对距离，返回一个矩阵，
其中每个元素 `dists[i, j]` 表示 `sample_features` 中第 `i` 个点和 `features` 中第 `j` 个点之间的欧氏距离。
在你的例子中，`sample_features` 有 2 个点，`features` 有 4 个点，
所以 `dists` 将是一个 `2x4` 的矩阵。`dists` 的每一行代表 `sample_features` 中
一个点到 `features` 中所有点的距离。
`dists.topk(3, largest=False)` 函数返回 `dists` 矩阵中每一行的最小的 3 个距离值的索引。
`largest=False` 参数表示我们需要的是最小的距离值，而不是最大的。
让我们逐步计算 `dists` 矩阵和 `topk` 的结果：
1. 计算 `sample_features` 中第一个点 `[1, 2, 3]` 到 `features` 中每个点的距离：
   - 到 `[1, 2, 3]` 的距离是 `0`（因为它们是同一个点）。
   - 到 `[7, 8, 9]` 的距离是 `sqrt((7-1)^2 + (8-2)^2 + (9-3)^2) = sqrt(36 + 36 + 36) = sqrt(108)`。
   - 到 `[9, 8, 7]` 的距离是 `sqrt((9-1)^2 + (8-2)^2 + (7-3)^2) = sqrt(64 + 36 + 16) = sqrt(116)`。
   - 到 `[6, 5, 4]` 的距离是 `sqrt((6-1)^2 + (5-2)^2 + (4-3)^2) = sqrt(25 + 9 + 1) = sqrt(35)`。
2. 计算 `sample_features` 中第二个点 `[4, 5, 6]` 到 `features` 中每个点的距离：
   - 到 `[1, 2, 3]` 的距离是 `sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(9 + 9 + 9) = sqrt(27)`。
   - 到 `[7, 8, 9]` 的距离是 `sqrt((4-7)^2 + (5-8)^2 + (6-9)^2) = sqrt(9 + 9 + 9) = sqrt(27)`。
   - 到 `[9, 8, 7]` 的距离是 `sqrt((4-9)^2 + (5-8)^2 + (6-7)^2) = sqrt(25 + 9 + 1) = sqrt(35)`。
   - 到 `[6, 5, 4]` 的距离是 `0`（因为它们是同一个点）。
所以，`dists` 矩阵大致如下（这里只给出了距离的近似值）：
```
[[0,  sqrt(108),  sqrt(116),  sqrt(35)],
 [sqrt(27),  sqrt(27),  sqrt(35),  0]]
```
3. 对每一行应用 `topk(3, largest=False)`，我们得到每个点的 3 个最近邻居的索引：
   - 对于第一个点，最近的 3 个邻居是 `[0, 3, 1]`（索引对应于 `features` 中的点）。
   - 对于第二个点，最近的 3 个邻居是 `[3, 0, 1]`。
因此，输出的 `neighbor_indices_tensor` 是：
```
tensor([[0, 3, 1],
        [3, 0, 1]])
```
这就是 `topk` 函数计算出来的每个点的 3 个最近邻居的索引。

'''