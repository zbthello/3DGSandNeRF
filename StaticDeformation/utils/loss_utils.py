
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

# ##
def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()
# ##

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ##
def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.

    计算 3D 点云的邻域一致性损失，使用 Top-k 邻居和 KL 散度。

    :param features: 形状为 (N, D) 的张量，其中 N 是点的数量，D 是特征的维度。
    :param predictions: 形状为 (N, C) 的张量，其中 C 是类别的数量。
    :param k: 要考虑的邻居数量。
    :param lambda_val: 损失的加权因子。
    :param max_points: 点的最大数量用于下采样。如果点的数量超过这个值，它们将被随机下采样。
    :param sample_size: 随机采样用于计算损失的点的数量。

    :return: 计算出的损失值。
    """
    # Conditionally downsample if points exceed max_points
    # 如果点的数量超过了 max_points，则随机选择 max_points 个点及其对应的预测结果。
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]


    # Randomly sample points for which we'll compute the loss
    # 从所有点中随机选择 sample_size 个点及其对应的预测结果，用于计算损失。
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    # 使用 torch.cdist 计算 sample_features 和所有点之间的成对距离，然后使用 topk 方法找到每个点的 k 个最近邻居的索引。
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    # 根据最近邻居的索引，从 predictions 中获取这些邻居的预测结果。
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    # 计算每个点的预测概率与邻居预测概率之间的 KL 散度。
    # 这里使用了 unsqueeze 来增加一个维度，以便进行广播运算。
    # 1e-10 是为了防止对数运算中的数值不稳定。最后，对 KL 散度求和并计算平均值，得到损失。sample_preds是预测的类别，neighbor_preds是邻居点预测的类别
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    # 将损失归一化到 [0, 1] 范围内，方法是将损失除以类别数量 num_classes。
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss
# ##



