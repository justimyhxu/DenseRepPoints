import torch
import torch.nn as nn
import numpy as np


# Adapted from https://github.com/dfdazac/wassdistance
class ChamferDistance(nn.Module):
    r"""
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, reduction='none'):
        super(ChamferDistance, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        if x.shape[0] == 0:
            return x.sum()
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points, y_points = x.shape[-2], y.shape[-2]

        # compute chamfer loss
        min_x2y, _ = C.min(-1)
        d1 = min_x2y.mean(-1)
        min_y2x, _ = C.min(-2)
        d2 = min_y2x.mean(-1)
        cost = (d1 + d2) / 2.0

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        # C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        C = torch.norm(torch.abs(x_col - y_lin), 2, -1)
        return C

def chamfer_loss(point_set_1, point_set_2):
    '''Computation of optimal transport distance via sinkhorn algorithm.
    - Input:
        - point_set_1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
        - point_set_2:	torch.Tensor	[..., num_points_2, point_dim]
        		(the dimensions of point_set_2 except the last two should be the same as point_set_1)
    - Output:
        - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
    '''
    bs = point_set_1.reshape((-1, point_set_1.shape[-2], point_set_1.shape[-1])).shape[0]
    x_points, y_points = point_set_1.shape[-2], point_set_2.shape[-2]

    chamfer = ChamferDistance(reduction=None)

    assert point_set_1.dim() == point_set_2.dim()
    assert point_set_1.shape[-1] == point_set_2.shape[-1]
    if point_set_1.dim() <= 3:
        dist = chamfer(point_set_1, point_set_2)
    else:
        point_dim = point_set_1.shape[-1]
        num_points_1, num_points_2 = point_set_1.shape[-2], point_set_2.shape[-2]
        point_set_1t = point_set_1.reshape((-1, num_points_1, point_dim))
        point_set_2t = point_set_2.reshape((-1, num_points_2, point_dim))
        dist_t = chamfer(point_set_1t, point_set_2t)
        dist_dim = point_set_1.shape[:-2]
        dist = dist_t.reshape(dist_dim)
    return dist


if __name__ == '__main__':
    n_points_1, n_points_2 = 50, 100
    a = np.array([[np.random.rand(), 0] for i in range(n_points_1)])
    b = np.array([[np.random.rand(), 10] for i in range(n_points_2)])
    x = torch.tensor([[a], [a]], dtype=torch.float).squeeze(1).cuda(3)
    y = torch.tensor([[b], [b]], dtype=torch.float).squeeze(1).cuda(3)

    # a = np.array([[np.random.rand(), np.random.rand(), 0] for i in range(n_points_1)])
    # b = np.array([[np.random.rand(), np.random.rand(), 10] for i in range(n_points_2)])
    # x = torch.tensor([[a,a,a], [b,b,a]], dtype=torch.float).squeeze(1)
    # y = torch.tensor([[a,b,a], [b,b,b]], dtype=torch.float).squeeze(1)
    print(x.shape)
    print(y.shape)

    # sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    # dist, P, C = sinkhorn(x, y)
    import time

    start = time.time()
    from tqdm import tqdm
    for i in tqdm(range(100)):
        dist = chamfer_loss(x, y)
    end = time.time()
    print((end - start) / (100.0 * 2.0))
    print(dist.cpu().numpy())
    # import pdb
    # pdb.set_trace()

