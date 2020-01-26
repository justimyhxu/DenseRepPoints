import torch
import torch.nn as nn


class ChamferDistance(nn.Module):
    def __init__(self, reduction='none'):
        super(ChamferDistance, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        if x.shape[0] == 0:
            return x.sum()
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.norm(torch.abs(x_col - y_lin), 2, -1)
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


def chamfer_loss(point_set_1, point_set_2):
    chamfer = ChamferDistance(reduction='none')

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
