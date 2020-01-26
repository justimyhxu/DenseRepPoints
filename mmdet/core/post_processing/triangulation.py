import torch
import time
import mmcv
from scipy.spatial import Delaunay
import numpy as np


def knn(ref, que, k):
    ref = ref[None, :, :]
    que = que[:, None]
    dist = que - ref
    dist, _ = dist.abs().max(dim=-1)

    dist_list = []
    index_list = []

    for i in range(k):
        dist_sort, index_sort = torch.min(dist, dim=-1)
        rang = torch.arange(0, dist.shape[0])
        _i = torch.stack([rang, index_sort]).numpy()
        dist[_i] = torch.tensor(float('inf'))
        dist_list.append(dist_sort)
        index_list.append(index_sort)
    rdist = torch.stack(dist_list, dim=1)
    rindex = torch.stack(index_list, dim=1)
    return rdist, rindex


def _meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def gen_d_grid(feat_h, feat_w, r_h, r_w):
    out_shift_x = torch.arange(0, r_w)
    out_shift_y = torch.arange(0, r_h)
    out_shift_xx, out_shift_yy = _meshgrid(out_shift_x, out_shift_y)
    out_shift_xx = ((out_shift_xx.float() + 0.5) * (feat_w) / (r_w) - 0.5).clamp(0, feat_w - 1)
    out_shift_yy = ((out_shift_yy.float() + 0.5) * (feat_h) / (r_h) - 0.5).clamp(0, feat_h - 1)
    out_shifts = torch.stack([out_shift_xx, out_shift_yy], dim=-1)
    return out_shifts


def tri_add(ref, que, k):
    ref = ref.numpy()
    que = que.numpy()

    tri = Delaunay(ref)
    index_tri = tri.find_simplex(que)
    _three_point = ref[tri.simplices[index_tri]]

    reque = (que - tri.transform[index_tri][:, 2])
    renen = tri.transform[index_tri][:, :2]
    weight2 = np.matmul(renen, reque[:, :, np.newaxis]).squeeze()
    weight1 = 1 - weight2.sum(axis=-1)
    weight = np.concatenate([weight2, weight1[:, np.newaxis]], axis=-1)

    return torch.tensor(weight).float(), torch.tensor(tri.simplices[index_tri]).long()


def interplate(s_array, im_pts, im_pts_score, output_size, use_tri=True, s_shape=(28, 28)):
    assert s_array.shape == (2, 2)
    s_h, s_w = s_shape
    r_h, r_w = output_size

    d_shifts = gen_d_grid(s_h, s_w, r_h, r_w)

    s_shifts = torch.Tensor([[0, 0], [27, 0], [0, 27], [27, 27]]).float()
    corner_idxs = torch.Tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).float()
    _s_array = s_array[corner_idxs.numpy().transpose(1, 0)].reshape(-1, 1)

    s_shifts = torch.cat([s_shifts, im_pts], dim=0)
    _s_array = torch.cat([_s_array, im_pts_score.unsqueeze(-1)], dim=0)

    try:
        if use_tri:
            dist, index = tri_add(s_shifts, d_shifts, 4)
        else:
            dist, index = knn(s_shifts, d_shifts, 4)
    except:
        import mmcv
        return torch.tensor(mmcv.imresize(s_array.numpy(), (r_w, r_h)))
    try:
        _index = s_shifts[index]
        _values = _s_array[index]
    except Exception as e:
        raise e
    scores = (dist.numpy() * _values.squeeze().numpy()).sum(axis=-1)
    scores = scores.reshape(r_h, r_w)
    scores = torch.tensor(scores)
    return scores
