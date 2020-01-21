import torch
import time
import mmcv
from scipy.spatial import Delaunay
import numpy as np


def knn(ref, que, k):
    ref = ref[None,:,:]
    que = que[:,None]
    dist = que-ref
    dist,_ = dist.abs().max(dim=-1)
    # dist_sort, index_sort = dist.sort(dim=-1)
    # dist_sort, index_sort = torch.topk(dist, k, dim=-1, largest=False,)
    # return dist_sort[:,:k], index_sort[:,:k]

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


def gen_s_grid(feat_h, feat_w):
    shift_x = torch.arange(0, feat_w)
    shift_y = torch.arange(0, feat_h)
    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy], dim=-1).float()
    return shifts


def gen_d_grid(feat_h, feat_w, r_h, r_w):
    out_shift_x = torch.arange(0, r_w)
    out_shift_y = torch.arange(0, r_h)
    out_shift_xx, out_shift_yy = _meshgrid(out_shift_x, out_shift_y)
    out_shift_xx = ((out_shift_xx.float() + 0.5) * (feat_w) / (r_w) - 0.5).clamp(0, feat_w - 1)
    out_shift_yy = ((out_shift_yy.float() + 0.5) * (feat_h) / (r_h) - 0.5).clamp(0, feat_h - 1)
    out_shifts = torch.stack([out_shift_xx, out_shift_yy], dim=-1)
    return out_shifts


def get_score(i, d_shifts, s_shifts,  s_array, _s_array, dist, index):
    x, y = d_shifts[i].numpy()
    s_h, s_w = s_array.shape
    if x in list(range(s_w)) and y in list(range(s_h)):
        _index = d_shifts[i]
        cindex = _index.unsqueeze(dim=0).long().numpy()[:, ::-1].transpose(1, 0)
        score = s_array[cindex].squeeze()
    elif x in list(range(s_w)) or y in list(range(s_h)):
        w1, w2 = dist[i][:2]
        assert w1 + w2 == 1
        v1, v2 = _s_array[index[i][:2]].squeeze()
        score = w1 * v2 + w2 * v1
    else:
        _index = s_shifts[index[i]]
        _values = _s_array[index[i]]

        mus = (_index[:, 0] * _index[:, 1]).reshape(-1, 1)
        ones = torch.ones(4, 1)
        mu_matrix = torch.cat([ones, _index, mus], dim=-1)

        try:
            a_matrix = torch.matmul(mu_matrix.inverse(), _values.reshape(-1, 1))
        except:
            print(mu_matrix)
        real_index = d_shifts[i].reshape(-1)
        mu = (real_index[0] * real_index[1]).reshape(-1)
        one = torch.ones(1).reshape(-1)
        real_matrix = torch.cat([one, real_index, mu], dim=0)
        score = torch.matmul(a_matrix.squeeze(), real_matrix)
    return {'score':score}


def tri(ref, que, k):
    ref = ref.numpy()
    que = que.numpy()

    tri = Delaunay(ref)
    index_tri = tri.find_simplex(que)
    _three_point = ref[tri.simplices[index_tri]]
    index_nei = tri.neighbors[index_tri]
    index_nei[:, 0][(index_nei[:, 0] == -1)] = index_nei[:, 2][(index_nei[:, 0] == -1)]
    index_nei[:, 1][(index_nei[:, 1] == -1)] = index_nei[:, 2][(index_nei[:, 1] == -1)]
    index_nei[:, 2][(index_nei[:, 2] == -1)] = index_nei[:, 0][(index_nei[:, 2] == -1)]

    nei_points = ref[tri.simplices[index_nei]]
    _nei_points = nei_points.reshape(nei_points.shape[0], -1, 2)

    re_three_point = _three_point[:,-2:] - _three_point[:, [0]]
    ori_area = np.abs(np.linalg.det(re_three_point))

    re_nei_points = _nei_points[:,:,np.newaxis,:] - _three_point[:,np.newaxis,:,:]
    tile_re_nei_points = np.tile(re_nei_points, (1,1,2,1))
    tile_re_nei_points = tile_re_nei_points.reshape(tile_re_nei_points.shape[0], -1, 2, 2)
    det_value = np.abs(np.linalg.det(tile_re_nei_points))
    new_area = det_value.reshape(det_value.shape[0], -1, 3).sum(axis=-1)
    filter_pts = _nei_points[np.abs(new_area -ori_area[:,np.newaxis])>1e-4].reshape(_nei_points.shape[0], 3, 2)
    _four_indexs = tri.simplices[index_nei].reshape(-1,9)[np.abs(new_area -ori_area[:,np.newaxis])>1e-4]\
        .reshape(_nei_points.shape[0], -1)

    _four_pt = np.argmin(np.abs((que[:, np.newaxis, :] - filter_pts)).max(axis=-1), axis=-1)
    len_list = np.arange(0, _four_indexs.shape[0], 1)
    _four_index = _four_indexs[len_list, _four_pt]
    index_list = np.concatenate([tri.simplices[index_tri], _four_index[:, np.newaxis]], axis=-1)

    # index_list  = np.argsort(ref[index_list]-que[:,np.newaxis,:])
    dist = np.sort(np.abs(ref[index_list]-que[:,np.newaxis,:]).max(axis=-1))

    # import IPython
    # IPython.embed()

    # import matplotlib.pyplot as plt
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.show()
    return torch.tensor(dist).float() , torch.tensor(index_list).long()


def tri_add(ref, que, k):
    ref = ref.numpy()
    que = que.numpy()

    tri = Delaunay(ref)
    import matplotlib.pyplot as plt
    # plt.triplot(ref[:, 0], ref[:, 1], tri.simplices.copy())
    # plt.plot(ref[:, 0], ref[:, 1], 'o')
    # plt.show()
    index_tri = tri.find_simplex(que)
    _three_point = ref[tri.simplices[index_tri]]

    reque = (que - tri.transform[index_tri][:, 2])
    renen = tri.transform[index_tri][:, :2]
    weight2 = np.matmul(renen, reque[:, :, np.newaxis]).squeeze()
    weight1 = 1-weight2.sum(axis=-1)
    weight = np.concatenate([weight2, weight1[:, np.newaxis]], axis=-1)

    return torch.tensor(weight).float(), torch.tensor(tri.simplices[index_tri]).long()


def tri_add_v2(ref, que, k):
    # tri_start_t = time.time()
    # all_start_t = tri_start_t
    tri_result = Delaunay(ref)

    index_tri = tri_result.find_simplex(que)
    simplices_index = tri_result.simplices[index_tri]
    _three_point = ref[simplices_index]

    transform_tri = tri_result.transform[index_tri]
    # tri_end_t = time.time()

    reque = (que - transform_tri[:, 2])
    renen = transform_tri[:, :2]
    weight2 = np.matmul(renen, reque[:, :, np.newaxis]).squeeze()
    weight1 = 1-weight2.sum(axis=-1)
    weight = np.concatenate([weight2, weight1[:, np.newaxis]], axis=-1)
    # all_end_t = time.time()

    # print('Delaunay time single gt:{:.5f}'.format(tri_end_t - tri_start_t))
    # print('All time single gt:{:.5f}'.format(all_end_t - all_start_t))

    return weight, simplices_index


def interplate_v6(corner_s_array, im_pts, im_pts_score, output_size, use_tri=True, s_shape=(28, 28)):
    assert use_tri, 'only support use_tri'
    if isinstance(corner_s_array, torch.Tensor):
        corner_s_array = corner_s_array.numpy()
    if isinstance(im_pts, torch.Tensor):
        im_pts = im_pts.numpy()
    if isinstance(im_pts_score, torch.Tensor):
        im_pts_score = im_pts_score.numpy()

    # start_t = time.time()
    assert corner_s_array.shape == (2, 2)
    s_h, s_w = s_shape
    r_h, r_w = output_size

    d_shifts = gen_d_grid(s_h, s_w, r_h, r_w).numpy()

    corner_s_shifts = np.array([[0, 0], [27, 0], [0, 27], [27, 27]], dtype=im_pts.dtype)
    corner_idxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.long)
    corner_s_array = corner_s_array[np.transpose(corner_idxs)].reshape(-1, 1)

    s_shifts = np.concatenate([corner_s_shifts, im_pts], axis=0)
    s_array = np.concatenate([corner_s_array, im_pts_score[:, np.newaxis]], axis=0)

    try:
        dist, index = tri_add_v2(s_shifts, d_shifts, 4)
    except:
        return mmcv.imresize(s_array, (r_w, r_h))
    try:
        _index = s_shifts[index]
        _values = s_array[index]
    except Exception as e:
        raise e
    scores = (dist * _values.squeeze()).sum(axis=-1)
    scores = scores.reshape(r_h, r_w)

    # end_t = time.time()
    # print('all time single gt:{:.5f}'.format(end_t - start_t))
    # print()
    return scores


def interplate_v5(s_array, im_pts, im_pts_score, output_size, use_tri=True, s_shape=(28, 28)):
    assert s_array.shape == (2, 2)
    s_h, s_w = s_shape
    r_h, r_w = output_size

    d_shifts = gen_d_grid(s_h, s_w, r_h, r_w)

    s_shifts = torch.Tensor([[0, 0], [27, 0], [0, 27], [27, 27]]).float()
    corner_idxs = torch.Tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).float()
    _s_array = s_array[corner_idxs.numpy().transpose(1, 0)].reshape(-1, 1)

    # s_shifts = gen_s_grid(s_h, s_w)
    # darray = torch.zeros(r_w*r_h)
    # _s_array = s_array.reshape(-1,1)

    s_shifts = torch.cat([s_shifts, im_pts], dim=0)
    _s_array = torch.cat([_s_array, im_pts_score.unsqueeze(-1)], dim=0)

    # s_shifts = im_pts
    # _s_array = im_pts_score.unsqueeze(-1)
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


def interplate_v4(s_array, im_pts, im_pts_score, output_size, use_tri=True):

    s_h, s_w = s_array.shape
    r_h, r_w = output_size

    d_shifts = gen_d_grid(s_h, s_w, r_h, r_w)

    s_shifts = torch.tensor([[0,0], [27,0], [0, 27], [27, 27]]).float()
    _s_array = s_array[s_shifts.numpy().transpose(1, 0)].reshape(-1,1)

    # s_shifts = gen_s_grid(s_h, s_w)
    # darray = torch.zeros(r_w*r_h)
    # _s_array = s_array.reshape(-1,1)

    s_shifts = torch.cat([s_shifts, im_pts], dim=0)
    _s_array = torch.cat([_s_array, im_pts_score.unsqueeze(-1)], dim=0)

    # s_shifts = im_pts
    # _s_array = im_pts_score.unsqueeze(-1)
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
    except:
        import ipdb
        ipdb.set_trace()
    scores = (dist.numpy() * _values.squeeze().numpy()).sum(axis=-1)
    scores = scores.reshape(r_h, r_w)
    scores = torch.tensor(scores)
    return scores

