import torch

import numpy as np
import mmcv
import cv2

def extreme_points(index):
    inds_x, inds_y = index[:, 0], index[:, 1]
    x_min_id = inds_x.argmin(0)
    left = index[x_min_id]
    x_max_id = inds_x.argmax(0)
    right = index[x_max_id]
    y_min_id = inds_y.argmin(0)
    up = index[y_min_id]
    y_max_id = inds_y.argmax(0)
    bottom = index[y_max_id]
    extremes = np.stack([left, right, up, bottom], 0)
    return extremes


def sample_foreground(gt_bboxes, gt_masks, cfg, num_pts,):
    pts_list = []
    pts_label_list = []
    for i in range(len(gt_bboxes)):
        if not hasattr(cfg, 'resize_sample') or (hasattr(cfg, 'resize_sample') and cfg.resize_sample):
            x1, y1, x2, y2 = gt_bboxes[i].cpu().numpy().astype(np.int32)
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            mask = mmcv.imresize(gt_masks[i][y1:y1 + h, x1:x1 + w],
                                 (cfg.mask_size, cfg.mask_size))
            index_y, index_x = np.nonzero(mask > 0)
            index = np.stack([index_x, index_y], axis=-1)
            _len = index.shape[0]
            if _len == 0:
                pts = np.zeros([2*num_pts])
            else:
                repeat = num_pts//_len
                mod = num_pts % _len
                perm = np.random.choice(_len, mod, replace=False)
                draw = [index.copy() for i in range(repeat)]
                draw.append(index[perm])
                draw = np.concatenate(draw, 0)
                draw = draw + np.random.rand(*draw.shape)
                x_scale = float(w) / cfg.mask_size
                y_scale = float(h) / cfg.mask_size
                draw[:, 0] = draw[:, 0] * x_scale + x1
                draw[:, 1] = draw[:, 1] * y_scale + y1
                pts = draw.reshape(2 * num_pts)
        else:
            index_y, index_x = np.nonzero(gt_masks[i] > 0)
            index = np.stack([index_x, index_y], axis=-1)
            _len = index.shape[0]
            if _len == 0:
                pts = np.zeros([2 * num_pts])
            else:
                perm = np.random.choice(_len, num_pts)
                pts = index[perm].reshape(2 * num_pts)

        pts_list.append(pts)
        pts_long = pts.astype(np.long)
        pts_label = gt_masks[i][pts_long[1::2], pts_long[0::2]]
        pts_label_list.append(pts_label)
    pts_list = np.stack(pts_list, 0)
    pts_label_list = np.stack(pts_label_list, 0)
    return pts_list, pts_label_list

def sample_uniform(gt_bboxes, gt_masks, cfg, num_pts):
    pts_list = []
    pts_label_list = []
    _len = int(np.sqrt(num_pts))
    assert _len**2 == num_pts
    for i in range(len(gt_bboxes)):
        x1, y1, x2, y2 = gt_bboxes[i].cpu().numpy().astype(np.int32)
        x_line = np.linspace(x1, x2, _len)
        y_line = np.linspace(y1, y2, _len)
        x_grid, y_grid = np.meshgrid(x_line, y_line)
        grid = np.stack([x_grid, y_grid], -1)
        pts = grid.reshape(-1, 2).reshape(-1)
        pts_list.append(pts)
        pts_long = pts.astype(np.long)
        pts_label = gt_masks[i][pts_long[1::2], pts_long[0::2]]
        pts_label_list.append(pts_label)
    pts_list = np.stack(pts_list, 0)
    pts_label_list = np.stack(pts_label_list, 0)
    return pts_list, pts_label_list


def mask_to_poly(mask, visualize=False):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            polygons.append(contour)
    return polygons

def sample_on_polygon(polygon, n_points=50):
    y = polygon[0::2]
    x = polygon[1::2]
    x_t, y_t = x[1:], y[1:]
    x_t.append(x[0])
    y_t.append(y[0])
    x_t, y_t = np.asarray(x_t), np.asarray(y_t)
    dist = np.sqrt((x_t - x) * (x_t - x) + (y_t - y) * (y_t - y))
    dist_sum = np.cumsum(dist)
    stride = dist_sum[-1] / (n_points+1e-6)

    x_set, y_set = [], []
    for i in range(n_points):
        length = i * stride
        idx = np.where(length <= dist_sum)[0][0]
        length_remained = dist_sum[idx] - length
        eps = 1e-6
        alpha = 1 - length_remained / (dist[idx] + eps)
        if idx < np.shape(dist)[0] - 1:
            x_i = (1 - alpha) * x[idx] + alpha * x[idx + 1]
            y_i = (1 - alpha) * y[idx] + alpha * y[idx + 1]
        else:
            x_i = (1 - alpha) * x[idx] + alpha * x[0]
            y_i = (1 - alpha) * y[idx] + alpha * y[0]
        x_set.append(x_i)
        y_set.append(y_i)
    x_set, y_set = np.array(x_set), np.array(y_set)
    point_set = np.concatenate((y_set[:,None], x_set[:,None]), 1)
    point_set = point_set.reshape(-1)
    return point_set

def polygon_len(polygon):
    y = polygon[0::2]
    x = polygon[1::2]
    x_t, y_t = x[1:], y[1:]
    x_t.append(x[0])
    y_t.append(y[0])
    x_t, y_t = np.asarray(x_t), np.asarray(y_t)
    dist = np.sqrt((x_t - x) * (x_t - x) + (y_t - y) * (y_t - y))
    perimeter = sum(dist)
    return perimeter

def sample_contour(gt_bboxes, gt_masks, cfg, num_pts):
    pts_list = []
    pts_label_list = []
    for i in range(len(gt_bboxes)):
        polygons = mask_to_poly(gt_masks[i])
        if len(polygons) == 0:
            pts = np.zeros([2 * num_pts], np.float64)
        else:
            polygons_len = np.array([polygon_len(polygon) for polygon in polygons])
            polygons_ratio = polygons_len / polygons_len.sum()
            polynums_num = (polygons_ratio * num_pts).astype(np.int)
            polynums_num[-1] = num_pts - polynums_num[:-1].sum()
            fuse_polygon = []
            for poly_numlen, polygon in zip(polynums_num, polygons):
                fuse_polygon.append(sample_on_polygon(polygon, poly_numlen))
            pts = np.concatenate(fuse_polygon)

        pts_list.append(pts)
        pts_long = pts.astype(np.long)
        pts_label = gt_masks[i][pts_long[1::2], pts_long[0::2]]
        pts_label_list.append(pts_label)
    pts_list = np.stack(pts_list, 0)
    pts_label_list = np.stack(pts_label_list, 0)
    return pts_list, pts_label_list


def sample_dist(gt_bboxes, gt_masks, cfg, num_pts):
    sample_dist_p = cfg.get('sample_dist_p', 1.5)
    pts_list = []
    pts_label_list = []
    # _len = int(np.sqrt(num_pts))
    # assert _len**2 == num_pts
    for i in range(len(gt_bboxes)):
        x1, y1, x2, y2 = gt_bboxes[i].cpu().numpy().astype(np.int32)
        if cfg.get('resize_sample', True):
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            mask = mmcv.imresize(gt_masks[i][y1:y1 + h, x1:x1 + w],
                                 (cfg.mask_size, cfg.mask_size))
            polygons = mask_to_poly(mask)
            distance_map = np.ones(mask.shape).astype(np.uint8)
            for poly in polygons:
                poly = np.array(poly).astype(np.int)
                for j in range(len(poly) // 2):
                    x_0, y_0 = poly[2 * j:2 * j + 2]
                    if j == len(poly) // 2 - 1:
                        x_1, y_1 = poly[0:2]
                    else:
                        x_1, y_1 = poly[2 * j + 2:2 * j + 4]
                    cv2.line(distance_map, (x_0, y_0), (x_1, y_1), (0), thickness=2)
            roi_dist_map = cv2.distanceTransform(distance_map, cv2.DIST_L2, 3)
            con_index = np.stack(np.nonzero(roi_dist_map == 0)[::-1], axis=-1)
            roi_dist_map[roi_dist_map == 0] = 1
            prob_dist_map = 1 / roi_dist_map
            prob_dist_map = np.power(prob_dist_map, sample_dist_p)
            prob_dist_map = prob_dist_map / prob_dist_map.sum()

            index_y, index_x = np.nonzero(prob_dist_map > 0)
            index = np.stack([index_x, index_y], axis=-1)
            _len = index.shape[0]
            if len(con_index) == 0:
                pts = np.zeros([2 * num_pts])
            else:
                repeat = num_pts // _len
                mod = num_pts % _len
                perm = np.random.choice(_len, mod, replace=False, p=prob_dist_map.reshape(-1))
                draw = [index.copy() for i in range(repeat)]
                draw.append(index[perm])
                draw = np.concatenate(draw, 0)
                # draw[:num_extreme] = extremes[:num_extreme]
                draw = draw + np.random.rand(*draw.shape)
                x_scale = float(w) / cfg.mask_size
                y_scale = float(h) / cfg.mask_size
                draw[:, 0] = draw[:, 0] * x_scale + x1
                draw[:, 1] = draw[:, 1] * y_scale + y1
                pts = draw.reshape(2 * num_pts)
        else:
            polygons = mask_to_poly(gt_masks[i])
            distance_map = np.ones(gt_masks[i].shape).astype(np.uint8)
            for poly in polygons:
                poly = np.array(poly).astype(np.int)
                for j in range(len(poly) // 2):
                    x_0, y_0 = poly[2 * j:2 * j + 2]
                    if j == len(poly) // 2 - 1:
                        x_1, y_1 = poly[0:2]
                    else:
                        x_1, y_1 = poly[2 * j + 2:2 * j + 4]
                    cv2.line(distance_map, (x_0, y_0), (x_1, y_1), (0), thickness=2)
            dist = cv2.distanceTransform(distance_map, cv2.DIST_L2, 3)
            roi_dist_map = dist[y1:y2, x1:x2]
            con_index = np.stack(np.nonzero(roi_dist_map == 0)[::-1], axis=-1)
            roi_dist_map[roi_dist_map == 0] = 1
            prob_dist_map = 1/roi_dist_map
            prob_dist_map = np.power(prob_dist_map, sample_dist_p)
            prob_dist_map = prob_dist_map/prob_dist_map.sum()

            index_y, index_x = np.nonzero(prob_dist_map > 0)
            index = np.stack([index_x, index_y], axis=-1)
            _len = index.shape[0]
            if len(con_index) == 0:
                pts = np.zeros([2 * num_pts])
            else:
                repeat = num_pts // _len
                mod = num_pts % _len
                perm = np.random.choice(_len, mod, replace=False, p=prob_dist_map.reshape(-1))
                draw = [index.copy() for i in range(repeat)]
                draw.append(index[perm])
                draw = np.concatenate(draw, 0)
                draw[:, 0] = draw[:, 0] + x1
                draw[:, 1] = draw[:, 1] + y1
                pts = draw.reshape(2 * num_pts)

        pts_list.append(pts)
        pts_long = pts.astype(np.long)
        pts_label = gt_masks[i][pts_long[1::2], pts_long[0::2]]
        pts_label_list.append(pts_label)
    pts_list = np.stack(pts_list, 0)
    pts_label_list = np.stack(pts_label_list, 0)
    return pts_list, pts_label_list