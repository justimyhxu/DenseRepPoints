from __future__ import division
from mmdet.ops.nms import nms_wrapper

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F
from mmdet.core import (PointGenerator,
                        point_mask_score_target,
                        multi_apply)

from mmdet.models.losses import chamfer_loss
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule


@HEADS.register_module
class DenseRepPointsMaskHead(nn.Module):
    """Dense RepPoint head.

    Args:
        TODO: documents needed
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 stacked_mask_convs=3,
                 num_group=9,
                 use_sparse_pts_for_cls=True,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_mask_score_init=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_mask_weight_init=5,
                 loss_mask_weight_refine=10,
                 use_grid_points=False,
                 center_init=True,
                 transform_method='minmax',
                 moment_mul=0.01):
        super(DenseRepPointsMaskHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.stacked_mask_convs = stacked_mask_convs
        self.num_points = num_points
        self.group_num = num_group
        self.use_sparse_pts_for_cls = use_sparse_pts_for_cls
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_mask_score_init = build_loss(loss_mask_score_init)
        self.loss_mask_weight_init = loss_mask_weight_init
        self.loss_mask_weight_refine = loss_mask_weight_refine
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            "The points number should be a square number."
        assert self.dcn_kernel % 2 == 1, \
            "The points number should be an odd square number."
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        self.reppoints_cls = nn.ModuleList()
        self.reppoints_pts_init = nn.ModuleList()
        self.reppoints_pts_refine = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        for i in range(self.stacked_mask_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points

        self.reppoints_cls_conv = nn.Conv2d(self.feat_channels * self.group_num, self.point_feat_channels, 1, 1, 0)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

        self.reppoints_pts_refine_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)

        pts_out_dim = 4 * self.num_points if self.use_grid_points else 2 * self.num_points
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

        self.reppoints_mask_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_mask_init_out = nn.Conv2d(self.point_feat_channels, self.num_points, 1, 1, 0)

        self.reppoints_mask_score_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 1, 1, 0)
        self.reppoints_mask_score_out = nn.Conv2d(self.point_feat_channels, 1, 1, 1, 0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)
        normal_init(self.reppoints_mask_init_conv, std=0.01)
        normal_init(self.reppoints_mask_init_out, std=0.01)
        normal_init(self.reppoints_mask_score_conv, std=0.01)
        normal_init(self.reppoints_mask_score_out, std=0.01)

    def points2bbox(self, pts, y_first=True):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                          ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                          ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                    self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                dim=1)
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """
        Base on the previous bboxes and regression values, we compute the
            regressed bboxes and generate the grids on the bboxes.
        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def sample_offset(self, x, flow, padding_mode):
        """sample feature based on offset
            Args:
                x (Tensor): size (n, c, h, w)
                offset (Tensor): size (n, 2, h, w), values relevant to feature resolution
                padding_mode (str): 'zeros' or 'border'
            Returns:
                Tensor: warped feature map (n, c, h, w)
        """
        assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = x.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid = grid + flow
        gx = 2 * grid[:, 0, :, :] / (w - 1) - 1
        gy = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid = torch.stack([gx, gy], dim=1)
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, padding_mode=padding_mode)

    def compute_offset_feature(self, x, offset, padding_mode):
        """sample feature based on offset
            Args:
                x (Tensor) : size (n, C, h, w), x first
                offset (Tensor) : size (n, sample_pts*2, h, w), x first
                padding_mode (str): 'zeros' or 'border'
            Returns:
                Tensor: offset_feature: size (n, sample_pts, C, h, w)
        """
        offset_reshape = offset.view(offset.shape[0], -1, 2, offset.shape[2], offset.shape[
            3])  # (n, sample_pts, 2, h, w)
        num_pts = offset_reshape.shape[1]
        offset_reshape = offset_reshape.contiguous().view(-1, 2, offset.shape[2],
                                                          offset.shape[3])  # (n*sample_pts, 2, h, w)
        x_repeat = x.unsqueeze(1).repeat(1, num_pts, 1, 1, 1)  # (n, sample_pts, C, h, w)
        x_repeat = x_repeat.view(-1, x_repeat.shape[2], x_repeat.shape[3], x_repeat.shape[4])  # (n*sample_pts, C, h, w)
        sampled_feat = self.sample_offset(x_repeat, offset_reshape, padding_mode)  # (n*sample_pts, C, h, w)
        sampled_feat = sampled_feat.view(-1, num_pts, sampled_feat.shape[1], sampled_feat.shape[2],
                                         sampled_feat.shape[3])  # (n, sample_pts, C, h, w)
        return sampled_feat

    def aggregate_offset_feature(self, x, offset, padding_mode):
        feature = self.compute_offset_feature(x, offset, padding_mode=padding_mode)  # (b, n, C, h, w)
        feature = feature.max(dim=1)[0]  # (b, C, h, w)
        return feature

    def forward_mask_head(self, x, pts, mask_feat):
        b, _, h, w = x.shape
        mask_refine_field = self.reppoints_mask_init_out(
            self.relu(self.reppoints_mask_init_conv(mask_feat)))  # (b, n*1, h, w)
        mask_refine_field = mask_refine_field.view(-1, 1, h, w)  # (b*n, 1, h, w)
        _pts_reshape = pts.view(b, -1, 2, h, w).view(-1, 2, h, w)  # (b*n, 2, h, w)
        mask_out_refine = self.compute_offset_feature(mask_refine_field, _pts_reshape,
                                                      padding_mode='border')  # (b*n, 1, h, w)
        pts_score_out_init = mask_out_refine.view(b, -1, 1, h, w).view(b, -1, h, w)  # (b, n, h, w)
        return pts_score_out_init

    def forward_single(self, x, test=False):
        b, _, h, w = x.shape
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        mask_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        for mask_conv in self.mask_convs:
            mask_feat = mask_conv(mask_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_detach = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init

        if not self.use_sparse_pts_for_cls:
            group_card = int(self.num_points * 2 / self.group_num)
            cls_offset = pts_out_init_detach.view(b, self.group_num, group_card, h, w)  # (b, g, n/g, h, w)
            cls_pts_feature = [
                self.aggregate_offset_feature(cls_feat, cls_offset[:, i, ...], padding_mode='border')
                for i in range(self.group_num)]
            cls_pts_feature = torch.stack(cls_pts_feature, dim=1)  # list of (b, g, C, h, w)
            cls_pts_feature = cls_pts_feature.contiguous().view(b, -1, h, w)
            cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_pts_feature)))
        else:
            cls_offset = pts_out_init_detach.view(b, self.group_num, -1, 2, h, w)
            cls_offset = cls_offset[:, :, 0, ...].reshape(b, -1, h, w)
            cls_pts_feature = self.compute_offset_feature(cls_feat, cls_offset, padding_mode='border')
            cls_pts_feature = cls_pts_feature.contiguous().view(b, -1, h, w)
            cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_pts_feature)))

        input_bfeat = pts_feat

        pts_refine_field = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(input_bfeat)))  # (b, n*2, h, w)
        pts_refine_field = pts_refine_field.view(b * self.num_points, -1, h, w)  # (b*n, 2, h, w)
        pts_out_init_detach_reshape = pts_out_init_detach.view(b, -1, 2, h, w).view(-1, 2, h, w)  # (b*n, 2, h, w)
        pts_out_refine = self.compute_offset_feature(pts_refine_field, pts_out_init_detach_reshape,
                                                     padding_mode='border')  # (b*n, 2, h, w)
        pts_out_refine = pts_out_refine.view(b, -1, h, w)  # (b, n*2, h, w)

        if self.use_grid_points:
            pts_out_refine = pts_out_refine.view(b, -1, 4, h, w).mean(1)
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init_detach

        if test:
            pts_out_init_detach = pts_out_refine

        pts_score_out_init = self.forward_mask_head(x, pts_out_init_detach, mask_feat)
        pts_out_init = (1 - self.pts_offset_gradient_mul) * pts_out_init.detach() + \
                       self.pts_offset_gradient_mul * pts_out_init
        pts_out_refine = (1 - self.pts_offset_gradient_mul) * pts_out_refine.detach() + \
                         self.pts_offset_gradient_mul * pts_out_refine
        return cls_out, pts_out_init, pts_out_refine, pts_score_out_init

    def forward(self, feats, test=False):
        return multi_apply(self.forward_single, feats, test=test)

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def bboxes_to_centers(self, bbox_list):
        """Get center points according to bboxes. Only used in MaxIOUAssigner.
        """
        point_list = []
        for i_lvl in range(len(self.point_strides)):
            point_x = (bbox_list[i_lvl][..., 0] + bbox_list[i_lvl][..., 2]) / 2.
            point_y = (bbox_list[i_lvl][..., 1] + bbox_list[i_lvl][..., 3]) / 2.
            point_stride = (bbox_list[i_lvl][..., 2] - bbox_list[i_lvl][..., 0]) / self.point_base_scale
            point = torch.stack([point_x, point_y, point_stride], 2)
            point_list.append(point)
        return point_list

    def proposal_to_pts(self, center_list, pred_list, y_first=True):
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                if y_first:
                    yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                    pts_y = yx_pts_shift[..., 0::2]
                    pts_x = yx_pts_shift[..., 1::2]
                    xy_pts_shift = torch.stack([pts_x, pts_y], -1)
                    xy_pts_shift = xy_pts_shift.view(*pts.shape[:-1], -1)
                else:
                    xy_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def proposal_to_imgpts(self, center_list, pred_list, y_first=True):
        pts_list = []
        for i_img, point in enumerate(center_list):
            pts_img = []
            for i_lvl in range(len(center_list[0])):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                if y_first:
                    yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                    pts_y = yx_pts_shift[..., 0::2]
                    pts_x = yx_pts_shift[..., 1::2]
                    xy_pts_shift = torch.stack([pts_x, pts_y], -1)
                    xy_pts_shift = xy_pts_shift.view(*pts.shape[:-1], -1)
                else:
                    xy_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_img.append(pts)
            pts_list.append(pts_img)
        return pts_list

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, pts_score_pred_init,
                    labels, label_weights, bbox_gt_init, mask_gt_init, mask_gt_label_init,
                    bbox_weights_init, bbox_gt_refine, mask_gt_refine, mask_gt_label_refine,
                    bbox_weights_refine, stride, num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_refine)

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=num_total_samples_init)
        loss_pts_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)

        # mask loss
        # mask_gt_init = mask_gt_init.reshape(-1, mask_gt_init.shape[-1])
        valid_mask_gt_init = torch.cat(mask_gt_init, 0)
        valid_mask_gt_init = valid_mask_gt_init.view(-1, self.num_points, 2)
        mask_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        valid_mask_pred_init = mask_pred_init[bbox_weights_init[:, 0] > 0]
        valid_mask_pred_init = valid_mask_pred_init.view(-1, self.num_points, 2)
        valid_mask = valid_mask_gt_init.sum(-1).sum(-1) > 0
        num_total_samples = max(num_total_samples_init, 1)
        mask_loss = chamfer_loss(valid_mask_gt_init[valid_mask] / normalize_term,
                                 valid_mask_pred_init[valid_mask] / normalize_term).sum() / num_total_samples
        loss_mask_init = self.loss_mask_weight_init * mask_loss

        # mask_gt_refine = mask_gt_refine.reshape(-1, mask_gt_refine.shape[-1])
        valid_mask_gt_refine = torch.cat(mask_gt_refine, 0)
        valid_mask_gt_refine = valid_mask_gt_refine.view(-1, self.num_points, 2)
        mask_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        valid_mask_pred_refine = mask_pred_refine[bbox_weights_refine[:, 0] > 0]
        valid_mask_pred_refine = valid_mask_pred_refine.view(-1, self.num_points, 2)
        valid_mask = valid_mask_gt_refine.sum(-1).sum(-1) > 0
        num_total_samples = max(num_total_samples_refine, 1)
        mask_loss = chamfer_loss(valid_mask_gt_refine[valid_mask] / normalize_term,
                                 valid_mask_pred_refine[valid_mask] / normalize_term).sum() / num_total_samples
        loss_mask_refine = self.loss_mask_weight_refine * mask_loss
        valid_mask_gt_label_refine = torch.cat(mask_gt_label_refine, 0)
        valid_mask_gt_label_refine = valid_mask_gt_label_refine.view(-1, self.num_points, 1)
        mask_score_pred_init = pts_score_pred_init.reshape(-1, self.num_points)
        valid_mask_score_pred_init = mask_score_pred_init[bbox_weights_refine[:, 0] > 0]
        valid_mask_score_pred_init = valid_mask_score_pred_init.view(-1, self.num_points, 1)
        valid_mask_score = (valid_mask_gt_label_refine.sum(-1).sum(-1) > 0)
        num_total_samples = max(num_total_samples_refine, 1)

        # todo normalize term is key, what is loss is important

        loss_mask_score_init = self.loss_mask_score_init(
            valid_mask_score_pred_init[valid_mask_score],
            valid_mask_gt_label_refine[valid_mask_score],
            bbox_weights_init.new_ones(*valid_mask_score_pred_init[valid_mask_score].shape),
            avg_factor=num_total_samples
        )
        loss_mask_score_init = loss_mask_score_init / self.num_points
        return loss_cls, loss_pts_init, loss_mask_init, loss_pts_refine, loss_mask_refine, loss_mask_score_init

    #
    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             pts_preds_score_init,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        proposal_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        real_pts_preds_init = self.proposal_to_pts(proposal_list, pts_preds_init, y_first=False)
        proposal_pts_list = self.proposal_to_imgpts(proposal_list, pts_preds_init, y_first=False)
        real_pts_preds_score_init = []
        for lvl_pts_score in pts_preds_score_init:
            b = lvl_pts_score.shape[0]
            real_pts_preds_score_init.append(lvl_pts_score.permute(0, 2, 3, 1).view(b, -1, self.num_points))
        if cfg.init.assigner['type'] != 'PointAssigner':
            proposal_list = self.centers_to_bboxes(proposal_list)

        cls_reg_targets_init = point_mask_score_target(
            proposal_list,
            proposal_pts_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            cfg.init,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            num_pts=self.num_points)

        (*_, bbox_gt_list_init, mask_gt_list_init, mask_gt_label_list_init, proposal_list_init,
         bbox_weights_list_init, num_total_pos_init, num_total_neg_init) = cls_reg_targets_init
        num_total_samples_init = (num_total_pos_init + num_total_neg_init if self.sampling else num_total_pos_init)

        # target for refinement stage
        proposal_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        real_pts_preds_refine = self.proposal_to_pts(proposal_list, pts_preds_refine, y_first=False)
        bbox_pts_list = self.proposal_to_imgpts(proposal_list, pts_preds_init, y_first=False)

        bbox_list = []
        for i_img, point in enumerate(proposal_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.transform_box(pts_preds_init[i_lvl].detach(), y_first=False)
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).contiguous().view(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = point_mask_score_target(
            bbox_list,
            bbox_pts_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            cfg.refine,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            num_pts=self.num_points)
        (labels_list, label_weights_list, bbox_gt_list_refine, mask_gt_list_refine, mask_gt_label_list_refine,
         proposal_list_refine,
         bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine
        num_total_samples_refine = (
            num_total_pos_refine + num_total_neg_refine if self.sampling else num_total_pos_refine)

        # compute loss
        losses_cls, losses_pts_init, losses_mask_init, losses_pts_refine, losses_mask_refine, losses_mask_score_init = multi_apply(
            self.loss_single,
            cls_scores,
            real_pts_preds_init,
            real_pts_preds_refine,
            real_pts_preds_score_init,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            mask_gt_list_init,
            mask_gt_label_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            mask_gt_list_refine,
            mask_gt_label_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_refine)
        loss_dict_all = {'loss_cls': losses_cls,
                         'loss_pts_init': losses_pts_init,
                         'losses_mask_init': losses_mask_init,
                         'loss_pts_refine': losses_pts_refine,
                         'losses_mask_refine': losses_mask_refine,
                         'losses_mask_score_init': losses_mask_score_init}
        return loss_dict_all

    def get_bboxes(self, cls_scores, pts_preds_init, pts_preds_refine, pts_preds_score_refine, img_metas, cfg,
                   rescale=False, nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        bbox_preds_refine = [self.transform_box(pts_pred_refine, y_first=False) for pts_pred_refine in pts_preds_refine]
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            pts_pred_list = [
                pts_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            mask_pred_list = [
                pts_preds_score_refine[i][img_id].sigmoid().detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, pts_pred_list, mask_pred_list,
                                               mlvl_points, img_shape, scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          pts_preds,
                          mask_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_pts = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        for i_lvl, (cls_score, bbox_pred, pts_pred, mask_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, pts_preds, mask_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            pts_pred = pts_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                pts_pred = pts_pred[topk_inds, :]
                mask_pred = mask_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts_pred * self.point_strides[i_lvl] + pts_pos_center
            pts[:, 0::2] = pts[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            pts[:, 1::2] = pts[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            bboxes[:, 0::2] = bboxes[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes[:, 1::2] = bboxes[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            mlvl_pts.append(pts)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_masks.append(mask_pred)
        mlvl_pts = torch.cat(mlvl_pts)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_pts /= mlvl_pts.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_masks = torch.cat(mlvl_masks)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_bboxes, det_pts, det_masks, det_labels = multiclass_bbox_pts_nms(
                mlvl_bboxes, mlvl_pts, mlvl_scores, mlvl_masks, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_pts, det_masks, det_labels
        else:
            return mlvl_bboxes, mlvl_pts, mlvl_masks, mlvl_scores


def multiclass_bbox_pts_nms(multi_bboxes,
                            multi_pts,
                            multi_scores,
                            multi_masks,
                            score_thr,
                            nms_cfg,
                            max_num=-1,
                            score_factors=None):
    num_classes = multi_scores.shape[1]
    bboxes, pts, labels, masks = [], [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        _pts = multi_pts[cls_inds, :]
        _masks = multi_masks[cls_inds, :]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_pts = torch.cat([_pts, _scores[:, None]], dim=1)
        cls_masks = torch.cat([_masks, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_pts = cls_pts[_, :]
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0],), i - 1, dtype=torch.long)
        cls_masks = cls_masks[_, :]
        bboxes.append(cls_dets)
        pts.append(cls_pts)
        masks.append(cls_masks)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        pts = torch.cat(pts)
        labels = torch.cat(labels)
        masks = torch.cat(masks)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            pts = pts[inds]
            labels = labels[inds]
            masks = masks[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        pts = multi_bboxes.new_zeros((0, 52))
        masks = multi_masks.new_zeros((0, 26))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

    return bboxes, pts, masks, labels