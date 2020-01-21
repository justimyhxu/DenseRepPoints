import torch
import time
from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import numpy as np
import cv2
from mmcv.image import imread, imwrite
from mmdet.core import tensor2imgs, get_classes
import mmcv
from pycocotools import mask as maskUtils
from mmdet.core.post_processing import interplate_v5, interplate_v6


@DETECTORS.register_module
class DenseRepPointsMaskDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DenseRepPointsMaskDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, test=False)
        loss_inputs = outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)
        # self.bbox_head.show_train(img, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, test=True)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        _, det_points, det_masks, det_cls_labels = bbox_list[0]
        if det_points.shape[0] > 0:
            _det_points = det_points[:, :-1].reshape(det_points.shape[0], -1, 2)
            _det_masks = det_masks[:, :-1].reshape(det_masks.shape[0], -1, 1)
            _det_points = torch.cat([_det_points, _det_masks], dim=-1).reshape(det_points.shape[0], -1)
            det_points = torch.cat([_det_points, det_points[:, [-1]]], dim=-1)

        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        # import ipdb
        # ipdb.set_trace()
        bbox_results = [
            bbox2result(det_bbox, det_labels, self.bbox_head.num_classes)
            for det_bbox, det_pts, det_masks, det_labels in bbox_list
        ][0]
        pts_results = [
            pts2result(det_points, det_cls_labels, self.bbox_head.num_classes)
            # for det_pts,  det_labels in zip(det_points, det_cls_labels)
        ][0]
        # masks_results = [
        #     masks2result(det_masks, det_labels, self.bbox_head.num_classes)
        #     for det_bbox, det_pts, det_masks, det_labels in bbox_list
        # ][0]
        if True:
            rle_results = [
               self.get_seg_masks(det_masks[:, :-1], det_pts[:, :-1], det_bbox, det_labels, self.test_cfg,
                                           ori_shape, scale_factor, rescale)
               for i, (det_bbox, det_pts, det_masks, det_labels) in enumerate(bbox_list)
            ][0]
        else:
            rle_results = []
            for i_class in range(len(pts_results)):
                rle_result = []
                for i_seg in range(len(pts_results[i_class])):
                    pts_seg = pts_results[i_class][i_seg][:-1].reshape(-1, 3)
                    pts_seg = pts_seg[:, :-1]
                    rle_seg = self.heatmap_seg(pts_seg, img_meta[0], rescale, img)
                    rle_result.append(rle_seg)
                rle_results.append(rle_result)

        return bbox_results, rle_results

    def get_seg_masks(self, pts_score, det_pts, det_bboxes, det_labels, test_cfg,
                      ori_shape, scale_factor, rescale=False):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            pts_score (Tensor or ndarray): shape (n, #class+1, num_pts).
                For single-scale testing, pts_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_pts (Tensor): shape (n, num_pts)
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """

        cls_segms = [[] for _ in range(self.bbox_head.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)

        scale_factor = 1.0

        # start_t = time.time()
        # core_time_count = 0
        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)


            # _pts_score = pts_score[i, 0].sigmoid()
            _pts_score = pts_score[i]

            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            im_pts = det_pts[i].clone()
            im_pts = im_pts.reshape(-1, 2)
            im_pts[:, 0] = (im_pts[:, 0] - bbox[0]) * (28 - 1) / (w - 1)
            im_pts[:, 1] = (im_pts[:, 1] - bbox[1]) * (28 - 1) / (h - 1)

            # im_pts[:, 0] = ((im_pts[:, 0] - bbox[0]+0.5) * (28) / (w)-0.5).clamp(0,w-1)
            # im_pts[:, 1] = ((im_pts[:, 1] - bbox[1]+0.5) * (28) / (h)-0.5).clamp(0,h-1)
            im_pts_score = _pts_score

            _corner_mask_pred_ = torch.zeros(2, 2)

            # core_start_t = time.time()
            bbox_mask = interplate_v5(_corner_mask_pred_, im_pts.cpu(), im_pts_score.cpu(), (h, w)).numpy()
            # core_end_t = time.time()
            # core_time_count += core_end_t - core_start_t

            bbox_mask = (bbox_mask > test_cfg.get('mask_thr_binary', 0.5)).astype(np.uint8)
            # import matplotlib.pyplot as plt
            # plt.imshow(bbox_mask)
            # plt.show()

            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = maskUtils.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)
        # end_t = time.time()
        # print()
        # print('core time per bbox: {:.5f}'.format(core_time_count / bboxes.shape[0]))
        # print('mask time per bbox: {:.5f}'.format((end_t - start_t) / bboxes.shape[0]))
        return cls_segms



    def heatmap_seg(self, pts_seg, img_meta, rescale, img, vis=False):
        import matplotlib.pyplot as plt

        if rescale:
            h, w = img_meta['ori_shape'][0], img_meta['ori_shape'][1]
        else:
            h, w = img_meta['img_shape'][0], img_meta['img_shape'][1]
        sigma = 2
        im_mask = np.zeros((h, w), dtype=np.uint8)
        proposal_l = int(pts_seg[:, 0].min())
        proposal_r = int(pts_seg[:, 0].max())
        proposal_u = int(pts_seg[:, 1].min())
        proposal_b = int(pts_seg[:, 1].max())
        proposal_w = proposal_r - proposal_l + 1
        proposal_h = proposal_b - proposal_u + 1
        pts_in_box_x = (pts_seg[:, 0] - proposal_l) / max(proposal_w - 1, 1) * 27
        pts_in_box_y = (pts_seg[:, 1] - proposal_u) / max(proposal_h - 1, 1) * 27
        pts_in_box = np.stack([pts_in_box_x, pts_in_box_y], -1)
        heatmap = make_heatmap((28, 28), pts_in_box, sigma).cpu().numpy()
        heatmap = mmcv.imresize(heatmap, (proposal_w, proposal_h))
        heatmap = (heatmap > 0.5).astype(np.uint8)
        # img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        # img_np = tensor2imgs(img, **img_norm_cfg)[0]
        # roi = img_np[proposal_u:proposal_u + proposal_h, proposal_l:proposal_l + proposal_w]
        # heatmap = self.dense_crf(heatmap, roi)
        im_mask[proposal_u:proposal_u + proposal_h, proposal_l:proposal_l + proposal_w] = heatmap

        # size = int(np.sqrt(proposal_w**2+proposal_w**2)/np.sqrt(self.seg_head.num_pts))
        # d_kernel = np.ones((size,size),np.uint8)
        # e_size = int(1.1*size)
        # e_kernel = np.ones((e_size,e_size),np.uint8)
        # mask = cv2.dilate(mask, d_kernel)
        # mask = cv2.erode(mask, e_kernel)
        rle = maskUtils.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        return rle

    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).
            rcnn_test_cfg (dict): rcnn test config.

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            # TODO more flexible
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, self.test_cfg.score_thr,
            self.test_cfg.nms, self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset='coco',
                    score_thr=0.3,
                    out_file=None
                    ):
        bbox_seg_result, pts_result = result
        if isinstance(bbox_seg_result, tuple):
            bbox_result, segm_result = bbox_seg_result
        else:
            bbox_result, segm_result = bbox_seg_result, None
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            pts = np.vstack(pts_result)
            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(pts.shape[0], i, dtype=np.int32)
                for i, pts in enumerate(pts_result)
            ]
            labels = np.concatenate(labels)
            imshow_det_pts(
                img_show,
                bboxes,
                pts,
                labels,
                out_file=out_file,
                class_names=class_names,
                score_thr=score_thr)


def pts2result(pts, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if pts.shape[0] == 0:
        return [
            np.zeros((0, pts.shape[1]), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        pts = pts.cpu().numpy()
        labels = labels.cpu().numpy()
        return [pts[labels == i, :] for i in range(num_classes - 1)]

def masks2result(masks, labels, num_classes):
    if masks.shape[0] == 0:
        return [
            np.zeros((0, masks.shape[1]), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        masks = masks.cpu().numpy()
        labels = labels.cpu().numpy()
        return [masks[labels == i, :] for i in range(num_classes - 1)]


def imshow_det_pts(img,
                   bboxes,
                   pts,
                   labels,
                   class_names=None,
                   score_thr=0,
                   bbox_color='green',
                   text_color='green',
                   thickness=1,
                   font_scale=0.5,
                   show=True,
                   win_name='',
                   wait_time=0,
                   out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert pts.ndim == 2
    assert labels.ndim == 1
    assert pts.shape[0] == labels.shape[0]
    img = imread(img)

    if score_thr > 0:
        scores = pts[:, -1]
        inds = scores > score_thr
        pts = pts[inds, :]
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    # colors_pts = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
    #               (255, 255, 0), (255, 0, 255), (0, 255, 255),
    #               (127, 64, 0), (127, 0, 64), (0, 127, 64)]

    for pt, bbox, label in zip(pts, bboxes, labels):
        r_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        pt_x = pt[0:-1:3]
        pt_y = pt[1:-1:3]
        pt_score = pt[2:-1:3]
        left_top = (int(bbox[0]), int(bbox[1]))
        right_bottom = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(
            img, left_top, right_bottom, r_color, thickness=thickness)
        for i in range(pt_x.shape[0]):
            r_fore_color = r_color
            r_back_color = [int(i / 2) for i in r_color]
            r_show_color = r_fore_color if pt_score[i] > 0.5 else r_back_color
            cv2.circle(img, (pt_x[i], pt_y[i]), 1, r_show_color, thickness=2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(pt) % 2 == 1:
            label_text += '|{:.02f}'.format(pt[-1])
        cv2.putText(img, label_text, (int(bbox[0]), int(bbox[1]) - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, r_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)


def make_heatmap(img_shape, pts, sigma=10):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img_shape
    pts = torch.tensor(pts).cuda()
    heatmap = make_gaussian((h, w), pts, sigma=sigma)  # (N, H, W)
    heatmap = heatmap.max(0)[0]  # (H, W)
    # heatmap = heatmap.sum(0)  # (H, W)
    return heatmap


def make_gaussian(size, pts, sigma=1, use_gpu=True):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = torch.arange(0, size[1], 1, dtype=torch.float32).cuda()
    y = torch.arange(0, size[0], 1, dtype=torch.float32).cuda()
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(-1)
    x0 = pts[:, 0].unsqueeze(-1).unsqueeze(-1)
    y0 = pts[:, 1].unsqueeze(-1).unsqueeze(-1)
    return torch.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
