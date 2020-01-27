import torch
from mmdet.core import bbox2result
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import numpy as np
import cv2
from mmcv.image import imread, imwrite
from mmdet.core import tensor2imgs, get_classes
import mmcv
from pycocotools import mask as maskUtils
from mmdet.core.post_processing import interplate


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
        bbox_results = [
            bbox2result(det_bbox, det_labels, self.bbox_head.num_classes)
            for det_bbox, det_pts, det_masks, det_labels in bbox_list
        ][0]
        pts_results = [
            pts2result(det_points, det_cls_labels, self.bbox_head.num_classes)
        ][0]
        segm_results = [
           self.get_seg_masks(det_masks[:, :-1], det_pts[:, :-1], det_bbox, det_labels, self.test_cfg,
                                       ori_shape, scale_factor, rescale)
           for i, (det_bbox, det_pts, det_masks, det_labels) in enumerate(bbox_list)
        ][0]

        if rescale:
            return bbox_results, segm_results
        else:
            # store pts_result if we want visualization
            return (bbox_results, segm_results), pts_results

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

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)
            _pts_score = pts_score[i]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            im_pts = det_pts[i].clone()
            im_pts = im_pts.reshape(-1, 2)
            im_pts[:, 0] = (im_pts[:, 0] - bbox[0]) * (28 - 1) / (w - 1)
            im_pts[:, 1] = (im_pts[:, 1] - bbox[1]) * (28 - 1) / (h - 1)
            im_pts_score = _pts_score
            _corner_mask_pred_ = torch.zeros(2, 2)
            bbox_mask = interplate(_corner_mask_pred_, im_pts.cpu(), im_pts_score.cpu(), (h, w)).numpy()
            bbox_mask = (bbox_mask > test_cfg.get('mask_thr_binary', 0.5)).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = maskUtils.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)
        return cls_segms

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
    """ Make the ground-truth for landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img_shape
    pts = torch.tensor(pts).cuda()
    heatmap = make_gaussian((h, w), pts, sigma=sigma)  # (N, H, W)
    heatmap = heatmap.max(0)[0]  # (H, W)
    return heatmap


def make_gaussian(size, pts, sigma=1):
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
