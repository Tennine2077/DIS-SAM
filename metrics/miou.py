import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm
from mbiou import mask_to_boundary
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # acc = (TP) / TP + FP
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)
        return Acc_class

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        MIoU = MIoU[1]
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        if gt_image.max() > 1:
            gt_image[gt_image<128] = 0
            gt_image[gt_image>=128] = 1
        if pre_image.max() > 1:
            pre_image[pre_image<128] = 0
            pre_image[pre_image>=128] = 1
        self.confusion_matrix = self._generate_matrix(gt_image, pre_image)

    def add_b_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        if gt_image.max() > 1:
            gt_image[gt_image<128] = 0
            gt_image[gt_image>=128] = 1
        if pre_image.max() > 1:
            pre_image[pre_image<128] = 0
            pre_image[pre_image>=128] = 1
        gt_b = mask_to_boundary(gt_image)
        pre_b = mask_to_boundary(pre_image)
        self.confusion_matrix = self._generate_matrix(gt_b, pre_b)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def measure_pa_miou(num_class, gt_image, pre_image):
    metric = Evaluator(num_class)
    metric.add_batch(gt_image, pre_image)
    acc = metric.Pixel_Accuracy()
    mIoU = metric.Mean_Intersection_over_Union()
    metric.add_box(gt_image, pre_image)
    mbIoU = metric.Mean_Intersection_over_Union()
    print("像素准确度PA:", acc, "平均交互度mIOU:", mIoU,"平均BOX交互度mIOU:",mbIoU)
    return acc,mIoU,mbIoU

def get_files(path,name='.pkl'):
    file_lan = []
    for filepath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            if name not in os.path.join(filepath,filename):
                file_lan.append(os.path.join(filepath,filename))
    return file_lan

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        if gts.max() > 1:
            gts[gts<128] = 0
            gts[gts>=128] = 1
        if predictions.max() > 1:
            predictions[predictions<128] = 0
            predictions[predictions>=128] = 1
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def add_b_batch(self, predictions, gts):
        if gts.max() > 1:
            gts[gts<128] = 0
            gts[gts>=128] = 1
        if predictions.max() > 1:
            predictions[predictions<128] = 0
            predictions[predictions>=128] = 1
        gt_b = mask_to_boundary(gts)
        pre_b = mask_to_boundary(predictions)
        for lp, lt in zip(pre_b, gt_b):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

    def reset(self):
        self.hist = self.hist*0

if __name__=='__main__':
    gt_roots = [
        r"D:/Code/Design_for_graduation/DIS-main/IS-Net/DIS5K/DIS5K/DIS-VD/gt/",
                r"D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/COIFT/gt/",
                r'D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/HRSOD/gt/',
                r"D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/ThinObject5K/gt/"
                ]
    
    # pred_roots = [
    #     r"D:/Code/Design_for_graduation/segment-anything-main/Validation/vd/sam_raw/",
    #         r"D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/COIFT/sam/",
    #         r'D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/HRSOD/sam/',
    #             r"D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/ThinObject5K/sam/"
    #               ]

    # pred_roots = [
    #     r"D:\Code\Design_for_graduation\segment-anything-main\Validation\vd\sam_hq/",
    #         r"D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/COIFT/hqsam/",
    #         r'D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/HRSOD/hqsam/',
    #             r"D:/Code/Design_for_graduation/HQSeg_44K/HQSeg/testdata/ThinObject5K/hqsam/"
    #             ]

    # pred_roots = [
    # r'D:/Code/Design_for_graduation/DIS-main/your-results/hqsegdata_raw/DIS5K-VD-m/',
    #               r'D:/Code/Design_for_graduation/DIS-main/your-results/hqsegdata_raw/DIS5K-dataset_COIFT/',
    #               r'D:/Code/Design_for_graduation/DIS-main/your-results/hqsegdata_raw/DIS5K-dataset_HRSOD/',
    #               r'D:/Code/Design_for_graduation/DIS-main/your-results/hqsegdata_raw/DIS5K-dataset_ThinObject5K_TE/'
    #               ]

    pred_roots = [
    r'D:/Code/Design_for_graduation/DIS-main/your-results/train_hqdata/DIS5K-VD-m/',
                  r'D:/Code/Design_for_graduation/DIS-main/your-results/train_hqdata/COIFT/',
                  r'D:/Code/Design_for_graduation/DIS-main/your-results/train_hqdata/HRSOD/',
                  r'D:/Code/Design_for_graduation/DIS-main/your-results/train_hqdata/ThinObject5K_TE/'
                  ]

    allfile = pd.DataFrame()
    for i in range(gt_roots.__len__()):
        gt_root = gt_roots[i]
        pred_root = pred_roots[i]
        gt_name_list = get_files(pred_root)
        gt_name_list = sorted([x.split('/')[-1] for x in gt_name_list])
        metric = IOUMetric(2) 
        for gt_name in tqdm.tqdm(gt_name_list):
            gt_path = os.path.join(gt_root, gt_name)
            pred_path = os.path.join(pred_root, gt_name)
            # print(gt_path,pred_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            metric.add_batch(gt, pred)
        acc, acc_cls, iu, mean_iu, fwavacc = metric.evaluate()
        # print(
        #     # "像素准确度PA:", acc,
        #       "平均交互度mIOU:", iu[1],
        #       acc, acc_cls, iu, mean_iu, fwavacc)
        metric.reset()
        for gt_name in tqdm.tqdm(gt_name_list):
            gt_path = os.path.join(gt_root, gt_name)
            pred_path = os.path.join(pred_root, gt_name)
            # print(gt_path,pred_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            metric.add_b_batch(gt, pred)
        b_acc, b_acc_cls, b_iu, mean_b_iu, b_fwavacc = metric.evaluate()
        print("像素准确度PA:",acc_cls,
            "平均交互度mIOU:", iu[1],
              "平均边界交互度mBIOU:", b_iu[1],
              )
        