import numpy as np
from scipy import stats
from collections import Counter

class ClassificationAcc:
    def __init__(self, preds, labels, num_classes):
        super().__init__()
        self.preds = np.array(preds)
        self.labels = np.array(labels)
        self.num_classes = num_classes
        self.count_matrix = np.zeros((self.num_classes,self.num_classes), dtype=np.int32)
        self.TP_class = np.zeros((self.num_classes), dtype=np.int32)
        self.T_class = np.zeros((self.num_classes), dtype=np.int32)
        self.P_class = np.zeros((self.num_classes), dtype=np.int32)

        for i in range(len(self.preds)):
            self.count_matrix[self.labels[i]][self.preds[i]]+=1
        self.recall_matrix = self.count_matrix / np.sum(self.count_matrix, axis=-1)[...,None]
        self.precision_matrix = self.count_matrix / np.sum(self.count_matrix, axis=0)[None,...]

        for c in range(0, num_classes):
            self.TP_class[c] += np.sum((self.preds == c) & (self.labels == c))
            self.P_class[c] += np.sum(self.preds == c)
            self.T_class[c] += np.sum(self.labels == c)
        self.precision_class = self.TP_class / (self.P_class + 0.0001)
        self.recall_class = self.TP_class / self.T_class
        self.precision = np.sum(self.TP_class) / np.sum(self.P_class)
        self.recall = np.sum(self.TP_class) / np.sum(self.T_class)
        self.avg_precision = np.mean(self.precision_class)
        self.avg_recall = np.mean(self.recall_class)

    def __call__(self):
        return self.avg_precision, self.avg_recall


class ClusterEvalIoU:
    def __init__(self, preds, labels, IoU_thres=0.5):
        super().__init__()
        self.preds = preds
        self.labels = labels
        self.IoU_thres = IoU_thres

        unique_labels = Counter(list(self.labels))
        unique_preds = Counter(list(self.preds))

        self.TP = 0
        for cluster_id in unique_labels.keys():
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            mode, count = stats.mode(self.preds[point_ids], axis=None)
            mode = mode[0]
            count = count[0]
            IoU = count / (unique_labels[cluster_id]+unique_preds[mode]-count)
            if mode>-1 and IoU >= IoU_thres:
                self.TP+=1

        self.P = len(unique_preds) - (1 if unique_preds[-1]!=0 else 0)
        self.T = len(unique_labels)
        self.precision = np.nan if self.P==0 else self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = np.nan if self.T==0 else self.TP / self.T  # what percent of actual clusters are identified

    def __call__(self):
        return self.precision, self.recall



