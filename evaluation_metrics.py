import numpy as np
from scipy import stats
from collections import Counter
import sklearn
from sklearn import metrics

class Purity:
    def __init__(self, preds, labels):
        contingency_matrix = metrics.cluster.contingency_matrix(labels, preds)
        self.purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def __call__(self):
        return self.purity


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
        self.precision = 0 if self.P==0 else self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = 0 if self.T==0 else self.TP / self.T  # what percent of actual clusters are identified
        self.F1 = 0 if (self.precision==0 or self.recall==0) else 2 * self.precision * self.recall / (self.precision + self.recall)

    def __call__(self):
        return self.precision, self.recall


class ClusterEvalMode:
    def __init__(self, preds, labels):
        super().__init__()
        self.preds = preds
        self.labels = labels

        unique_labels = Counter(list(self.labels))
        unique_preds = Counter(list(self.preds))

        self.TP = 0
        for cluster_id in unique_labels.keys():
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            mode_pred, count_pred = stats.mode(self.preds[point_ids], axis=None)
            point_ids_preds = np.argwhere(self.preds == mode_pred)[:,0]
            mode_label, count_label = stats.mode(self.labels[point_ids_preds], axis=None)
            if mode_pred>-1 and mode_label>-1 and mode_label == cluster_id:
                self.TP += 1

        self.P = len(unique_preds) - (1 if unique_preds[-1]!=0 else 0)
        self.T = len(unique_labels)
        self.precision = 0 if self.P==0 else self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = 0 if self.T==0 else self.TP / self.T  # what percent of actual clusters are identified
        self.F1 = 0 if (self.precision==0 or self.recall==0) else 2 * self.precision * self.recall / (self.precision + self.recall)

    def __call__(self):
        return self.precision, self.recall


class ClusterEvalModeC:
    def __init__(self, preds, labels):
        super().__init__()
        self.preds = preds
        self.labels = labels

        unique_labels = set(list(self.labels))

        self.TP_C = 0
        for cluster_id in unique_labels:
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            mode_pred, count_pred = stats.mode(self.preds[point_ids], axis=None)
            point_ids_preds = np.argwhere(self.preds == mode_pred)[:,0]
            mode_label, count_label = stats.mode(self.labels[point_ids_preds], axis=None)
            if mode_pred>-1 and mode_label>-1 and mode_label == cluster_id:
                self.TP_C += count_label[0]

        self.recall_C = self.TP_C / len(self.labels)

    def __call__(self):
        return self.recall_C

class ClusterEvalAll:
    def __init__(self, preds, labels):
        super().__init__()
        self.results = {}
        IoU = ClusterEvalIoU(preds, labels, IoU_thres=0.5)
        self.results['IoU_TP'] = IoU.TP
        self.results['IoU_T'] = IoU.T
        self.results['IoU_P'] = IoU.P
        self.results['IoU_precision'] = IoU.precision
        self.results['IoU_recall'] = IoU.recall
        self.results['IoU_F1'] = IoU.F1

        Mode = ClusterEvalMode(preds, labels)
        self.results['Mode_TP'] = Mode.TP
        self.results['Mode_T'] = Mode.T
        self.results['Mode_P'] = Mode.P
        self.results['Mode_precision'] = Mode.precision
        self.results['Mode_recall'] = Mode.recall
        self.results['Mode_F1'] = Mode.F1

        ModeC = ClusterEvalModeC(preds, labels)
        self.results['Mode_TP_C'] = ModeC.TP_C
        self.results['Mode_recall_C'] = ModeC.recall_C

        purity = Purity(preds, labels)
        self.results['Purity'] = purity()

        AMI = sklearn.metrics.adjusted_mutual_info_score(labels, preds)
        self.results['AMI'] = AMI

        ARand = sklearn.metrics.adjusted_rand_score(labels, preds)
        self.results['ARand'] = ARand

    def __call__(self):
        return self.results

    @classmethod
    def aggregate(cls, results_list):
        f_results = {key:0 for key in results_list[0].keys()}
        for results in results_list:
            for key, value in results.items():
                f_results[key] += value
        f_results = {key:value/len(results_list) for key,value in f_results.items()}
        return f_results



