import copy
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve, auc


class KinshipEvaluator:

    def __init__(self, log_path):
        self.log_path = log_path
        self.model_scores = list()
        self.labels = list()
        self.best_metrics = {
            "acc": -1,
            "recall": -1,
            "precision": -1,
            "f1-score": -1,

            "precision_curve": -1,
            "recall_curve": -1,
            "thresholds": -1,
            "auc": -1
        }
        self.metrics_hist = {
            "acc": list(),
            "recall": list(),
            "precision": list(),
            "f1-score": list(),

            "precision_curve": list(),
            "recall_curve": list(),
            "thresholds": list(),
            "auc": list()
        }

    def reset(self):
        self.model_scores = list()
        self.labels = list()

    def add_batch(self, scores, labels):
        self.model_scores += scores
        self.labels += labels

    def get_metrics(self, target_metric="acc"):
        metrics = dict()
        probas = np.array(self.model_scores)
        targets = np.array(self.labels)
        predictions = np.zeros_like(probas)
        predictions[probas > 0.5] = 1

        metrics["acc"] = accuracy_score(targets, predictions)
        metrics["recall"] = recall_score(targets, predictions)
        metrics["precision"] = precision_score(targets, predictions)
        metrics["f1-score"] = f1_score(targets, predictions)

        precisions, recalls, thresholds = precision_recall_curve(targets, probas)
        metrics["precision_curve"] = precisions
        metrics["recall_curve"] = recalls
        metrics["thresholds"] = thresholds
        metrics["auc"] = auc(recalls, precisions)

        if metrics[target_metric] > self.best_metrics[target_metric]:
            self.best_metrics = copy.deepcopy(metrics)

        for key, _ in self.metrics_hist.items():
            self.metrics_hist[key].append(metrics[key])

        return metrics
