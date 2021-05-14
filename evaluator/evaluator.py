import os
import copy
import utils
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve, auc


class KinshipEvaluator:

    def __init__(self, set_name, pair, log_path, fold=None):
        plt.ioff()
        self.set_name = set_name
        self.pair = pair
        self.log_path = log_path
        self.fold = fold
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

    def save_hist(self):

        title = f"{self.pair.upper()} {self.set_name} Metrics"
        fig_path = os.path.join(self.log_path, f"{self.set_name}_{self.pair}_fold_{self.fold}.png")
        if self.fold is not None:
            title += f" Fold {self.fold}"

        fig = plt.figure()
        plt.title(title)
        plt.plot(self.metrics_hist["acc"], color="tomato", label="Accuracy")
        plt.plot(self.metrics_hist["f1-score"], color="turquoise", label="F1-Score", linestyle="--")
        plt.plot(self.metrics_hist["auc"], color="gold", label="AUC", linestyle=":")
        plt.legend()
        plt.grid(color='black', linestyle='--', linewidth=1, alpha=0.15)
        fig.savefig(fig_path)
        plt.close()
        utils.save_json(os.path.join(self.log_path, f"{self.set_name}_hist_{self.pair}_fold_{self.fold}.json"),
                        self.metrics_hist)
