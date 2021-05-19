import os
import json
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
        self.best_model_scores = None
        self.best_model_labels = None

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

        metrics["acc"] = float(accuracy_score(targets, predictions))
        metrics["recall"] = float(recall_score(targets, predictions))
        metrics["precision"] = float(precision_score(targets, predictions))
        metrics["f1-score"] = float(f1_score(targets, predictions))

        precisions, recalls, thresholds = precision_recall_curve(targets, probas)
        metrics["precision_curve"] = precisions
        metrics["recall_curve"] = recalls
        metrics["thresholds"] = thresholds
        metrics["auc"] = float(auc(recalls, precisions))

        if metrics[target_metric] > self.best_metrics[target_metric]:
            self.best_metrics = copy.deepcopy(metrics)
            self.best_model_scores = copy.deepcopy(self.model_scores)
            self.best_model_labels = copy.deepcopy(self.labels)

        for key, _ in self.metrics_hist.items():
            self.metrics_hist[key].append(metrics[key])

        return metrics

    def save_hist(self):

        title = f"{self.pair.upper()} {self.set_name} Metrics"
        log_name = f"{self.pair.lower()}_hist_{self.set_name.lower()}"
        if self.fold is not None:
            title += f" Fold {self.fold}"
            log_name += f"_fold_{self.fold}"
        fig_path = os.path.join(self.log_path, f"{log_name}.png")
        fig = plt.figure()
        plt.title(title)
        plt.plot(self.metrics_hist["acc"], color="tomato", label="Accuracy")
        plt.plot(self.metrics_hist["f1-score"], color="turquoise", label="F1-Score", linestyle="--")
        plt.plot(self.metrics_hist["auc"], color="gold", label="AUC", linestyle=":")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(color='black', linestyle='--', linewidth=1, alpha=0.15)
        fig.savefig(fig_path)
        plt.close()
        utils.save_json(os.path.join(self.log_path,
                                     f"{log_name}.json"),
                        self.metrics_hist)

    def save_best_metrics(self):
        title = f"{self.pair.upper()} {self.set_name} Precision Recall Curve"
        log_name = f"{self.pair.lower()}_{self.set_name.lower()}"
        if self.fold is not None:
            title += f" Fold {self.fold}"
            log_name += f"_fold_{self.fold}"

        fig_path = os.path.join(self.log_path, f"{log_name}.png")
        fscore = (2 * self.best_metrics["precision_curve"] * self.best_metrics["recall_curve"]) / \
                 (self.best_metrics["precision_curve"] + self.best_metrics["recall_curve"])
        ix = np.nanargmax(fscore)
        best_threshold = float(self.best_metrics["thresholds"][ix])

        fig = plt.figure()
        plt.plot(self.best_metrics["recall_curve"], self.best_metrics["precision_curve"],  color="turquoise",
                 label="PR", linestyle="--")
        plt.scatter(self.best_metrics["recall_curve"][ix], self.best_metrics["precision_curve"][ix],
                    marker='o', color='tomato', label='Best')
        plt.title(f"{title} AUC: {auc(self.best_metrics['recall_curve'], self.best_metrics['precision_curve']):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(color='black', linestyle='--', linewidth=1, alpha=0.15)
        fig.savefig(fig_path)
        plt.close()

        best_metrics = copy.deepcopy(self.best_metrics)
        best_metrics["best_threshold"] = best_threshold

        best_metrics["precision_curve"] = list(map(float, best_metrics["precision_curve"]))
        best_metrics["recall_curve"] = list(map(float, best_metrics["recall_curve"]))
        best_metrics["thresholds"] = list(map(float, best_metrics["thresholds"]))

        utils.save_json(os.path.join(self.log_path, f"{log_name}.json"), best_metrics)

    def get_kinface_pair_metrics(self, evaluators, pair_type):
        accs, recalls, precisions, f_scores, scores, labels = list(), list(), list(), list(), list(), list()

        for evaluator in evaluators:
            accs.append(evaluator.best_metrics["acc"])
            recalls.append(evaluator.best_metrics["recall"])
            precisions.append(evaluator.best_metrics["precision"])
            f_scores.append(evaluator.best_metrics["f1-score"])
            scores += evaluator.best_model_scores
            labels += evaluator.best_model_labels
        pair_precisions, pair_recalls, pair_thresholds = precision_recall_curve(labels, scores)
        pair_metrics = dict()
        pair_metrics["acc"] = float(np.mean(accs))
        pair_metrics["recall"] = float(np.mean(recalls))
        pair_metrics["precision"] = float(np.mean(precisions))
        pair_metrics["f1-score"] = float(np.mean(f_scores))
        pair_metrics["auc"] = auc(pair_recalls, pair_precisions)

        fscore = (2 * pair_precisions * pair_recalls) / (pair_precisions + pair_recalls)
        ix = np.nanargmax(fscore)
        fig_path = os.path.join(self.log_path, f"{pair_type.upper()}.png")
        title = f"{pair_type.upper()} Precision Recall Curve"
        fig = plt.figure()
        plt.plot(pair_recalls, pair_precisions, color="turquoise", label="PR", linestyle="--")
        plt.scatter(pair_recalls[ix], pair_precisions[ix], marker='o', color='tomato', label='Best')
        plt.title(f"{title} AUC: {pair_metrics['auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(color='black', linestyle='--', linewidth=1, alpha=0.15)
        fig.savefig(fig_path)
        plt.close()

        pair_metrics["precision_curve"] = list(map(float, pair_precisions))
        pair_metrics["recall_curve"] = list(map(float, pair_recalls))
        pair_metrics["thresholds"] = list(map(float, pair_thresholds))

        utils.save_json(os.path.join(self.log_path,
                                     f"{self.pair}.json"),
                        pair_metrics)
