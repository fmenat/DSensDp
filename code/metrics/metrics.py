import numpy as np
        
from .metric_predictions import OverallAccuracy,F1Score,Precision,Recall,Kappa, ConfusionMatrix,ROC_AUC
from .metric_predictions import Get_Data
from .metric_predictions import CatEntropy,LogP,P_max, P_true, mAP
    
class BaseMetrics(object):
    """central metrics class to provide standard metric types"""

    def __init__(self, task_type="classification", metric_types=[]):
        """build BaseMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        self.task_type = task_type
        self.metric_types = [v.lower() for v in metric_types]
        self.metric_dict = {}

    def __call__(self, prediction, target):
        """call forward for each metric in collection

        Parameters
        ----------
        prediction : array_like (n_samples, n_outputs)
            prediction tensor
        target : array_like     (n_samples, n_outputs)
            ground truth tensor, in classification: n_outputs=1, currently working only =1
        """
        if not isinstance(prediction, np.ndarray):
            prediction = np.asarray(prediction)
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)

        return {name: func(prediction, target) for (name, func) in self.metric_dict.items()}

    def get_metric_types(self):
        """return list of metric types inside collection

        Returns
        -------
        list of strings
        """
        return list(self.metric_dict.keys())

    def reverse_forward(self, target,prediction):
        return self(prediction, target)
    

class ClassificationMetrics(BaseMetrics):
    def __init__(self, metric_types=["OA","KAPPA","F1 MACRO","P MACRO","AA","F1 Weighted","P Weighted", "R Weighted", "ENTROPY","LOGP"],
                 task_type="classification"): 
        """build ClassificationMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(ClassificationMetrics,self).__init__(task_type, metric_types)
        for metric in self.metric_types:
            if "oa"==metric:
                self.metric_dict["OA"] = OverallAccuracy()
            elif "aa"==metric or "r macro" == metric or "recall"==metric:
                self.metric_dict[metric.upper()] = Recall("macro") #AverageAccuracy()
            elif "f1" in metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"F1 {avg_mode.upper()}"] = F1Score(avg_mode)
            elif "p " in metric or "precision"==metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"P {avg_mode.upper()}"] = Precision(avg_mode)
            elif "r " in metric or "recall"==metric :
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"R {avg_mode.upper()}"] = Recall(avg_mode)
            elif "kappa"==metric:
                self.metric_dict["KAPPA"] = Kappa()
            elif "confusion" in metric or "matrix" in metric:
                self.metric_dict["MATRIX"] = ConfusionMatrix()
            elif "ntrue"==metric:
                self.metric_dict["N TRUE"] = Get_Data(self.task_type,"true")
            elif "npred"==metric:
                self.metric_dict["N PRED"] = Get_Data(self.task_type,"pred")


class SoftClassificationMetrics(BaseMetrics):
    def __init__(self, metric_types=["ENTROPY","LOGP", "PMAX", "PTRUE"]):
        """build SoftClassificationMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(SoftClassificationMetrics,self).__init__("classification", metric_types)
        for metric in self.metric_types:
            if "auc" in metric or "roc-auc"==metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"AUC {avg_mode.upper()}"] = ROC_AUC(avg_mode)
            elif "map" in metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"mAP {avg_mode.upper()}"] = mAP(avg_mode)
            elif "entropy" == metric:
                self.metric_dict["ENTROPY"] = CatEntropy()
            elif "logp"==metric:
                self.metric_dict["LOGp"] = LogP()
            elif "pmax"==metric:
                self.metric_dict["Pmax"] = P_max()
            elif "ptrue"==metric:
                self.metric_dict["Ptrue"] = P_true()


class BaseMetrics_3args(object):
    """central metrics class to provide standard metric types"""

    def __init__(self, metric_types=[]):
        """build BaseMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        self.metric_types = [v.lower() for v in metric_types]
        self.metric_dict = {}

    def __call__(self, real_prediction, noise_prediction, ground_truth):
        """call forward for each metric in collection

        Parameters
        ----------
        real_prediction : array_like (n_samples, n_outputs)
            real prediction tensor
        noise_prediction: array_like (n_samples, n_outputs)
            noise prediction tensor
        ground_truth : array_like     (n_samples, n_outputs)
            ground truth tensor, in classification: n_outputs=1, currently working only =1
        """
        if not isinstance(real_prediction, np.ndarray):
            real_prediction = np.asarray(real_prediction)
        if not isinstance(noise_prediction, np.ndarray):
            noise_prediction = np.asarray(noise_prediction)
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = np.asarray(ground_truth)
                  
        #forward over all metrics
        return {name: func(real_prediction, noise_prediction, ground_truth) for (name, func) in self.metric_dict.items()}

    def get_metric_types(self):
        """return list of metric types inside collection

        Returns
        -------
        list of strings
        """
        return list(self.metric_dict.keys())
    