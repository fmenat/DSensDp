from typing import Any
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,roc_auc_score, matthews_corrcoef, average_precision_score

import numpy as np

# ALL CLASSIFICATION METRICS HAS (y_pre, y_true) order!!
class OverallAccuracy():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(y_pred == y_true)
	
class AverageAccuracy():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return balanced_accuracy_score(y_true, y_pred)
	
class F1Score():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
		return f1_score(y_true, y_pred, average=self.average)

class Precision():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
	    return precision_score(y_true, y_pred, average=self.average)
    
class Recall():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
	    return recall_score(y_true, y_pred, average=self.average)
    
class Kappa():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return cohen_kappa_score(y_true, y_pred)
	
class MCC():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return matthews_corrcoef(y_true, y_pred)
	
class ConfusionMatrix():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return confusion_matrix(y_true, y_pred)
	
class Get_Data():
	def __init__(self, task_type="classification", which="true"):
		self.which = which
		self.task_type = task_type

	def __call__(self, y_pred,y_true):
		data_check = y_true if self.which == "true" else y_pred
		if self.task_type == "classification":
			labels_n = np.unique(y_true)
			return [np.sum(data_check==v) for v in labels_n]
		elif self.task_type == "multilabel":
			labels_n = np.arange(y_true.shape[-1])
			return [ data_check[:,v].sum(axis=0) for v in labels_n]
	

#METRICS FOR GIVING PROBABILITIES AS OUTPUT
class ROC_AUC():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) > 1 and np.max(y_true) == 1:
			y_pred = y_pred[:,1]
		return roc_auc_score(y_true, y_pred, average=self.average, multi_class="ovr")
	
class mAP():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
		return average_precision_score(y_true, y_pred, average=self.average)
	
class CatEntropy(): #normalized
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) > 1:
			K = y_pred.shape[1]
			y_pred = np.clip(y_pred, 1e-10, 1.0)
			entropy = - (y_pred * np.log(y_pred)).sum(axis=-1) / np.log(K)
			return entropy.mean(axis=0)
    
class P_max():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) == 2:
			N, K = y_pred.shape
			p_max_x =  np.max(y_pred, axis =-1)
			return p_max_x.mean(axis=0)
    
class LogP(): #un-normalized
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) == 2:
			N, K = y_pred.shape
			y_pred = np.clip(y_pred, 1e-10, 1.0)
			return ( np.log(y_pred[np.arange(N), y_true]) ).mean(axis=0)
    
class P_true():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) == 2:
			N, K = y_pred.shape
			return ( y_pred[np.arange(N), y_true] ).mean(axis=0)
