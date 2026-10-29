import os
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from typing import Any
import torch.nn as nn
import torch
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC
import torch.optim as optim
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassAUROC,
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F

class MulticlassMetrics:
    def __init__(self, num_labels,device):
        """
        Args:
            num_labels (int): Number of labels/classes.
        """
        self.num_labels = num_labels
        self.predictions = []
        self.targets = []

    def update(self, preds, targets):
        """
        Update predictions and targets.
        
        Args:
            preds (torch.Tensor or np.ndarray): Logits (raw model outputs).
            targets (torch.Tensor or np.ndarray): True labels (0 to num_labels-1).
        """

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy() 
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()  

        probs = F.softmax(torch.tensor(preds), dim=1).numpy()  
        pred_classes = np.argmax(probs, axis=1)  

        self.predictions.extend(pred_classes)
        self.targets.extend(targets)

    def compute(self):

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        precision = precision_score(targets, predictions, average='macro', zero_division=0)
        recall = recall_score(targets, predictions, average='macro', zero_division=0)
        f1 = f1_score(targets, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(targets, predictions)


        try:
            roc_auc = roc_auc_score(np.eye(self.num_labels)[targets], 
                                    np.eye(self.num_labels)[predictions], 
                                    average='macro', multi_class='ovr')
        except ValueError:
            roc_auc = float('nan')  
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        }

    def reset(self):
        """
        Reset all stored predictions and targets.
        """
        self.predictions = []
        self.targets = []



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path)
    plt.close()

def create_save_directory(base_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
import torch
import torch.nn as nn
import traceback
from typing import Any

def load_model_weights(model: nn.Module, checkpoint_path: str, device: Any) -> None:

    try:
       
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except TypeError:
            print("weights_only 인자를 지원하지 않는 버전의 PyTorch입니다. 기본 동작으로 진행합니다.")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        

        if "model_state_dict" in checkpoint:
            checkpoint_state_dict = checkpoint["model_state_dict"]
        else:
            raise KeyError("The checkpoint does not contain 'model_state_dict'.")

        is_ddp = isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))

        if is_ddp:
            if not any(key.startswith("module.") for key in checkpoint_state_dict.keys()):
                print("DataParallel 모델이지만 checkpoint 키에 'module.' 접두사가 없습니다. 접두사를 추가합니다.")
                checkpoint_state_dict = {f"module.{k}": v for k, v in checkpoint_state_dict.items()}
        else:
            if any(key.startswith("module.") for key in checkpoint_state_dict.keys()):
                print("일반 모델이지만 checkpoint 키에 'module.' 접두사가 있습니다. 접두사를 제거합니다.")
                checkpoint_state_dict = {k.replace("module.", ""): v for k, v in checkpoint_state_dict.items()}

        model_state_dict = model.state_dict()
        
    
        matched_state_dict = {}
        for key, ckpt_param in checkpoint_state_dict.items():
            if key in model_state_dict:
                model_param = model_state_dict[key]
                if model_param.size() == ckpt_param.size():

                    matched_state_dict[key] = ckpt_param
                else:

                    if model_param.dim() > 0 and ckpt_param.dim() > 0:

                        if model_param.size()[1:] == ckpt_param.size()[1:] and model_param.size(0) >= ckpt_param.size(0):
                            print(f"키 '{key}': 모델 파라미터는 {model_param.size()}, checkpoint 파라미터는 {ckpt_param.size()} -> 앞부분만 업데이트합니다.")

                            new_param = model_param.clone()
                            new_param[:ckpt_param.size(0)] = ckpt_param
                            matched_state_dict[key] = new_param
                        else:
                            print(f"Skipping '{key}': shape mismatch (checkpoint: {ckpt_param.size()}, model: {model_param.size()})")
                    else:
                        print(f"Skipping '{key}': shape mismatch (checkpoint: {ckpt_param.size()}, model: {model_param.size()})")
            else:
                print(f"Skipping '{key}': key not found in model.")
        

        missing_keys = set(model_state_dict.keys()) - set(matched_state_dict.keys())
        if missing_keys:
            print(f"Warning: 다음 key들은 checkpoint에 없거나 shape mismatch로 인해 업데이트되지 않았습니다: {missing_keys}")
        
        # 모델 state_dict 업데이트 후 로드
        model_state_dict.update(matched_state_dict)
        model.load_state_dict(model_state_dict)
        print("Model weights loaded successfully!")
    
    except KeyError as e:
        print(f"KeyError during loading: {e}")
        print(traceback.format_exc())
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())

        
def save_detailed_metrics(y_true, y_pred, save_dir,step="step1"):

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(save_dir, str(step)+"_"+"confusion_matrix_detailed.png"))
    plt.close()


    report = classification_report(y_true, y_pred, target_names=None, digits=4)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

def save_roc_curve(y_true, y_scores, num_classes, save_path):

    try:
       
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        if y_scores.ndim != 2:
            raise ValueError("y_scores must be a 2D array with shape (n_samples, n_classes).")

        if num_classes == 2:
            _plot_binary_roc(y_true, y_scores, save_path)
        elif num_classes > 2:
            _plot_multiclass_roc(y_true, y_scores, num_classes, save_path)
        else:
            raise ValueError("Number of classes must be at least 2.")

        print(f"ROC Curve saved to {save_path}")

    except Exception as e:
        print(f"Error in saving ROC Curve: {e}")


def _plot_binary_roc(y_true, y_scores, save_path):
    """Helper function to plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Binary)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def _plot_multiclass_roc(y_true, y_scores, num_classes, save_path):
    """Helper function to plot ROC curve for multi-class classification."""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))  

    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.4f})', color=colors[i])
        except ValueError as e:
            print(f"Could not compute ROC for class {i}: {e}")

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multi-class)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_path)
    plt.close()



def get_optimizer(optimizer_name, model_parameters, **kwargs):

    lr = kwargs.get("lr", 1e-3) 
    weight_decay = kwargs.get("weight_decay", 0) 
    
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "rmsprop":
        momentum = kwargs.get("momentum", 0)
        return optim.RMSprop(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adagrad":
        return optim.Adagrad(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adadelta":
        return optim.Adadelta(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
def hamming_loss(y_true, y_pred):

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"
    return torch.mean((y_true != y_pred).float()).item()

def precision_at_k(y_true, y_pred_scores, k):
    
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred_scores, np.ndarray):
        y_pred_scores = torch.tensor(y_pred_scores)
    
    n_samples = y_true.size(0)
    precision_scores = []

    for i in range(n_samples):
        top_k_indices = torch.topk(y_pred_scores[i], k=k).indices  
        relevant = y_true[i, top_k_indices].sum().item()  
        precision_scores.append(relevant / k)

    return torch.tensor(precision_scores).mean().item()

def recall_at_k(y_true, y_pred_scores, k):

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred_scores, np.ndarray):
        y_pred_scores = torch.tensor(y_pred_scores)
    
    n_samples = y_true.size(0)
    recall_scores = []

    for i in range(n_samples):
        top_k_indices = torch.topk(y_pred_scores[i], k=k).indices  
        relevant = y_true[i, top_k_indices].sum().item()  
        total_relevant = y_true[i].sum().item()  
        if total_relevant > 0:
            recall_scores.append(relevant / total_relevant)
        else:
            recall_scores.append(0.0)

    return torch.tensor(recall_scores).mean().item()