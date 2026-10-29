import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
class QRLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(QRLoss, self).__init__()
        self.reduction = reduction

    def forward(self, query_similarity, reference_similarity):
        loss = torch.norm(query_similarity - reference_similarity, p=2, dim=-1) ** 2
        if self.reduction == "mean":
            return loss.mean().unsqueeze(0)  
        elif self.reduction == "sum":
            return loss.sum().unsqueeze(0)  
        else:
            return loss  

def calculate_alpha(class_counts, normalize=True):

    if not isinstance(class_counts, torch.Tensor):
        class_counts = torch.tensor(class_counts, dtype=torch.float32)

    alpha = 1.0 / class_counts


    if normalize:
        alpha /= alpha.sum()

    return alpha

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets.to("cpu")]
            focal_loss = alpha_t.to("cuda") * focal_loss.to("cuda")

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DynamicBCEWithLogitsLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(DynamicBCEWithLogitsLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):

        num_positive = targets.sum(dim=0)
        num_negative = targets.size(0) - num_positive


        pos_weight = num_negative / (num_positive + self.epsilon)


        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


        loss = criterion(logits, targets)

        return loss

class PenaltyForOtherClasses(nn.Module):
    def __init__(self, penalty_weight=1.0):
        super(PenaltyForOtherClasses, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):

        penalty_mask = targets[:, 0] == 1  

        if penalty_mask.sum() > 0: 
            logits_filtered = logits[penalty_mask]  
            penalty = logits_filtered[:, 1:].sum(dim=1).mean() 
        else:
            penalty = 0

        total_loss = self.penalty_weight * penalty
        return total_loss



class PenaltyForMissedClassZero(nn.Module):
    def __init__(self, penalty_weight=1.0):
        super(PenaltyForMissedClassZero, self).__init__()
        self.penalty_weight = penalty_weight
    def forward(self, logits, targets):
       
        penalty_mask = targets[:, 0] == 1  
        if penalty_mask.sum() > 0:  
            logits_filtered = logits[penalty_mask] 
            penalty = (1 - torch.sigmoid(logits_filtered[:, 0])).mean()  
        else:
            penalty = 0

        total_loss = self.penalty_weight * penalty
        return total_loss
def load_class_frequencies(csv_path):

    df = pd.read_csv(csv_path)
    try:
        class_counts = df.groupby('diagnosis')['count'].sum()
    except KeyError:  
        class_counts = df.groupby('level')['count'].sum()
    
    return class_counts
class Penalty(nn.Module):
    def __init__(self, loss, penalty_weight=1.0):
        super(Penalty, self).__init__()
        self.penalty_weight = penalty_weight
        self.loss_main = loss
        self.loss_penalty_1 = PenaltyForOtherClasses()


    def forward(self, logits, targets):

        loss = self.loss_main(logits, targets)

        penalty_1 = self.loss_penalty_1(logits, targets)

        return loss + penalty_1

class AdaptiveLDAMLoss(nn.Module):

    def __init__(self, cls_num_list):
        super(AdaptiveLDAMLoss, self).__init__()

        cls_num_tensor = torch.tensor(cls_num_list, dtype=torch.float)
        m_list = 1.0 / torch.pow(cls_num_tensor, 0.25)
        self.register_buffer('m_list', m_list)
    
    def forward(self, logits, target):


        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        

        margin = self.m_list.unsqueeze(0).to(logits.device)
 
        margin = torch.where(index, margin, torch.zeros_like(logits))
        

        logits_adjusted = logits - margin
        
        
        s_auto = 1.0 / (torch.std(logits.detach()) + 1e-5)
        
   
        scaled_logits = s_auto * logits_adjusted
        loss = F.cross_entropy(scaled_logits, target)
        return loss
def compute_class_weights_from_csv(csv_file):



    df = pd.read_csv(csv_file)
    

    try:
        class_counts = df.groupby('diagnosis')['count'].sum()
    except KeyError:  
        class_counts = df.groupby('level')['count'].sum()
    

    total_count = class_counts.sum()
    num_classes = len(class_counts)

    print("클래스별 샘플 개수:")
    print(class_counts)
    print()


    balanced_weights = total_count / (num_classes * class_counts)


    weights_sorted = balanced_weights.sort_index().values
    weights_tensor = torch.tensor(weights_sorted, dtype=torch.float).cuda()

    print("클래스별 가중치 (Balanced Weight):")
    for cls_name, w_val in zip(balanced_weights.sort_index().index, weights_sorted):
        print(f"  Class {cls_name}: {w_val:.4f}")
    
    return weights_tensor

def get_loss_function(loss_function, **kwargs):

    if loss_function == "cross_entropy":
        data = kwargs.get("data", None)

        if data == "1":
            print("data_number : ", data)
            
            csv_file = r""
            return nn.CrossEntropyLoss(weight=compute_class_weights_from_csv(csv_file))

        elif data == "2":
            print("data_number : ", data)
            csv_file = r""
            return nn.CrossEntropyLoss(weight=compute_class_weights_from_csv(csv_file))
        else:
            print("data_number : ", data)
            return nn.CrossEntropyLoss()
            
    elif loss_function == "mse":
        return nn.MSELoss()
    elif loss_function == "bce_with_logits":

        dynamic = kwargs.get("dynamic", False)
        if dynamic:
            return DynamicBCEWithLogitsLoss()
        else:
            pos_weight = kwargs.get("pos_weight", None)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_function == "AdaptiveLDAMLoss":
        data = kwargs.get("data", None)

        if data == "1":

            cls_num_list = kwargs.get("cls_num_list", load_class_frequencies(r""))
        elif data == "2":

            cls_num_list = kwargs.get("cls_num_list", load_class_frequencies(r""))
        elif data == "3":
 
            cls_num_list = kwargs.get("cls_num_list", load_class_frequencies(r""))
        
        return AdaptiveLDAMLoss(cls_num_list = cls_num_list)
    elif loss_function == "focal":
        data = kwargs.get("data", None)

        if data == "1":
            gamma = kwargs.get("gamma", 2)
            alpha = kwargs.get("alpha", compute_class_weights_from_csv(r""))
        elif data == "2":
            gamma = kwargs.get("gamma", 2)
            alpha = kwargs.get("alpha", compute_class_weights_from_csv(r""))
        elif data == "3":
            gamma = kwargs.get("gamma", 2)
            alpha = kwargs.get("alpha", compute_class_weights_from_csv(r""))
        
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif loss_function == "dice":
        smooth = kwargs.get("smooth", 1)
        return DiceLoss(smooth=smooth)
    
    elif loss_function == "cross_entropy_1":
        return Penalty( loss = nn.CrossEntropyLoss() )
    elif loss_function == "mse":
        return Penalty( loss = nn.MSELoss() )
    elif loss_function == "bce_with_logits_1":

        dynamic = kwargs.get("dynamic", False)
        if dynamic:
            return Penalty( loss = DynamicBCEWithLogitsLoss() )
        else:
            pos_weight = kwargs.get("pos_weight", None)
            return Penalty( loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) )
    elif loss_function == "focal_1":
        gamma = kwargs.get("gamma", 2)
        alpha = kwargs.get("alpha", 0.25)
        return Penalty( loss = FocalLoss(gamma=gamma, alpha=alpha) )
    elif loss_function == "dice_1":
        smooth = kwargs.get("smooth", 1)
        return Penalty( loss = DiceLoss(smooth=smooth) )
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
