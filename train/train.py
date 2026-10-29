from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
from util.utils import *
import os
from network.loss import *
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def _restore_fixed_prompts(model):
    prompt_pool = model.module.get_prompt_pool() if hasattr(model, "module") else model.get_prompt_pool()
    if hasattr(prompt_pool, "restore_fixed_prompts"):
        prompt_pool.restore_fixed_prompts()

def train_step1( model, train_loader, optimizer, scheduler, criterion, device, epoch, lambda_qr=0.001):
    model.train()
    total_loss = 0
    
    train_loader = tqdm(train_loader, desc=f"Training lr: {optimizer.param_groups[0]['lr']:.6f}", leave=False)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, suloss = model(images)
        
        ce_loss = criterion(logits, labels) - (lambda_qr*suloss.sum())

        loss = ce_loss
        loss.backward()
        optimizer.step()
        _restore_fixed_prompts(model)

        total_loss += loss.item()

        train_loader.set_postfix(loss=loss.item())

    scheduler.step()

    return total_loss / len(train_loader)

def train_step2(model, train_loader, optimizer, scheduler, criterion, qr_criterion, device, epoch, lambda_qr=0.001):
    model.train()
    total_loss = 0
    train_loader = tqdm(train_loader, desc=f"Training lr: {optimizer.param_groups[0]['lr']:.6f}", leave=False)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, qr_loss = model(images)

        ce_loss = criterion(logits, labels)

        qr_loss = qr_loss.mean()

        loss = ce_loss - (lambda_qr * qr_loss.sum())
        loss.backward()
        optimizer.step()
        _restore_fixed_prompts(model)

        total_loss += loss.item()

        train_loader.set_postfix(loss=loss.item())

    scheduler.step()
    return total_loss / len(train_loader)

def validate_step1(model, val_loader, device, criterion,metrix=None, lambda_qr=0.001):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits,suloss = model(images)
            loss = criterion(logits, labels) - (lambda_qr * suloss.sum())
            val_loss += loss.item()
            metrix.update(logits, labels)


    return val_loss / len(val_loader), metrix.compute()

def validate_step2( model, val_loader, device, criterion,metrix=None, lambda_qr=0.001):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, suloss = model(images)
            loss = criterion(logits, labels) - (lambda_qr * suloss)
            val_loss += loss.item()
            metrix.update(logits, labels)


    return val_loss / len(val_loader), metrix.compute()

from sklearn.metrics import classification_report

def test_step(model, test_loader, device, criterion, save_dir, num_classes,
               metrix=None, class_names=None, threshold=0.5,data_name=None, lambda_qr=0.001):

    model.eval()
    test_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing Step 1", leave=False):
            images, labels = images.to(device), labels.to(device) 

            logits,suloss = model(images)        
            loss = criterion(logits, labels) - ( lambda_qr * suloss.sum())
            test_loss += loss.item()


            if metrix is not None:
                metrix.update(logits, labels)

            probs = torch.softmax(logits, dim=1, dtype=None)
            _, probs = torch.max(probs, 1)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu().long()) 

    avg_loss = test_loss / len(test_loader)


    all_preds = torch.cat(all_preds, dim=0).numpy()   
    all_labels = torch.cat(all_labels, dim=0).numpy() 

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]


    report_str = classification_report(
        all_labels,    
        all_preds,      
        target_names=class_names,
        zero_division=0
    )
    print(f"\n Classification Report :\n", report_str)


    if save_dir is not None:
        report_path = f"{save_dir}/test_{data_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"[Info] Classification report saved at: {report_path}")


    conf_mat = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation='vertical')  
    ax.set_title("Confusion Matrix (Step 1)")
    plt.tight_layout()

    if save_dir is not None:
        cm_path = os.path.join(save_dir, f"test_{data_name}_classification_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"[Info] Confusion matrix saved at: {cm_path}")

    plt.close()  

    if metrix is not None:
        metric_result = metrix.compute()
    else:
        metric_result = {}
    
    return avg_loss, metric_result,str(data_name)


def test_step2(model, test_loader, device, criterion, save_dir, num_classes,
               metrix=None, class_names=None, threshold=0.5,data_name=None, lambda_qr=0.001):

    model.eval()
    test_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing Step 1", leave=False):
            images, labels = images.to(device), labels.to(device)  

            logits,suloss  = model(images)         
            loss = criterion(logits, labels) - (lambda_qr*suloss.sum())
            test_loss += loss.item()


            if metrix is not None:
                metrix.update(logits, labels)

            probs = torch.softmax(logits, dim=1, dtype=None)
            _, probs = torch.max(probs, 1)


            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu().long())  

    avg_loss = test_loss / len(test_loader)


    all_preds = torch.cat(all_preds, dim=0).numpy()   
    all_labels = torch.cat(all_labels, dim=0).numpy() 

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]


    report_str = classification_report(
        all_labels,     
        all_preds,      
        target_names=class_names,
        zero_division=0
    )
    print(f"\n Classification Report :\n", report_str)

    if save_dir is not None:
        report_path = f"{save_dir}/test_{data_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"[Info] Classification report saved at: {report_path}")

    conf_mat = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation='vertical')  
    ax.set_title("Confusion Matrix (Step 1)")
    plt.tight_layout()

    if save_dir is not None:
        cm_path = os.path.join(save_dir, f"test_{data_name}_classification_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"[Info] Confusion matrix saved at: {cm_path}")

    plt.close() 

    if metrix is not None:
        metric_result = metrix.compute()
    else:
        metric_result = {}
    
    return avg_loss, metric_result,str(data_name)