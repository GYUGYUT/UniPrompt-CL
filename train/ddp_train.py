import os
import csv
import json
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm
from util.utils import *
from train.train import *
from network.model import *
from network.loss import QRLoss, get_loss_function
from util.scheduler import get_scheduler
from util.early_stopping import EarlyStopping
from transformers import ViTModel
import socket
import torch.distributed as dist
from dataloader.diabetic_dataloader import diabetic_create_dataloader
from dataloader.atop_dataloader import atop_create_dataloader
from torch.utils.data import Dataset, DataLoader, DistributedSampler,ConcatDataset
from dataloader.DDR_dataloader import DDR_create_dataloader
from dataloader.medical_stream_dataloader import create_stage_dataloader, get_dataset_name, get_stage_num_classes, get_stage_domains

using_device_list = [0]
def format_value(value):
    return f"{value:.4f}" if isinstance(value, (float, int)) else "N/A"

        
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _use_stage_dataloaders(args):
    stage_split_dir = getattr(args, "stage_split_dir", None)
    return (
        get_dataset_name(args) == "skin"
        or getattr(args, "cache_mode", "none") != "none"
        or stage_split_dir not in (None, "", "None")
    )


def _active_num_classes(args):
    if _use_stage_dataloaders(args):
        return get_stage_num_classes(args)
    return int(getattr(args, "num_classes", 3))


def _stage_or_legacy_loader(args, stage, split, legacy_fn):
    if _use_stage_dataloaders(args):
        return create_stage_dataloader(args, stage=stage, split=split)
    return legacy_fn()


def _legacy_test_loader(args, stage):
    if stage == 1:
        return atop_create_dataloader(args, is_train="test")
    if stage == 2:
        return DDR_create_dataloader(args, is_train="test")
    if stage == 3:
        return diabetic_create_dataloader(args, is_train="test")
    raise IndexError(f"stage {stage} out of range")


def _test_loader_for_stage(args, stage):
    return _stage_or_legacy_loader(args, stage, "test", lambda: _legacy_test_loader(args, stage))


def _domain_names(args):
    if _use_stage_dataloaders(args):
        return get_stage_domains(args)
    return ["aptos", "ddr", "diabetic"]


def _ensure_cl_matrices(args):
    total_stages = len(_domain_names(args))
    if not hasattr(args, "cl_accuracy_matrix"):
        args.cl_accuracy_matrix = [[None for _ in range(total_stages)] for _ in range(total_stages)]
    if not hasattr(args, "cl_f1_matrix"):
        args.cl_f1_matrix = [[None for _ in range(total_stages)] for _ in range(total_stages)]


def _evaluate_loader_metrics(model, loader, device, criterion, args, lambda_qr=0.001):
    model.eval()
    metric = MulticlassMetrics(num_labels=_active_num_classes(args), device=device)
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="CL metric eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, suloss = model(images)
            loss = criterion(logits, labels) - (lambda_qr * suloss.sum())
            total_loss += loss.item()
            metric.update(logits, labels)
    values = metric.compute()
    values["loss"] = total_loss / max(len(loader), 1)
    return values


def _compute_cl_summary(matrix, current_stage):
    rows = [row[:current_stage] for row in matrix[:current_stage]]
    avg_acc = sum(
        sum(rows[i][: i + 1]) / (i + 1)
        for i in range(current_stage)
    ) / current_stage

    if current_stage <= 1:
        bwt = 0.0
        avg_f = 0.0
    else:
        final_row = rows[current_stage - 1]
        bwt = sum(final_row[j] - rows[j][j] for j in range(current_stage - 1)) / (current_stage - 1)
        avg_f = sum(
            max(rows[i][j] for i in range(j, current_stage)) - final_row[j]
            for j in range(current_stage - 1)
        ) / (current_stage - 1)
    return {"AvgACC": avg_acc, "BWT": bwt, "AvgF": avg_f}


def _write_matrix_csv(path, matrix, domains):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_stage"] + domains)
        for i, row in enumerate(matrix, start=1):
            writer.writerow([i] + ["" if value is None else f"{value:.6f}" for value in row])


def _save_cl_metrics(args, root_save_dir, current_stage):
    domains = _domain_names(args)
    acc_summary = _compute_cl_summary(args.cl_accuracy_matrix, current_stage)
    f1_summary = _compute_cl_summary(args.cl_f1_matrix, current_stage)
    metrics = {
        "current_stage": current_stage,
        "domain_names": domains,
        "accuracy": acc_summary,
        "macro_f1": f1_summary,
        "accuracy_matrix": args.cl_accuracy_matrix,
        "macro_f1_matrix": args.cl_f1_matrix,
    }
    _write_matrix_csv(os.path.join(root_save_dir, "cl_accuracy_matrix.csv"), args.cl_accuracy_matrix, domains)
    _write_matrix_csv(os.path.join(root_save_dir, "cl_macro_f1_matrix.csv"), args.cl_f1_matrix, domains)
    with open(os.path.join(root_save_dir, "cl_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(root_save_dir, "cl_metrics_summary.txt"), "w") as f:
        f.write(f"current_stage: {current_stage}\n")
        for metric_name, summary in (("accuracy", acc_summary), ("macro_f1", f1_summary)):
            f.write(f"[{metric_name}]\n")
            f.write(f"AvgACC: {summary['AvgACC']:.6f}\n")
            f.write(f"BWT: {summary['BWT']:.6f}\n")
            f.write(f"AvgF: {summary['AvgF']:.6f}\n")
    print(f"[CL metrics] saved to {root_save_dir}")


def _update_cl_metrics(args, model, criterion, device, root_save_dir, current_stage):
    _ensure_cl_matrices(args)
    for eval_stage in range(1, current_stage + 1):
        loader = _test_loader_for_stage(args, eval_stage)
        values = _evaluate_loader_metrics(model, loader, device, criterion, args, lambda_qr=args.lambda_qr)
        args.cl_accuracy_matrix[current_stage - 1][eval_stage - 1] = float(values["accuracy"])
        args.cl_f1_matrix[current_stage - 1][eval_stage - 1] = float(values["f1_score"])
        print(
            f"[CL metrics] train_stage={current_stage}, eval_stage={eval_stage}, "
            f"acc={values['accuracy']:.4f}, f1={values['f1_score']:.4f}"
        )
    _save_cl_metrics(args, root_save_dir, current_stage)

def ddp_train_step1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.save_path
    metrix_val = MulticlassMetrics(num_labels = _active_num_classes(args),device=device)
    metrix_test = MulticlassMetrics(num_labels = _active_num_classes(args),device=device)
    if args.train_mode != 2:
        args.save_path = save_dir
        save_dir = os.path.join(save_dir,str("step1"))


    prompt_pool = PromptPool(num_prompts=args.num_prompts, prompt_dim=args.prompt_dim,fixed_ratio=0).to(device)
    Train_vit = ViTWithCustomHead(num_classes=_active_num_classes(args),num_layers=5, prompt_pool = prompt_pool)
    Train_vit.to(device)

    model = nn.DataParallel(Train_vit,device_ids=using_device_list)


    create_directory_if_not_exists(save_dir)
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        for arg, value in vars(args).items():
            log_file.write(f"{arg}: {value}\n")
        log_file.write(str(model) + "\n")


    train_loader = _stage_or_legacy_loader(args, 1, "train", lambda: atop_create_dataloader(args , is_train="train"))
    val_loader = _stage_or_legacy_loader(args, 1, "val", lambda: atop_create_dataloader(args , is_train="val"))
    test_loader = _stage_or_legacy_loader(args, 1, "test", lambda: atop_create_dataloader(args , is_train="test"))





    optimizer = get_optimizer(
                                args.optimizer, 
                                model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay
                            )
    scheduler = get_scheduler(optimizer, args.scheduler, args)
    criterion = get_loss_function(args.loss_function,data="5", dynamic=True)
    

    early_stopping = EarlyStopping(patience=args.patience, mode="min" if args.save_criteria == "loss" else "max")
    best_score = None


    for epoch in range(args.epochs):
        train_loss  = train_step1( model, train_loader, optimizer, scheduler, criterion, device, epoch, args.lambda_qr)
        val_loss,val_metrix  = validate_step1( model, val_loader, device, criterion,metrix_val, lambda_qr=args.lambda_qr)

        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}"
            f"Val Precision: {val_metrix['precision']:.4f}, Val Recall: {val_metrix['recall']:.4f}, "
            f"Val F1: {val_metrix['f1_score']:.4f}, Val Accuracy: {val_metrix['accuracy']:.4f}"
        )



        with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
            log_file.write(
                f"Epoch {epoch + 1}:\n"
                f"  Train - Loss: {train_loss:.4f}"
                f"  Validation - Loss: {val_loss:.4f}, Precision: {val_metrix['precision']:.4f}, Recall: {val_metrix['recall']:.4f}, "
                f"F1: {val_metrix['f1_score']:.4f}, Accuracy: {val_metrix['accuracy']:.4f}\n"
            )

        if args.save_criteria == "loss":
            current_score = val_loss 
        elif args.save_criteria == "accuracy" :
            current_score = val_metrix['accuracy']
        elif args.save_criteria == "precision" :
            current_score = val_metrix['precision']
        elif args.save_criteria == "recall" :
            current_score = val_metrix['recall']
        elif args.save_criteria == "f1_score" :
            current_score = val_metrix['f1_score']

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_score": best_score,
            "args": vars(args),
        }
        if best_score is None or (args.save_criteria == "loss" and current_score < best_score) or (args.save_criteria == "accuracy" and current_score > best_score) or (args.save_criteria == "precision" and current_score > best_score) or (args.save_criteria == "recall" and current_score > best_score) or (args.save_criteria == "f1_score" and current_score > best_score):
            best_score = current_score
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            prompt = model.module.get_prompt_pool()
            torch.save(prompt.state_dict(), os.path.join(save_dir, "best_prompt.pth"))
            print(f"New best model saved at epoch {epoch + 1} with {args.save_criteria}: {current_score:.4f}")

        torch.save(checkpoint, os.path.join(save_dir, "last_model.pth"))
        prompt = model.module.get_prompt_pool()
        torch.save(prompt.state_dict(), os.path.join(save_dir, "last_prompt.pth"))

        early_stopping(current_score)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        else:
            print("early_stopping_count : ",early_stopping.counter)
        metrix_val.reset()
    load_model_weights(model, os.path.join(save_dir, "best_model.pth"), device)
    _update_cl_metrics(args, model, criterion, device, args.save_path, 1)
    test_loss, test_metrix,data_name  = test_step(

        model= model,
        test_loader = test_loader,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="atop")

    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"stage_1_Test Loss_{data_name}: {format_value(test_loss)}, "
            f"stage_1_Test Precision_{data_name}: {format_value(test_metrix['precision'])}, "
            f"stage_1_Test Recall_{data_name}: {format_value(test_metrix['recall'])}, "
            f"stage_1_Test F1_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"stage_1_Test Accuracy_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"stage_1_Test ROC AUC_{data_name}: {format_value(test_metrix['roc_auc'])}\n"


        )


    metrix_test.reset()
    test_loader2 = _stage_or_legacy_loader(args, 2, "test", lambda: DDR_create_dataloader(args , is_train="test"))
    test_loss, test_metrix, data_name  = test_step(
        model= model,
        test_loader = test_loader2,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="DDR")
    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"stage_1_target_Test Loss_{data_name}: {format_value(test_loss)}, "
            f"stage_1_target_Test Precision_{data_name}: {format_value(test_metrix['precision'])}, "
            f"stage_1_target_Test Recall_{data_name}: {format_value(test_metrix['recall'])}, "
            f"stage_1_target_Test F1_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"stage_1_target_Test Accuracy_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"stage_1_target_Test ROC AUC_{data_name}: {format_value(test_metrix['roc_auc'])}\n"

        )

    metrix_test.reset()

    test_loader2 = _stage_or_legacy_loader(args, 3, "test", lambda: diabetic_create_dataloader(args , is_train="test"))
    test_loss, test_metrix, data_name  = test_step(
        model= model,
        test_loader = test_loader2,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="dia")
    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"stage_1_target_Test Loss_{data_name}: {format_value(test_loss)}, "
            f"stage_1_target_Test Precision_{data_name}: {format_value(test_metrix['precision'])}, "
            f"stage_1_target_Test Recall_{data_name}: {format_value(test_metrix['recall'])}, "
            f"stage_1_target_Test F1_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"stage_1_target_Test Accuracy_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"stage_1_target_Test ROC AUC_{data_name}: {format_value(test_metrix['roc_auc'])}\n"

        )

    metrix_test.reset()
    


    

def ddp_train_step2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.save_path
    metrix_val = MulticlassMetrics(num_labels = _active_num_classes(args),device=device)
    metrix_test = MulticlassMetrics(num_labels = _active_num_classes(args),device=device)

    if( args.train_mode == 0 ):
        prompt_pool = PromptPool(num_prompts=args.num_prompts + args.expand_per_stage, prompt_dim=args.prompt_dim, fixed_ratio=args.num_prompts).to(device)
        model = ViTWithCustomHead(num_classes=_active_num_classes(args),num_layers=5, prompt_pool = prompt_pool).to(device)
        load_model_weights(model, os.path.join(save_dir,str("step1"),str("best_model.pth")), device)
        
        model.prompt_pool.freeze_fixed_prompts()
        model = nn.DataParallel(model,device_ids=using_device_list)
        

    
    if args.train_mode != 2:
        args.save_path = save_dir
        save_dir = os.path.join(save_dir,str("step2"))
        create_directory_if_not_exists(save_dir)

    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        for arg, value in vars(args).items():
            log_file.write(f"{arg}: {value}\n")
        log_file.write(str(model) + "\n")



    train_loader = _stage_or_legacy_loader(args, 2, "train", lambda: DDR_create_dataloader(args , is_train="train"))
    val_loader = _stage_or_legacy_loader(args, 2, "val", lambda: DDR_create_dataloader(args , is_train="val"))
    test_loader = _stage_or_legacy_loader(args, 2, "test", lambda: DDR_create_dataloader(args , is_train="test"))

    test_loader2 = _stage_or_legacy_loader(args, 1, "test", lambda: atop_create_dataloader(args, is_train="test"))

    optimizer = get_optimizer(
                                args.optimizer, 
                                model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay
                            )
    scheduler = get_scheduler(optimizer, args.scheduler, args)
    criterion = get_loss_function(args.loss_function,data="5", dynamic=True )
    

    early_stopping = EarlyStopping(patience=args.patience, mode="min" if args.save_criteria == "loss" else "max")
    best_score = None

    for epoch in range(args.epochs):
        train_loss  = train_step1( model, train_loader, optimizer, scheduler, criterion, device, epoch, args.lambda_qr)
        val_loss, val_metrix  = validate_step1( model, val_loader, device, criterion,metrix_val, lambda_qr=args.lambda_qr)
        
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f},  "
            f"Val Precision: {val_metrix['precision']:.4f}, Val Recall: {val_metrix['recall']:.4f}, "
            f"Val F1: {val_metrix['f1_score']:.4f}, Val Accuracy: {val_metrix['accuracy']:.4f}"
        )


        # Save logs to file
        with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
            log_file.write(
                f"Epoch {epoch + 1}:\n"
                f"  Train - Loss: {train_loss:.4f},"
                f"  Validation - Loss: {val_loss:.4f}, Precision: {val_metrix['precision']:.4f}, Recall: {val_metrix['recall']:.4f}, "
                f"F1: {val_metrix['f1_score']:.4f}, Accuracy: {val_metrix['accuracy']:.4f}\n"
            )

        # 모델 저장
        if args.save_criteria == "loss":
            current_score = val_loss 
        elif args.save_criteria == "accuracy" :
            current_score = val_metrix['accuracy']
        elif args.save_criteria == "precision" :
            current_score = val_metrix['precision']
        elif args.save_criteria == "recall" :
            current_score = val_metrix['recall']
        elif args.save_criteria == "f1_score" :
            current_score = val_metrix['f1_score']

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_score": best_score,
            "args": vars(args),
        }
        if best_score is None or (args.save_criteria == "loss" and current_score < best_score) or (args.save_criteria == "accuracy" and current_score > best_score) or (args.save_criteria == "precision" and current_score > best_score) or (args.save_criteria == "recall" and current_score > best_score) or (args.save_criteria == "f1_score" and current_score > best_score):
            
            best_score = current_score
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            prompt = model.module.get_prompt_pool()
            torch.save(prompt.state_dict(), os.path.join(save_dir, "best_prompt.pth"))
            print(f"New best model saved at epoch {epoch + 1} with {args.save_criteria}: {current_score:.4f}")

        torch.save(checkpoint, os.path.join(save_dir, "last_model.pth"))
        prompt = model.module.get_prompt_pool()
        torch.save(prompt.state_dict(), os.path.join(save_dir, "last_prompt.pth"))
        

        early_stopping(current_score)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        else:
            print("early_stopping_count : ",early_stopping.counter)
        metrix_val.reset()
    try : 
        del train_loader
        del val_loader
    except:
        pass
    load_model_weights(model, os.path.join(save_dir, "best_model.pth"), device)
    _update_cl_metrics(args, model, criterion, device, args.save_path, 2)

    test_loss, test_metrix,data_name = test_step2(
        model= model,
        test_loader = test_loader,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="DDR")
    print("dataset_one")
    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"Test Loss_step_{data_name}: {format_value(test_loss)}, "
            f"Test Precision_step_{data_name}: {format_value(test_metrix['precision'])}, "
            f"Test Recall_step_{data_name}: {format_value(test_metrix['recall'])}, "
            f"Test F1_step_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"Test Accuracy_step_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"Test ROC AUC_step_{data_name}: {format_value(test_metrix['roc_auc'])}\n"


        )

    metrix_test.reset()


    test_loss, test_metrix,data_name= test_step2( 
        model= model,
        test_loader = test_loader2,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="atops")
    print("dataset_Two")

    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"Test Loss_step_{data_name}: {format_value(test_loss)}, "
            f"Test Precision_step_{data_name}: {format_value(test_metrix['precision'])}, "
            f"Test Recall_step_{data_name}: {format_value(test_metrix['recall'])}, "
            f"Test F1_step_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"Test Accuracy_step_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"Test ROC AUC_step_{data_name}: {format_value(test_metrix['roc_auc'])}\n"

        )

        
    try:
        del test_loader2, train_loader,val_loader,test_loader
    except:
        pass

    metrix_test.reset()

    test_loader2 = _stage_or_legacy_loader(args, 3, "test", lambda: diabetic_create_dataloader(args , is_train="test"))
    test_loss, test_metrix, data_name  = test_step(
        model= model,
        test_loader = test_loader2,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="dia")
    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"stage_2_target_Test Loss_{data_name}: {format_value(test_loss)}, "
            f"stage_2_target_Test Precision_{data_name}: {format_value(test_metrix['precision'])}, "
            f"stage_2_target_Test Recall_{data_name}: {format_value(test_metrix['recall'])}, "
            f"stage_2_target_Test F1_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"stage_2_target_Test Accuracy_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"stage_2_target_Test ROC AUC_{data_name}: {format_value(test_metrix['roc_auc'])}\n"

        )


    metrix_test.reset()


def ddp_train_step3(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.save_path
    metrix_val = MulticlassMetrics(num_labels = _active_num_classes(args),device=device)
    metrix_test = MulticlassMetrics(num_labels = _active_num_classes(args),device=device)

    # 모델 초기화
    if( args.train_mode == 0 ):

        prompt_pool = PromptPool(num_prompts=args.num_prompts + 2 * args.expand_per_stage, prompt_dim=args.prompt_dim, fixed_ratio=args.num_prompts + args.expand_per_stage).to(device)
        # load_model_weights_prompt(prompt_pool, os.path.join(save_dir,str("step2"),str("best_prompt.pth")), device)
        model = ViTWithCustomHead(num_classes=_active_num_classes(args),num_layers=5, prompt_pool = prompt_pool).to(device)
        load_model_weights(model, os.path.join(save_dir,str("step2"),str("best_model.pth")), device)
        
        model.prompt_pool.freeze_fixed_prompts()
        model = nn.DataParallel(model,device_ids=using_device_list)



    if args.train_mode != 2:
        args.save_path = save_dir
        save_dir = os.path.join(save_dir,str("step3"))
        create_directory_if_not_exists(save_dir)
        
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        for arg, value in vars(args).items():
            log_file.write(f"{arg}: {value}\n")
        log_file.write(str(model) + "\n")



    test_loader2 = _stage_or_legacy_loader(args, 1, "test", lambda: atop_create_dataloader(args , is_train="test"))


    train_loader = _stage_or_legacy_loader(args, 3, "train", lambda: diabetic_create_dataloader(args , is_train="train"))
    val_loader = _stage_or_legacy_loader(args, 3, "val", lambda: diabetic_create_dataloader(args , is_train="val"))
    test_loader = _stage_or_legacy_loader(args, 3, "test", lambda: diabetic_create_dataloader(args , is_train="test"))


    test_loader3 = _stage_or_legacy_loader(args, 2, "test", lambda: DDR_create_dataloader(args , is_train="test"))


    optimizer = get_optimizer(
                                args.optimizer, 
                                model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay
                            )
    scheduler = get_scheduler(optimizer, args.scheduler, args)
    criterion = get_loss_function(args.loss_function,data="5", dynamic=True )
    

    early_stopping = EarlyStopping(patience=args.patience, mode="min" if args.save_criteria == "loss" else "max")
    best_score = None


    for epoch in range(args.epochs):
        train_loss  = train_step1( model, train_loader, optimizer, scheduler, criterion, device, epoch, args.lambda_qr)
        val_loss, val_metrix  = validate_step1( model, val_loader, device, criterion,metrix_val, lambda_qr=args.lambda_qr)
        
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f},  "
            f"Val Precision: {val_metrix['precision']:.4f}, Val Recall: {val_metrix['recall']:.4f}, "
            f"Val F1: {val_metrix['f1_score']:.4f}, Val Accuracy: {val_metrix['accuracy']:.4f}"
        )



        # Save logs to file
        with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
            log_file.write(
                f"Epoch {epoch + 1}:\n"
                f"  Train - Loss: {train_loss:.4f},"
                f"  Validation - Loss: {val_loss:.4f}, Precision: {val_metrix['precision']:.4f}, Recall: {val_metrix['recall']:.4f}, "
                f"F1: {val_metrix['f1_score']:.4f}, Accuracy: {val_metrix['accuracy']:.4f}\n"
            )

        # 모델 저장
        if args.save_criteria == "loss":
            current_score = val_loss 
        elif args.save_criteria == "accuracy" :
            current_score = val_metrix['accuracy']
        elif args.save_criteria == "precision" :
            current_score = val_metrix['precision']
        elif args.save_criteria == "recall" :
            current_score = val_metrix['recall']
        elif args.save_criteria == "f1_score" :
            current_score = val_metrix['f1_score']

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_score": best_score,
            "args": vars(args),
        }
        if best_score is None or (args.save_criteria == "loss" and current_score < best_score) or (args.save_criteria == "accuracy" and current_score > best_score) or (args.save_criteria == "precision" and current_score > best_score) or (args.save_criteria == "recall" and current_score > best_score) or (args.save_criteria == "f1_score" and current_score > best_score):
            
            best_score = current_score
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            prompt = model.module.get_prompt_pool()
            torch.save(prompt.state_dict(), os.path.join(save_dir, "best_prompt.pth"))
            print(f"New best model saved at epoch {epoch + 1} with {args.save_criteria}: {current_score:.4f}")

        torch.save(checkpoint, os.path.join(save_dir, "last_model.pth"))
        prompt = model.module.get_prompt_pool()
        torch.save(prompt.state_dict(), os.path.join(save_dir, "last_prompt.pth"))
        


        early_stopping(current_score)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        else:
            print("early_stopping_count : ",early_stopping.counter)
        metrix_val.reset()
    try : 
        del train_loader
        del val_loader
    except:
        pass

    load_model_weights(model, os.path.join(save_dir, "best_model.pth"), device)
    _update_cl_metrics(args, model, criterion, device, args.save_path, 3)

    test_loss, test_metrix,data_name = test_step2( 
        model= model,
        test_loader = test_loader,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="dia")
    print("dataset_one")
    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"Test Loss_step_{data_name}: {format_value(test_loss)}, "
            f"Test Precision_step_{data_name}: {format_value(test_metrix['precision'])}, "
            f"Test Recall_step_{data_name}: {format_value(test_metrix['recall'])}, "
            f"Test F1_step_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"Test Accuracy_step_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"Test ROC AUC_step_{data_name}: {format_value(test_metrix['roc_auc'])}\n"


        )

    metrix_test.reset()


    test_loss, test_metrix , data_name= test_step2( 
        model= model,
        test_loader = test_loader2,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="atops")
    print("dataset_Two")

    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"Test Loss_step_{data_name}: {format_value(test_loss)}, "
            f"Test Precision_step_{data_name}: {format_value(test_metrix['precision'])}, "
            f"Test Recall_step_{data_name}: {format_value(test_metrix['recall'])}, "
            f"Test F1_step_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"Test Accuracy_step_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"Test ROC AUC_step_{data_name}: {format_value(test_metrix['roc_auc'])}\n"

        )
        
    try:
        del test_loader2, train_loader,val_loader,test_loader
    except:
        pass

    metrix_test.reset()


    test_loss, test_metrix, data_name = test_step2(         
        model= model,
        test_loader = test_loader3,
        device = device,
        criterion = criterion,
        save_dir = save_dir,
        num_classes = _active_num_classes(args),
        metrix = metrix_test,
        data_name="ddr")
    print("dataset_3")

    print(
            f"Test_time: Test Precision: {test_metrix['precision']:.4f}, Test Recall: {test_metrix['recall']:.4f}, "
            f"Test F1: {test_metrix['f1_score']:.4f}, Test Accuracy: {test_metrix['accuracy']:.4f}, "

        )
    with open(os.path.join(save_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(
            f"Test Loss_step_{data_name}: {format_value(test_loss)}, "
            f"Test Precision_step_{data_name}: {format_value(test_metrix['precision'])}, "
            f"Test Recall_step_{data_name}: {format_value(test_metrix['recall'])}, "
            f"Test F1_step_{data_name}: {format_value(test_metrix['f1_score'])}, "
            f"Test Accuracy_step_{data_name}: {format_value(test_metrix['accuracy'])}, "
            f"Test ROC AUC_step_{data_name}: {format_value(test_metrix['roc_auc'])}\n"

        )
        
    try:
        del test_loader2, train_loader,val_loader,test_loader
    except:
        pass

    metrix_test.reset()


