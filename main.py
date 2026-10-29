import torch
import torch.multiprocessing as mp
from train.ddp_train import *
from util.utils import set_seed
import argparse
from util.utils import create_save_directory

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="OS-Prompt Training Script with DDP")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_prompts", type=int, default=500, help="Number of prompts in the initial prompt pool (L_p)")
    parser.add_argument("--prompt_dim", type=int, default=768, help="Dimension of each prompt")
    parser.add_argument("--expand_per_stage", type=int, default=100, help="Number of new prompts added to the pool at each stage after the first (psi; 20%% of L_p)")
    parser.add_argument("--wandb_project", type=str, default="OS-Prompt", help="WandB project name")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--save_criteria", type=str, default="loss", help="Criteria to save the best model")
    parser.add_argument("--scheduler", type=str, choices=["step", "multistep", "cosine", "plateau"], default="cosine", help="LR Scheduler type")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR and MultiStepLR")
    parser.add_argument("--milestones", type=int, nargs="+", default=[30, 60, 90], help="Milestones for MultiStepLR")
    parser.add_argument("--loss_function", type=str, default="cross_entropy", help="Loss function to use")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--train_mode", type=int, default=0, help="all_train : 0 , step1_train : 1, step2_train : 2")
    parser.add_argument("--step_1_model_pt_path", type=str, default=None, help="step_1_model_pt")
    parser.add_argument("--model_name", type=str, default=None, help="model_name")
    parser.add_argument("--save_path", type=str, default=None, help="save_path")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay")
    parser.add_argument("--lambda_qr", type=float, default=1e-3, help="Weight of the surrogate selection loss (lambda); the paper uses 1e-3")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")

    parser.add_argument("--dataset", choices=["dr", "skin"], default="dr", help="Medical stream to train: DR or skin-cancer")
    parser.add_argument("--stage_split_dir", type=str, default="None", help="Directory with stage{N}_{domain}_{split}.csv files")
    parser.add_argument("--medical_data_root", type=str, default="data", help="Root prepended to filepath column in stage CSVs")
    parser.add_argument("--cache_mode", choices=["none", "uint8"], default="none", help="Use uint8 tensor cache shards")
    parser.add_argument("--cache_dir", type=str, default="None", help="Tensor cache directory; defaults to dataloader/tensor_cache/<dataset>")
    parser.add_argument("--build_cache", action="store_true", help="Build missing cache shards before loading")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--cache_shard_size", type=int, default=2048)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--skin_classes", type=int, default=2)
    parser.add_argument("--skin_stage_split_dir", type=str, default="None")
    parser.add_argument("--skin_cache_dir", type=str, default="None")
    parser.add_argument("--skin_data_root", type=str, default="None")

    parser.add_argument("--train_csv_file", type=str, default="None")
    parser.add_argument("--val_csv_file", type=str, default="None")
    parser.add_argument("--test_csv_file", type=str, default="None")
    parser.add_argument("--image_folder", type=str, default="None")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--aptos_train_image_folder", type=str, default="None")
    parser.add_argument("--aptos_val_image_folder", type=str, default="None")
    parser.add_argument("--aptos_test_image_folder", type=str, default="None")
    parser.add_argument("--ddr_image_folder", type=str, default="None")

    parser.add_argument("--diabetic_train_csv_file", type=str, default="None")
    parser.add_argument("--diabetic_val_csv_file", type=str, default="None")
    parser.add_argument("--diabetic_test_csv_file", type=str, default="None")
    parser.add_argument("--diabetic_image_folder", type=str, default="None")
    parser.add_argument("--diabetic_classes", type=int, default=3)

    

    

    return parser.parse_args()


def main():
    # Argument Parsing
    args = parse_args()

    set_seed(1004)

    gpu_ids = args.gpu_ids.split(",")

    args.save_path = create_save_directory()

    if args.train_mode == 0:

        ddp_train_step1(args)

        ddp_train_step2(args)

        ddp_train_step3(args)




if __name__ == "__main__":
    main()