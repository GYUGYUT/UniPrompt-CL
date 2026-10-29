import torch

def get_scheduler(optimizer, scheduler_type, args):
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif scheduler_type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience, factor=args.gamma)