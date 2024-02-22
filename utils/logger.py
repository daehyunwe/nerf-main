from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def save_loss(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
):
    """
    Append loss to the loss text file.
    """
    log_path.mkdir(exist_ok=True)

    loss_txt_path = log_path / "loss.txt"

    if val_loss is None:
        with open(loss_txt_path.as_posix(), "a", encoding="UTF-8") as f:
            print(f"{epoch}, {train_loss}", file=f)
    else:
        with open(loss_txt_path.as_posix(), "a", encoding="UTF-8") as f:
            print(f"{epoch}, {train_loss}, {val_loss}", file=f)


def save_checkpoint(
    log_path: Path,
    epoch: int,
    model: dict,
    optimizer: optim.Optimizer,
    scheduler,
):
    """
    Save current epoch, model, optimizer, and scheduler state into log_path.
    A model is a dictionary whose keys are the name of each network
    and values are either nn.Module or nn.DataParallel object of that network.
    """
    log_path.mkdir(exist_ok=True)
    ckpt_path = log_path / f"ckpt_{str(epoch).zfill(4)}.pth"

    ckpt = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    for key in model:
        if isinstance(model[key], nn.DataParallel):
            ckpt[key] = model[key].module.state_dict()
        else:
            ckpt[key] = model[key].state_dict()

    ckpt_path.unlink(missing_ok=True)
    torch.save(ckpt, ckpt_path)


def load_checkpoint(
    log_path: Path,
    model: dict,
    optimizer: optim.Optimizer | None = None,
    scheduler=None,
) -> int:
    """
    Loads model, optimizer, and scheduler states.
    Returns the last training epoch number.
    """
    if not log_path.exists():
        print("No checkpoint found. Training from scratch.")
        return 0

    # find the newest ckpt file number
    newest_ckpt_num = 0
    for child in log_path.iterdir():
        if child.name.startswith("ckpt_") and int(child.name[5:9]) > newest_ckpt_num:
            newest_ckpt_num = int(child.name[5:9])

    ckpt_path = log_path / f"ckpt_{str(newest_ckpt_num).zfill(4)}.pth"

    if not ckpt_path.exists():
        print("No checkpoint found. Training from scratch.")
        return 0

    ckpt = torch.load(ckpt_path, map_location="cpu")

    start_epoch = ckpt["epoch"] + 1
    if not optimizer is None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if not scheduler is None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    for key in model:
        model[key].load_state_dict(ckpt[key])
        model[key].to(torch.cuda.current_device())

    return start_epoch
