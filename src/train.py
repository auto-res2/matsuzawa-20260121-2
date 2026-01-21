import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig

import optuna
import wandb

# Local absolute imports (src is a package)
from src.preprocess import build_loaders
from src.model import get_model, AMSRPlusLoss, BaselineLoss

################################################################################
# --------------------------  Utility helpers  --------------------------------#
################################################################################

def set_seed(seed: int) -> None:
    """Ensure full reproducibility as far as possible."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum()
    return correct.float() / targets.numel()


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Vectorised Expected Calibration Error (ECE)."""
    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    accuracy = preds.eq(labels)

    bin_bounds = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        in_bin = (conf > bin_bounds[i]) & (conf <= bin_bounds[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracy[in_bin].float().mean()
            conf_in_bin = conf[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece.item()

################################################################################
# -----------------------  Core training loops  ------------------------------ #
################################################################################

def _train_one_epoch(model: nn.Module,
                     criterion: nn.Module,
                     loader: torch.utils.data.DataLoader,
                     optimizer: torch.optim.Optimizer,
                     device: torch.device,
                     epoch: int,
                     cfg: DictConfig) -> Dict[str, float]:
    model.train()
    loss_running, correct_running, total_samples = 0.0, 0.0, 0

    for step, (imgs, aug_params, targets) in enumerate(loader):
        # Defensive checks on first batch
        if step == 0:
            assert imgs.ndim == 4, f"Expect NCHW tensor; got {imgs.shape}"
            assert targets.ndim == 1, f"Targets must be 1-D; got {targets.shape}"
            assert aug_params.shape == (imgs.size(0), 2), "Aug-param dimension mismatch"

        imgs, targets, aug_params = imgs.to(device, non_blocking=True), \
                                    targets.to(device, non_blocking=True), \
                                    aug_params.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, targets, aug_params) if isinstance(criterion, AMSRPlusLoss) else criterion(logits, targets)
        loss.backward()

        # CRITICAL: verify gradients exist & non-zero
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()), \
            "Gradients vanished – investigate loss graph"

        if cfg.training.gradient_clip and cfg.training.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        optimizer.step()

        acc = compute_accuracy(logits, targets)
        batch_size = imgs.size(0)
        loss_running += loss.item() * batch_size
        correct_running += acc.item() * batch_size
        total_samples += batch_size

        # Log every 50 steps or on last step
        if wandb.run is not None and (step % 50 == 0 or step == len(loader) - 1):
            wandb.log({
                "train_loss": loss.item(),
                "train_acc": acc.item(),
                "epoch": epoch,
                "global_step": epoch * len(loader) + step,
            })

        # Trial-mode short-circuit
        if cfg.mode == "trial" and step >= 1:
            break

    return {
        "loss": loss_running / total_samples,
        "acc": correct_running / total_samples,
    }


def _evaluate(model: nn.Module,
              criterion: nn.Module,
              loader: torch.utils.data.DataLoader,
              device: torch.device,
              split: str = "val",
              collect_preds: bool = False) -> Tuple[Dict[str, float], Dict[str, List]]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    logits_all, labels_all = [], []
    preds_all, conf_all = [], []

    with torch.no_grad():
        for imgs, aug_params, targets in loader:
            imgs, targets, aug_params = imgs.to(device, non_blocking=True), \
                                        targets.to(device, non_blocking=True), \
                                        aug_params.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, targets, aug_params) if isinstance(criterion, AMSRPlusLoss) else criterion(logits, targets)

            acc = compute_accuracy(logits, targets)
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_correct += acc.item() * bs
            total_samples += bs

            if collect_preds:
                probs = F.softmax(logits, dim=1)
                conf, preds = probs.max(dim=1)
                preds_all.extend(preds.cpu().tolist())
                conf_all.extend(conf.cpu().tolist())
                logits_all.append(logits.cpu())
                labels_all.append(targets.cpu())

            # Trial-mode speed-up
            if loader.dataset.__dict__.get("_trial", False):
                break

    metrics = {
        f"{split}_loss": total_loss / total_samples,
        f"{split}_acc": total_correct / total_samples,
    }

    if collect_preds:
        logits_cat = torch.cat(logits_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)
        metrics[f"{split}_ece"] = compute_ece(logits_cat, labels_cat)
        extra = {
            "y_true": labels_cat.tolist(),
            "y_pred": preds_all,
            "y_conf": conf_all,
        }
    else:
        metrics[f"{split}_ece"] = None
        extra = {}
    return metrics, extra

################################################################################
# ------------------  Encapsulated training routine  ------------------------- #
################################################################################

def _run_training(cfg: DictConfig,
                  device: torch.device,
                  log_to_wandb: bool = True) -> Dict[str, Any]:
    """Train & validate once – used by both Optuna trials and final run."""
    train_loader, val_loader, _ = build_loaders(cfg)
    model = get_model(cfg).to(device)
    # Assertion immediately after model build
    assert model.fc.out_features == 10, "ResNet-18 final layer mismatch for CIFAR-10"

    criterion: nn.Module
    if cfg.method.lower().startswith("adaptive"):
        criterion = AMSRPlusLoss(kl_weight=cfg.training.kl_weight).to(device)
    else:
        criterion = BaselineLoss().to(device)

    if cfg.training.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer {cfg.training.optimizer}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs) \
        if cfg.training.scheduler == "cosine" else None

    best_val_acc = 0.0
    for epoch in range(cfg.training.epochs):
        train_metrics = _train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, cfg)
        val_metrics, _ = _evaluate(model, criterion, val_loader, device, split="val")
        val_acc = val_metrics["val_acc"]
        best_val_acc = max(best_val_acc, val_acc)

        if scheduler:
            scheduler.step()

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
                "epoch": epoch,
            })

        # Trial-mode break safety
        if cfg.mode == "trial":
            break

    return {"model": model, "best_val_acc": best_val_acc}

################################################################################
# ----------------------  Optuna optimisation  ------------------------------- #
################################################################################

def _optuna_objective(trial: optuna.Trial, base_cfg: DictConfig, device: torch.device) -> float:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))  # deep copy
    for space in base_cfg.optuna.search_spaces:
        name = space.param_name
        if space.distribution_type == "uniform":
            suggestion = trial.suggest_float(name, space.low, space.high)
        elif space.distribution_type == "loguniform":
            suggestion = trial.suggest_float(name, space.low, space.high, log=True)
        else:
            raise ValueError(f"Distribution {space.distribution_type} not supported")

        if name == "kl_weight":
            cfg.training.kl_weight = suggestion
        elif name == "learning_rate":
            cfg.training.learning_rate = suggestion
        else:  # generic nested update
            OmegaConf.update(cfg, name, suggestion, merge=True)

    outcome = _run_training(cfg, device, log_to_wandb=False)
    # minimise negative accuracy (i.e. maximise accuracy)
    return -outcome["best_val_acc"]

################################################################################
# ---------------------------  Hydra main  ------------------------------------#
################################################################################

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg_base: DictConfig):
    original_cwd = Path(hydra.utils.get_original_cwd())

    # --------------------------------------------------------------
    # Merge run-specific YAML
    # --------------------------------------------------------------
    run_yaml = original_cwd / f"config/runs/{cfg_base.run}.yaml"
    assert run_yaml.exists(), f"Run config missing: {run_yaml}"
    run_cfg = OmegaConf.load(run_yaml)
    # Disable struct mode to allow run_cfg to add new keys like run_id
    OmegaConf.set_struct(cfg_base, False)
    cfg: DictConfig = OmegaConf.merge(cfg_base, run_cfg)

    # --------------------------------------------------------------
    # Mode-specific overrides
    # --------------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.training.epochs = 1
        cfg.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # Seed & device
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------
    # Optuna – disabled WandB during search
    # --------------------------------------------------------------
    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0:
        os.environ["WANDB_MODE"] = "disabled"
        study_db = f"sqlite:///{cfg.results_dir}/optuna_{cfg.run}.db"
        study = optuna.create_study(direction="minimize", storage=study_db, load_if_exists=True)
        study.optimize(lambda t: _optuna_objective(t, cfg, device), n_trials=cfg.optuna.n_trials)
        best_params = study.best_trial.params
        print(f"Optuna best parameters: {best_params}")
        # Apply best params back to cfg
        for k, v in best_params.items():
            if k == "kl_weight":
                cfg.training.kl_weight = v
            elif k == "learning_rate":
                cfg.training.learning_rate = v
        # Re-enable WandB for final run
        if cfg.wandb.mode != "disabled":
            os.environ.pop("WANDB_MODE", None)

    # --------------------------------------------------------------
    # WandB initialisation AFTER hyper-parameter search
    # --------------------------------------------------------------
    if cfg.wandb.mode == "disabled":
        run = None
    else:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id if "run_id" in cfg else cfg.run,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB URL: {run.get_url()}")

    # --------------------------------------------------------------
    # Final training with (possibly optimised) hyper-parameters
    # --------------------------------------------------------------
    outcome = _run_training(cfg, device, log_to_wandb=run is not None)
    model = outcome["model"]
    best_val_acc = outcome["best_val_acc"]

    # --------------------------------------------------------------
    # Test evaluation – collect predictions for confusion matrix
    # --------------------------------------------------------------
    _, _, test_loader = build_loaders(cfg)
    if cfg.method.lower().startswith("adaptive"):
        criterion = AMSRPlusLoss(kl_weight=cfg.training.kl_weight).to(device)
    else:
        criterion = BaselineLoss().to(device)

    test_metrics, extra_preds = _evaluate(model, criterion, test_loader, device, split="test", collect_preds=True)

    # --------------------------------------------------------------
    # WandB summary & predictions
    # --------------------------------------------------------------
    if run is not None:
        run.summary.update({
            "best_val_accuracy": best_val_acc,
            **test_metrics,
            **extra_preds,  # y_true, y_pred, y_conf – used by evaluate.py
        })
        run.finish()

    # Friendly stdout for CI/GitHub-Actions
    summary_out = {
        "status": "completed",
        "best_val_accuracy": best_val_acc,
        **test_metrics,
    }
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
