"""
Pair reward evaluator for validation.

Computes mean loss over val_loader using the model's own loss output
and records it for checkpoint selection.
"""

import torch
import wandb

import pointcept.utils.comm as comm
from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class PairRewardEvaluator(HookBase):
    def __init__(self, max_batches=None):
        """
        Args:
            max_batches: optional int to cap number of val batches per epoch.
        """
        self.max_batches = max_batches

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start PairReward Eval >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        loss_sum = 0.0
        count = 0
        with torch.no_grad():
            for i, input_dict in enumerate(self.trainer.val_loader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"].detach()
                loss_sum += loss.item()
                count += 1
                self.trainer.logger.info(
                    "Val: [{iter}/{max_iter}] Loss {loss:.4f}".format(
                        iter=i + 1,
                        max_iter=len(self.trainer.val_loader),
                        loss=loss.item(),
                    )
                )

        loss_avg = loss_sum / max(count, 1)
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            if self.trainer.cfg.enable_wandb and comm.is_main_process():
                wandb.log({"Epoch": current_epoch, "val/loss": loss_avg}, step=wandb.run.step)

        # higher metric is better; use negative loss to maximize
        self.trainer.comm_info["current_metric_value"] = -loss_avg
        self.trainer.comm_info["current_metric_name"] = "neg_val_loss"
        self.trainer.logger.info(
            f"Val result: loss {loss_avg:.4f} (metric {self.trainer.comm_info['current_metric_name']})"
        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End PairReward Eval <<<<<<<<<<<<<<<<<")

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("neg_val_loss", self.trainer.best_metric_value)
        )
