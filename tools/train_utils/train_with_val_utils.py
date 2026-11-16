import json

import numpy as np
import os

import torch
import torch.nn as nn
import tqdm
import glob

from eval_utils import eval_utils
from pcdet.utils import commu_utils
from train_utils.train_utils import (
    train_one_epoch, checkpoint_state, save_checkpoint, disable_augmentation_hook
)

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _freeze_bn(module):
    if isinstance(module, _BN_TYPES):
        module.eval()


def compute_val_loss(model, model_func, dataloader, dist_test=False):
    """
    Computes the average training-style loss on a validation dataloader.
    The loss branch of OpenPCDet detectors only runs when `model.training` is True,
    so the model is temporarily switched to train() (with BatchNorm layers frozen to
    eval() to avoid corrupting running stats with validation-batch statistics) and
    restored to its original mode afterward.
    """
    was_training = model.training
    model.train()
    model.apply(_freeze_bn)

    loss_sum = 0.0
    tb_sum = {}
    num_batches = 0
    with torch.no_grad():
        for batch_dict in dataloader:
            loss, tb_dict, _ = model_func(model, batch_dict)
            loss_sum += loss.item()
            for key, val in tb_dict.items():
                tb_sum[key] = tb_sum.get(key, 0.0) + val
            num_batches += 1

    avg_loss = loss_sum / max(num_batches, 1)
    avg_tb_dict = {key: val / max(num_batches, 1) for key, val in tb_sum.items()}

    if dist_test:
        avg_loss = commu_utils.average_reduce_value(avg_loss)
        avg_tb_dict = {key: commu_utils.average_reduce_value(val) for key, val in avg_tb_dict.items()}

    model.train(was_training)
    return avg_loss, avg_tb_dict


def train_model_with_val(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False, cfg=None,
                test_loader=None, eval_interval=0, eval_output_dir=None, dist_test=False, args=None,
                save_best_last=False, best_metric='AP_3D_mean/overall', train_eval_loader=None):
    """save_best_last: instead of checkpoint_epoch_N every ckpt_save_interval epochs keep only
    checkpoint_best.pth (best val `best_metric` of the periodic evaluations, details in best.json) and
    checkpoint_last.pth (the last epoch, which is always evaluated)"""
    accumulated_iter = start_iter
    best_file = ckpt_save_dir / 'best.json'
    best = json.load(open(best_file)) if (save_best_last and best_file.exists()) else None  # survives a resume

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None)
    augment_disable_flag = False

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            augment_disable_flag = disable_augmentation_hook(hook_config, dataloader_iter, total_epochs, cur_epoch, cfg, augment_disable_flag, logger)
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,

                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval,
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if save_best_last:
                if trained_epoch == total_epochs and rank == 0:
                    save_checkpoint(checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                                    filename=ckpt_save_dir / 'checkpoint_last')
            elif (trained_epoch % ckpt_save_interval == 0 or trained_epoch == total_epochs) and rank == 0:  # always keep the last epoch

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # periodic in-training validation
            if test_loader is not None and eval_interval > 0 and (trained_epoch % eval_interval == 0 or trained_epoch == total_epochs):
                eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                cur_result_dir = eval_output_dir / ('epoch_%d' % trained_epoch) / cfg.DATA_CONFIG.DATA_SPLIT['test']
                val_ret_dict = eval_utils.eval_one_epoch(
                    cfg, args, eval_model, test_loader, trained_epoch, logger,
                    dist_test=dist_test, result_dir=cur_result_dir,
                    eval_tag='val: %s' % cfg.DATA_CONFIG.DATA_SPLIT['test']
                )
                val_loss, val_tb_dict = compute_val_loss(
                    eval_model, model_func, test_loader, dist_test=dist_test
                )
                if rank == 0:
                    logger.info('*************** EPOCH %d VAL LOSS: %.4f *****************' % (trained_epoch, val_loss))
                    if tb_log is not None:
                        for key, val in val_ret_dict.items():
                            if np.isscalar(val) and np.isnan(val):
                                continue  # undefined (no gt in that bin)
                            tb_log.add_scalar('val/' + key, val, trained_epoch)
                        tb_log.add_scalar('val/loss', val_loss, trained_epoch)
                        for key, val in val_tb_dict.items():
                            tb_log.add_scalar('val_loss/' + key, val, trained_epoch)
                if train_eval_loader is not None:  # same evaluation on training frames (no augmentation)
                    te_ret_dict = eval_utils.eval_one_epoch(
                        cfg, args, eval_model, train_eval_loader, trained_epoch, logger, dist_test=dist_test,
                        result_dir=eval_output_dir / ('epoch_%d' % trained_epoch) / train_eval_loader.dataset.split,
                        eval_tag='train_eval: %s' % train_eval_loader.dataset.split
                    )
                    te_loss, te_tb_dict = compute_val_loss(eval_model, model_func, train_eval_loader, dist_test=dist_test)
                    if rank == 0:
                        logger.info('*************** EPOCH %d TRAIN_EVAL LOSS: %.4f, %s: train_eval %.2f / val %.2f *****************'
                                    % (trained_epoch, te_loss, best_metric, te_ret_dict.get(best_metric, float('nan')),
                                       val_ret_dict.get(best_metric, float('nan'))))
                        if tb_log is not None:
                            for key, val in te_ret_dict.items():
                                if np.isscalar(val) and np.isnan(val):
                                    continue  # undefined (no gt in that bin)
                                tb_log.add_scalar('train_eval/' + key, val, trained_epoch)
                            tb_log.add_scalar('train_eval/loss', te_loss, trained_epoch)
                            for key, val in te_tb_dict.items():
                                tb_log.add_scalar('train_eval_loss/' + key, val, trained_epoch)
                if rank == 0:
                    if save_best_last:
                        assert best_metric in val_ret_dict, 'best_metric %s not in %s' % (best_metric, sorted(val_ret_dict))
                        score = float(val_ret_dict[best_metric])
                        if best is None or score > best['score']:
                            best = {'metric': best_metric, 'score': score, 'epoch': trained_epoch,
                                    'val_loss': float(val_loss), 'iteration': int(accumulated_iter)}
                            save_checkpoint(checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                                            filename=ckpt_save_dir / 'checkpoint_best')
                            json.dump(best, open(best_file, 'w'), indent=2)
                        logger.info('best %s so far: %.2f at epoch %d' % (best_metric, best['score'], best['epoch']))
