import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    # Enable component logging (works whether model is Diffusion_TS or a wrapper with .diffusion)
    diff = getattr(model, 'diffusion', model)

    diff._train_components_dir = os.path.join(args.save_dir, "components_train")
    #num_feats = model_arg.input_shape[-1]
    diff._train_plot_dims = 2
    diff._train_log_every = 1000   # set to 1 for a quick sanity check

    diff._gen_components_dir = os.path.join(args.save_dir, "components_gen")
    diff._gen_plot_dims = 1
    diff._log_components_at_gen = True

    os.makedirs(diff._train_components_dir, exist_ok=True)
    os.makedirs(diff._gen_components_dir, exist_ok=True)
    print("[components] train ->", diff._train_components_dir)
    print("[components] gen   ->", diff._gen_components_dir)
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            # Map from [-1,1] back to [0,1]
            samples = unnormalize_to_zero_to_one(samples)
            # Then invert the MinMax scaler to ORIGINAL UNITS
            samples = dataset.scaler.inverse_transform(
                samples.reshape(-1, samples.shape[-1])
            ).reshape(samples.shape)

        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}_orig.npy'), samples)

        import shutil

        # where the dataset wrote the auxiliary npys
        samples_dir = os.path.join(args.save_dir, "samples")
        mask_src = os.path.join(samples_dir, f"{dataset.name}_masking_{dataset.window}.npy")
        gt_src   = os.path.join(samples_dir, f"{dataset.name}_ground_truth_{dataset.window}_test.npy")

        # your desired filenames at the run root
        val_masks_out         = os.path.join(args.save_dir, "val_masks.npy")
        val_groundtruth_out   = os.path.join(args.save_dir, "val_groundtruth.npy")
        val_reconstruction_out= os.path.join(args.save_dir, "val_reconstruction.npy")

        # save reconstructions with your preferred name
        np.save(val_reconstruction_out, samples)

        # copy (or re-save) masks and ground-truth to your preferred names
        if os.path.isfile(mask_src):
            shutil.copyfile(mask_src, val_masks_out)
        else:
            # if you prefer re-saving rather than copying:
            masks = np.load(mask_src)  # will raise if not found; remove this else-block if you only want copy
            np.save(val_masks_out, masks)

        if os.path.isfile(gt_src):
            shutil.copyfile(gt_src, val_groundtruth_out)
        else:
            gts = np.load(gt_src)
            np.save(val_groundtruth_out, gts)

        print("[saved]", val_reconstruction_out)
        print("[saved]", val_masks_out)
        print("[saved]", val_groundtruth_out)

    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
