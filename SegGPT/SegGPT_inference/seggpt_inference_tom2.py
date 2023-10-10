import os
import argparse

import torch
import numpy as np
import os
import sys
import glob

from seggpt_engine_tom import inference_image_tom, inference_video_tom
import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--in_dir', type=str, help='path to inputs',
                        default=None, required=True)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    in_dir = args.in_dir

    for file in os.listdir(in_dir):
        if (file != file.lower()):
            old_name = os.path.join(in_dir, file)
            new_name = os.path.join(in_dir, file.lower())
            os.rename(old_name, new_name)

    prompt_images = []
    prompt_targets = []
    for name in sorted(glob.glob(f"{in_dir}/input_?.jpg")):
        letter = name[-5]
        mask_file = f"{in_dir}/mask_{letter}.png"
        print(f"Processing input files: {os.path.basename(name)} + {os.path.basename(mask_file)}")
        if not os.path.isfile(mask_file):
            print(f"ERROR: file missing {os.path.basename(mask_file)}")
            sys.exit(1)
        prompt_images.append(name)
        prompt_targets.append(mask_file)

    input_image = os.path.join(in_dir, "input_new.jpg")
    out_path = os.path.join(in_dir, "mask_new.png")

    if os.path.isfile(input_image):
        print(f"Processing mask from input: {os.path.basename(input_image)} -> {os.path.basename(out_path)}")
        inference_image_tom(model, device, input_image, prompt_images, prompt_targets, out_path)
    else:
        filelist = glob.glob(f"{in_dir}/input_new?.jpg")
        if len(filelist) == 0:
                print(f"ERROR: could not find any input_new files")
                sys.exit(1)
        for input_image in sorted(filelist):
            letter = input_image[-5]
            out_path = f"{in_dir}/mask_new{letter}.png"
            print(f"Processing mask from input: {os.path.basename(input_image)} -> {os.path.basename(out_path)}")
            inference_image_tom(model, device, input_image, prompt_images, prompt_targets, out_path)
    
    print('Finished.')
