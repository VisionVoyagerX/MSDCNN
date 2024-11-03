from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis, RelativeAverageSpectralError, SpectralDistortionIndex
from torchmetrics.regression import MeanSquaredError
from torchinfo import summary

from data_loader.DataLoader import DIV2K, GaoFen2, Sev2Mod, WV3, GaoFen2panformer
from MSDCNN import MSDCNN_model
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def main():
    choose_dataset = 'WV3' #or 'WV3'

    if choose_dataset == 'GaoFen2':
        dataset = eval('GaoFen2')
        tr_dir = 'data/pansharpenning_dataset/GF2/train/train_gf2.h5'
        eval_dir = 'data/pansharpenning_dataset/GF2/val/valid_gf2.h5'
        test_dir =  'data/pansharpenning_dataset/GF2/test/test_gf2_multiExm1.h5'
        checkpoint_dir = 'checkpoints/MSDCNN_model_GF2/MSDCNN_model_GF2_2023_07_26-08_30_23.pth.tar'
        ms_channel = 4
    elif choose_dataset == 'WV3':
        dataset = eval('WV3')
        tr_dir = 'data/pansharpenning_dataset/WV3/train/train_wv3.h5'
        eval_dir = 'data/pansharpenning_dataset/WV3/val/valid_wv3.h5'
        test_dir =  'data/pansharpenning_dataset/WV3/test/test_wv3_multiExm1.h5'
        checkpoint_dir = 'checkpoints/MSDCNN_model_WV3/MSDCNN_model_WV3_2023_07_26-13_25_41.pth.tar'
        ms_channel = 8
    else:
        print(choose_dataset, ' does not exist')

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize DataLoader
    # Initialize DataLoader
    train_dataset = dataset(
        Path(tr_dir), transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])  # /home/ubuntu/project
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, drop_last=True)

    validation_dataset = dataset(
        Path(eval_dir))
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=64, shuffle=True)

    test_dataset = dataset(
        Path(test_dir))
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    model = MSDCNN_model(scale=4,  ms_channels=ms_channel, mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                         pan_std=train_dataset.pan_std.to(device)).to(device)

    optimizer = SGD(model.parameters(), lr=0.000001, momentum=0.9)

    criterion = MSELoss().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
        'rase' : RelativeAverageSpectralError().to(device),
        'mse' : MeanSquaredError().to(device),
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
        'rase' : RelativeAverageSpectralError().to(device),
        'mse' : MeanSquaredError().to(device),
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
        'rase' : RelativeAverageSpectralError().to(device),
        'mse' : MeanSquaredError().to(device),
    })

    sdi_metric = SpectralDistortionIndex().to(device)
    sdi_results = []


    tr_report_loss = 0
    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []
    best_eval_psnr = 0
    best_test_psnr = 0
    current_daytime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    steps = 250000
    save_interval = 1000
    report_interval = 50
    test_intervals = [20000, 40000, 60000, 80000, 100000, 120000,
                      140000, 160000, 180000, 200000, 220000, 240000, 250000]
    evaluation_interval = [20000, 40000, 60000, 80000, 100000,
                           120000, 140000, 160000, 180000, 200000, 220000, 240000, 250000]
    val_steps = 50
    continue_from_checkpoint = True

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics, test_metrics = load_checkpoint(torch.load(
            checkpoint_dir), model, optimizer, tr_metrics, val_metrics, test_metrics)
        print('Model Loaded ...')

    def scaleMinMax(x):
        return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))

    # evaluation mode
    model.eval()
    with torch.no_grad():
        test_iterator = iter(test_loader)
        for i, (pan, mslr, mshr) in enumerate(test_iterator):
            # forward
            pan, mslr, mshr = pan.to(device), mslr.to(
                device), mshr.to(device)
            mssr = model(pan, mslr)
            test_loss = criterion(mssr, mshr)
            test_metric = test_metric_collection.forward(mssr, mshr)
            test_report_loss += test_loss

            # Normalize preds and target for SDI
            # print(mssr.max())
            preds_normalized = mssr / mssr.max()
            target_normalized = mshr / mshr.max()

            # Calculate SDI on normalized predictions and targets
            sdi_value = sdi_metric(preds_normalized, target_normalized)
            # print(sdi_value)
            sdi_results.append(sdi_value.item())

            figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
            axis[0].imshow((scaleMinMax(mslr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[0].set_title('(a) LR')
            axis[0].axis("off")

            axis[1].imshow(pan.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...], cmap='gray')
            axis[1].set_title('(b) PAN')
            axis[1].axis("off")

            axis[2].imshow((scaleMinMax(mssr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[2].set_title(
                f'(c) MSDCNN {test_metric["psnr"]:.2f}dB/{test_metric["ssim"]:.4f}')
            axis[2].axis("off")

            axis[3].imshow((scaleMinMax(mshr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[3].set_title('(d) GT')
            axis[3].axis("off")

            plt.savefig(f'results/Images_{choose_dataset}_{i}.png')

            mslr = mslr.permute(0, 3, 2, 1).detach().cpu().numpy()
            pan = pan.permute(0, 3, 2, 1).detach().cpu().numpy()
            mssr = mssr.permute(0, 3, 2, 1).detach().cpu().numpy()
            gt = mshr.permute(0, 3, 2, 1).detach().cpu().numpy()

            np.savez(f'results/img_array_{choose_dataset}_{i}.npz', mslr=mslr,
                        pan=pan, mssr=mssr, gt=gt)
            
        # compute metrics
        test_metric = test_metric_collection.compute()
        test_metric_collection.reset()

        # Compute the average SDI
        average_sdi = sum(sdi_results) / len(sdi_results)

        # Print final scores
        print(f"Final scores:\n"
              f"ERGAS: {test_metric['ergas'].item()}\n"
              f"SAM: {test_metric['sam'].item()}\n"
              f"PSNR: {test_metric['psnr'].item()}\n"
              f"SSIM: {test_metric['ssim'].item()}\n"
              f"RASE: {test_metric['rase'].item()}\n"
              f"MSE: {test_metric['mse'].item()}\n"
              f"D_lambda: {average_sdi:.4f}")


if __name__ == '__main__':
    main()
