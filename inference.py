from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchsummary import summary

from data_loader.DataLoader import DIV2K, GaoFen2, Sev2Mod, WV3, GaoFen2panformer
from MSDCNN import PNNmodel
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Prepare device
    # TODO add more code for server
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize DataLoader
    train_dataset = GaoFen2(
        Path("F:/Data/GaoFen-2/train/train_gf2-001.h5"), transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])  # /home/ubuntu/project
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, drop_last=True)

    validation_dataset = GaoFen2(
        Path("F:/Data/GaoFen-2/val/valid_gf2.h5"))
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=64, shuffle=True)

    test_dataset = GaoFen2(
        Path("F:/Data/GaoFen-2/drive-download-20230623T170619Z-001/test_gf2_multiExm1.h5"))
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=64, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    # TODO is imge_size necesasary?
    model = PNNmodel(scale=4, mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                     pan_std=train_dataset.pan_std.to(device)).to(device)

    my_list = ['conv_3.weight', 'conv_3.bias']
    params = list(
        filter(lambda kv: kv[0] in my_list, model.parameters()))
    base_params = list(
        filter(lambda kv: kv[0] not in my_list, model.parameters()))

    optimizer = SGD([
        {'params': params},
        {'params': base_params, 'lr': 1e-9}
    ], lr=1e-8, momentum=0.9)

    criterion = MSELoss().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    tr_report_loss = 0
    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []
    best_eval_psnr = 0
    best_test_psnr = 0
    current_daytime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    steps = 200000
    save_interval = 1000
    report_interval = 50
    test_intervals = [100000, 200000, 300000, 400000,
                      500000, 600000, 700000, 800000, 900000, 1000000]
    evaluation_interval = [100000, 200000, 300000, 400000,
                           500000, 600000, 700000, 800000, 900000, 1000000]
    val_steps = 50
    continue_from_checkpoint = True

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics, test_metrics = load_checkpoint(torch.load(
            'checkpoints/pnn_model/pnn_model_2023_07_17-11_30_23_best_eval.pth.tar'), model, optimizer, tr_metrics, val_metrics, test_metrics)
        print('Model Loaded ...')

    def scaleMinMax(x):
        return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))

    idx = 0
    # evaluation mode
    model.eval()
    with torch.no_grad():
        test_iterator = iter(test_loader)
        for i, (pan, mslr, mshr) in enumerate(test_iterator):
            if idx == i:
                # forward
                pan, mslr, mshr = pan.to(device), mslr.to(
                    device), mshr.to(device)
                mssr = model(pan, mslr)
                test_loss = criterion(mssr, mshr)
                test_metric = test_metric_collection.forward(mssr, mshr)
                test_report_loss += test_loss

                # compute metrics
                test_metric = test_metric_collection.compute()

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
                    f'(c) PNN {test_metric["psnr"]:.2f}dB/{test_metric["ssim"]:.4f}')
                axis[2].axis("off")

                axis[3].imshow((scaleMinMax(mshr.permute(0, 3, 2, 1).detach().cpu()[
                               0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
                axis[3].set_title('(d) GT')
                axis[3].axis("off")

                plt.savefig('results/Images.png')


if __name__ == '__main__':
    main()
