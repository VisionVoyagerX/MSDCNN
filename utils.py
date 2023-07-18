from pathlib import Path
import torch
import datetime


def get_checkpoint_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    checkpoints_path = current_dir / "checkpoints"
    return checkpoints_path


def get_config_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir / "configs"
    return config_path


def get_data_loader_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    data_loader_path = current_dir / "data_loader"
    return data_loader_path


def get_model_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "model"
    return model_path


def save_checkpoint(state, filename: str, current_daytime: str):
    # print("=> Saving checkpoint")
    checkpoint_path = get_checkpoint_path()

    (checkpoint_path/filename).mkdir(parents=True, exist_ok=True)
    torch.save(state,  checkpoint_path / filename /
               f'{filename}_{current_daytime}.pth.tar')


def load_checkpoint(checkpoint, model, optimizer, tr_metrics, val_metrics, test_metrics):
    # print("=> Loading checkpoint")
    try:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        tr_metrics = checkpoint['tr_metrics']
        val_metrics = checkpoint['val_metrics']
        test_metrics = checkpoint['test_metrics']

        return (tr_metrics, val_metrics, test_metrics)  # val_metrics
    except:
        print('Could not load model')
        return Exception
