import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import zarr
import cv2
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    base_path = "D:/IVP _ project/data/SSL4EO-S12-v1.1/train/S1GRD"

    print("🚀 Training on 100 folders")
    train_dataset = SimCLRSARDataset(base_path, folder_limit=100)
    model = CNNTransformerEncoder()
    train_simclr(model, train_dataset)

    print("📥 Extracting embeddings from 398 folders")
    full_dataset = SimCLRSARDataset(base_path, folder_limit=398)
    model.load_state_dict(torch.load("simclr_encoder.pth"))
    extract_embeddings(model, full_dataset)
