import os
import pandas as pd
from datetime import datetime
import json
from typing import Tuple
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.elunet.elunet import ELUnet

# Global variables
eaii_datset_path = (
    "/home/rdadmin/Documents/Datasets/Coffee-Datasets/eaii_coffee_arabica/segmentation"
)

# ++++++++++++++++++++++++++++++++++++++++
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 100
# ++++++++++++++++++++++++++++++++++++++++

CURRENT_VAL_LOSS = float("inf")
CURRENT_EPOCH = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DF = pd.read_csv(os.path.join(eaii_datset_path, "dataset.csv"))
TRAIN_DF, TEST_DF = train_test_split(DATASET_DF, train_size=0.8, random_state=42)


class CoffeeArabicaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_path: str,
        target_size: Tuple = (256, 256),
        train: bool = True,
    ) -> None:
        super().__init__()
        self.df = df
        self.dataset_path = dataset_path
        self.train = train
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> torch.Tensor:
        # get image and mask file paths
        image_filename = self.df.iloc[index, 0]
        mask_filename = self.df.iloc[index, 1]
        # read image and masks
        image = cv2.imread(os.path.join(self.dataset_path, image_filename))
        mask = cv2.imread(os.path.join(self.dataset_path, mask_filename))
        # Convert image to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert mask to gray and threshold
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        # Convert image and mask to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # resize image and mask to the target size
        image = TF.resize(image, self.target_size)
        mask = TF.resize(mask, self.target_size)
        # add the same random rotation to the image and mask
        if self.train:
            # random rotate
            if random.random() < 0.5:
                angle = random.randint(-20, 20)

                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        return image, mask


def get_model():
    elunet_model = ELUnet(3, 1, 16)
    model = elunet_model.to(DEVICE)
    model.load_state_dict(
        torch.load("./elunet_symptom_lara_leaf_2023-03-24.pth", map_location=DEVICE)
    )
    count = 0
    for child in model.children():
        if count > 15:
            for param in child.parameters():
                param.requires_grad = False
        count += 1
    return model


# Define the dice loss function
# define dice coefficient and dice loss function
def calc_dice_loss(preds: torch.Tensor, targets: torch.Tensor):
    def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor):
        smooth = 1.0
        assert preds.size() == targets.size()

        iflat = preds.contiguous().view(-1)
        tflat = targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    return 1.0 - dice_coefficient(preds, targets)


def early_stopping(val_loss, epoch, patience=10, min_delta=0.01):
    """Helper method for model training early stopping"""

    global CURRENT_VAL_LOSS
    global CURRENT_EPOCH

    if val_loss < CURRENT_VAL_LOSS and CURRENT_VAL_LOSS - val_loss >= min_delta:
        # Val loss improved -> save model
        print(f"[Info] Val loss improved from {CURRENT_VAL_LOSS} to {val_loss}")
        # save_model(model)
        CURRENT_VAL_LOSS = val_loss
        CURRENT_EPOCH = epoch
        return False
    if val_loss >= CURRENT_VAL_LOSS and epoch - CURRENT_EPOCH >= patience:
        ## Stop training
        return True
    return False


@torch.no_grad()
def evaluate(model, val_loader, desc="Evaluate-val"):
    progress_bar = tqdm(val_loader, total=len(val_loader))

    val_loss = []
    for i, data in enumerate(progress_bar):
        image, mask = data
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        # get predicition
        mask_pred = model(image)

        loss = calc_dice_loss(mask_pred, mask)

        val_loss.append(loss.item())
        progress_bar.set_description(f"{desc}")
        progress_bar.set_postfix(
            eval_loss=np.mean(val_loss), eval_dice=1 - np.mean(val_loss)
        )
        progress_bar.update()

    return np.mean(val_loss)


def train(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = []
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for i, data in enumerate(progress_bar):
            image, mask = data
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            # get predicition
            mask_pred = model(image)

            loss = calc_dice_loss(mask_pred, mask)
            # empty gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            progress_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            progress_bar.set_postfix(
                loss=np.mean(epoch_loss), dice=1 - np.mean(epoch_loss)
            )
            progress_bar.update()
        # Evaluate
        eval_loss = evaluate(model, val_loader)
        scheduler.step(torch.mean(torch.tensor(eval_loss), dtype=torch.float))

        if early_stopping(eval_loss, epoch):
            print(f"[INFO] Early Stopping")
            print(f"Stopped!!")
            break


def cross_validate(k=5):
    # Define the cross-validation splitter
    kf = KFold(n_splits=k, shuffle=True)

    cross_val_scores = {"Loss": []}
    fold_scores = []
    best_model = None

    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(TRAIN_DF)):
        print(f"Fold {fold + 1}")
        model = get_model()

        # Define your training and validation sets for this fold
        train_df = TRAIN_DF.iloc[train_idx]
        val_df = TRAIN_DF.iloc[val_idx]
        # define the dataset
        train_dataset = CoffeeArabicaDataset(train_df, eaii_datset_path)
        val_dataset = CoffeeArabicaDataset(val_df, eaii_datset_path, train=False)
        test_dataset = CoffeeArabicaDataset(TEST_DF, eaii_datset_path, train=False)

        # Define the data loaders for this fold
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # Train and evaluate model
        train(model, train_loader, val_loader)

        loss = evaluate(model, test_loader, desc="Evaluate-test")
        fold_scores.append(loss)

        if loss <= np.min(fold_scores):
            best_model = model
            print(f"Current best fold -> {fold}")

        cross_val_scores["Loss"].append(
            {f"fold_{fold}": {"loss": loss, "mDice": 1 - loss}}
        )

    return best_model, cross_val_scores


def save_model(model):
    date_postfix = datetime.now().strftime("%Y-%m-%d")
    model_name = f"arabica_symptom_{date_postfix}.pth"
    save_path = "../coffee_arabica_weights"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        print(f"[INFO] Saving model to {os.path.join(save_path,model_name)}")
        torch.save(model.state_dict(), os.path.join(save_path, model_name))


if __name__ == "__main__":
    model, crossval_scores = cross_validate(5)
    # save model
    save_model(model)
    del model
    torch.cuda.empty_cache()
    # write dict to json

    with open("crossval_result.json", "w") as outfile:
        json.dump(crossval_scores, outfile)
