import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils import data as data_modul
import torch
import pytorch_lightning as pl
import pytorch_lightning

import MIPMLP


def load_data_2d_train_test(path_of_2D_matrix, tag_train, tag_test, params):
    def load_by_tag(_tag):
        X, y = [], []
        for index, tag in zip(_tag.index, list(_tag.values.flatten())):
            try:
                try:
                    _otu = np.load(f"../{path_of_2D_matrix}/{index}.npy", allow_pickle=True)
                except FileNotFoundError:
                    _otu = np.load(f"{path_of_2D_matrix}/{index}.npy", allow_pickle=True)

                X.append(_otu)
                y.append(tag)

            except KeyError:
                pass
        X = np.array(X)
        params["input_dim"] = (X.shape[1], X.shape[2])
        y = np.array(y)

        return X, y

    X_train, y_train = load_by_tag(tag_train)
    X_test, y_test = load_by_tag(tag_test)

    return X_train, X_test, y_train, y_test


def run_iMic(train_dataset, test_dataset,
             model: pl.LightningModule, parms: dict = None, mode=None, task="class", weighted=False):
    num_workers = 0
    # load data according to batches:
    trainloader = data_modul.DataLoader(train_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
    testloader = data_modul.DataLoader(test_dataset, batch_size=parms["batch_size"], num_workers=num_workers)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        tt = pl.Trainer(precision=32, max_epochs=100, gpus=1, logger=None,
                        )
    else:
        tt = pl.Trainer(precision=32, max_epochs=100, enable_checkpointing=True,
                        logger=None
                        )

    model = model(parms, task=task, mode=mode, weighted=weighted)
    tt.fit(model, trainloader)
    pred_train = model.predict(trainloader)
    pred_test = model.predict(testloader)
    return pred_train, pred_test


def apply_iMic(tag, folder, test_size=0.2, params={
    "l1_loss": 0.1,
    "weight_decay": 0.01,
    "lr": 0.001,
    "batch_size": 128,
    "activation": "elu",
    "dropout": 0.1,
    "kernel_size_a": 4,
    "kernel_size_b": 4,
    "stride": 2,
    "padding": 3,
    "padding_2": 0,
    "kernel_size_a_2": 2,
    "kernel_size_b_2": 7,
    "stride_2": 3,
    "channels": 3,
    "channels_2": 14,
    "linear_dim_divider_1": 10,
    "linear_dim_divider_2": 6,
    "input_dim": (8, 235)
}):
    # Upload ordered otu data
    otu = pd.read_csv(f"{folder}/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv", index_col=0)

    # Train test split
    o_train, o_test, t_train, t_test = train_test_split(otu, tag, test_size=test_size)

    # Divide images to train and test accordingly
    X_train, X_test, y_train, y_test = load_data_2d_train_test(folder, t_train, t_test, params)

    # Make the ndarrays into datasets
    train_dataset = data_modul.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = data_modul.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    model = MIPMLP.CNN  # (params)
    pred_train, pred_test = run_iMic(train_dataset, test_dataset, model, params)

    return {"pred_train": pred_train,
            "pred_test": pred_test,
            "y_train": y_train,
            "y_test": y_test}


if __name__ == "__main__":
    tag = pd.read_csv("data/ibd_tag.csv", index_col=0)
    otu = pd.read_csv("data/ibd_for_process.csv")
    # MIPMLP.micro2matrix(otu, folder="data/2D_images")
    dct = apply_iMic(tag, folder="data/2D_images")
