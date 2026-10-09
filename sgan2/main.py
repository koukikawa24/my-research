import torch
import os
import datetime
from config import get_config
from data import make_dataset
from train_StFold2 import train_gan    #train2
import torch.nn as nn
import numpy as np

def main():
    config = get_config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dataset = make_dataset(config.labeleddata_npypath, config.label_csvpath, device)
    if config.enable_ssl:
        lldataset = make_dataset(config.labellessdata_npypath, None, device)
    else:
        lldataset = None

    adv_loss = nn.BCELoss().to(device)
    aux_loss = nn.CrossEntropyLoss().to(device)

    now = datetime.datetime.now()
    log_path = f"./log_8class/{now:%Y%m%d%H%M}_result.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w') as f:
        accuracies, f1_scores, qwk_scores, conf_matrix, best_state, best_fold = train_gan(
            config, dataset, lldataset, adv_loss, aux_loss
        )

        f.write(f"config: {config}\n")
        f.write(f"Mean Accuracy: {np.mean(accuracies)}\n")
        f.write(f"Mean F1 Score: {np.mean(f1_scores)}\n")
        f.write(f"Mean QWK Score: {np.mean(qwk_scores)}\n")
        f.write(f"Accuracies: {accuracies}\n")
        f.write(f"F1 Scores: {f1_scores}\n")
        f.write(f"QWK Scores: {qwk_scores}\n")
        f.write(f"Total Confusion Matrix:\n{conf_matrix}\n")

        save_path = f"./saved_models/{now:%Y-%m-%d_%H-%M}_best_discriminator_fold{best_fold}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        f.write(f"Saved best model to {save_path}\n")

if __name__ == "__main__":
    main()
