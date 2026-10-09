import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.autograd import Variable
from models import Generator, Discriminator, weights_init_normal
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import datetime
from utils import evaluate_predictions2, get_qwk
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_gan(config, dataset, lldataset, adversarial_loss, auxiliary_loss):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    set_seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)
    
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=0)
    total_conf_matrix = np.zeros((config.num_classes, config.num_classes), dtype=int)
    all_accuracies, all_f1_scores, all_qwk_scores, fold_results = [], [], [], []

    best_accuracy, best_state, best_fold = 0.0, None, 0

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        generator = Generator(config).to(device)
        discriminator = Discriminator(config).to(device)
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        train_dataset = Subset(dataset, train_idx)
        if config.enable_ssl:
            train_dataset = ConcatDataset([train_dataset, lldataset])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=len(test_idx), shuffle=False)
        dataloaders = {'train': train_loader, 'valid': test_loader}

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.b1, config.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.b1, config.b2))
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

        loss_history = {'train': {'g_loss': [], 'd_loss': []},
                        'valid': {'g_loss': [], 'd_loss': []}}

        for epoch in tqdm(range(config.n_epochs), desc=f"Fold {fold+1}"):
            for phase in ['train', 'valid']:
                generator.train() if phase == 'train' else generator.eval()
                discriminator.train() if phase == 'train' else discriminator.eval()

                total_g_loss, total_d_loss, num_batches = 0.0, 0.0, 0

                for imgs, labels in dataloaders[phase]:
                    imgs = imgs.view(-1, 1, config.width, config.height).to(device)
                    labels = labels.to(device)

                    current_batch_size = imgs.size(0)

                    valid = Variable(FloatTensor(current_batch_size, 1).fill_(config.label_real), requires_grad=False).to(device)
                    fake = Variable(FloatTensor(current_batch_size, 1).fill_(config.label_fake), requires_grad=False).to(device)
                    fake_aux_gt = Variable(LongTensor(current_batch_size).fill_(config.num_classes), requires_grad=False).to(device)

                    real_imgs = Variable(imgs.type(FloatTensor))
                    labels = Variable(labels.type(LongTensor))

                    z = torch.randn(current_batch_size, config.latent_dim, device=device)
                    gen_imgs = generator(z)

                    # Generator
                    with torch.set_grad_enabled(phase == 'train'):
                        optimizer_G.zero_grad()
                        validity, _ = discriminator(gen_imgs)
                        valid_g = torch.ones_like(validity, device=device)
                        g_loss = adversarial_loss(validity, valid_g)
                        if phase == 'train':
                            g_loss.backward()
                            optimizer_G.step()

                    # Discriminator
                    with torch.set_grad_enabled(phase == 'train'):
                        optimizer_D.zero_grad()

                        fake_pred, fake_aux = discriminator(gen_imgs.detach())
                        d_loss_fake = adversarial_loss(fake_pred, fake) + \
                                    auxiliary_loss(fake_aux, fake_aux_gt)

                        if config.enable_ssl:
                            labeled_mask = (labels != -1)
                            unlabeled_mask = (labels == -1)
                            exist_labeled = False
                            exist_unlabeled = False

                            if labeled_mask.any():
                                exist_labeled = True
                                labeled_imgs = real_imgs[labeled_mask]
                                labeled_labels = labels[labeled_mask]
                                real_pred, real_aux = discriminator(labeled_imgs)
                                valid_l = torch.full_like(real_pred, config.label_real, device=device)
                                d_labeled_loss = adversarial_loss(real_pred, valid_l) + \
                                                auxiliary_loss(real_aux, labeled_labels)

                            if unlabeled_mask.any():
                                exist_unlabeled = True
                                unlabeled_imgs = real_imgs[unlabeled_mask]
                                real_pred_unlabeled, _ = discriminator(unlabeled_imgs)
                                valid_u = torch.full_like(real_pred_unlabeled, config.label_real, device=device)
                                d_unlabeled_loss = adversarial_loss(real_pred_unlabeled, valid_u)

                            if exist_labeled and exist_unlabeled:
                                d_loss = d_labeled_loss + d_unlabeled_loss + d_loss_fake
                            elif exist_labeled:
                                d_loss = d_labeled_loss + d_loss_fake
                            else:
                                d_loss = d_unlabeled_loss + d_loss_fake
                        else:
                            real_pred, real_aux = discriminator(imgs)
                            valid_r = torch.full_like(real_pred, config.label_real, device=device)
                            d_loss_real = (adversarial_loss(real_pred, valid_r) + auxiliary_loss(real_aux, labels)) / 2
                            d_loss = d_loss_real + d_loss_fake/2

                        if phase == 'train':
                            d_loss.backward()
                            optimizer_D.step()
                    
                    # 累積損失加算
                    total_g_loss += g_loss.item()
                    total_d_loss += d_loss.item()
                    num_batches += 1
                
                # 平均損失を計算し記録
                avg_g_loss = total_g_loss / num_batches
                avg_d_loss = total_d_loss / num_batches
                loss_history[phase]['g_loss'].append(avg_g_loss)
                loss_history[phase]['d_loss'].append(avg_d_loss)
            
            # 1エポック終了後に fold_results に記録
            fold_results.append({
                'fold': fold,
                'epoch': epoch + 1,
                'train_g_loss': loss_history['train']['g_loss'][-1],
                'train_d_loss': loss_history['train']['d_loss'][-1],
                'valid_g_loss': loss_history['valid']['g_loss'][-1],
                'valid_d_loss': loss_history['valid']['d_loss'][-1],
            })

        # 評価
        with torch.no_grad():
            real_aux = discriminator(imgs)[1]
            accuracy, f1, qwk, report, conf_matrix, output = evaluate_predictions2(real_aux, labels, config.num_classes)
            print(conf_matrix)
            all_accuracies.append(accuracy)
            all_f1_scores.append(f1)
            all_qwk_scores.append(qwk)
            total_conf_matrix += conf_matrix

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = discriminator.state_dict()
            best_fold = fold + 1
    
    now = datetime.datetime.now()
    os.makedirs("./loss_log", exist_ok=True)
    log_path = f"./loss_log/loss_history_{now:%Y-%m-%d_%H-%M}.csv"
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(log_path, index=False)

    return all_accuracies, all_f1_scores, all_qwk_scores, total_conf_matrix, best_state, best_fold