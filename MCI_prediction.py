import torch
import random
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from model import MRIImaging3DConvModel
from MRIdata import PreMCIDataset
from utility import *
from dataAugmentation import *

def generate_dataset():
    df = pd.read_csv('../ADdata/AlzheimerImagingData/ADNI_CAPS/split.pretrained.0.csv')
    patients = df[df['diagnosis'] == 'MCI']['participant_id'].unique()
    new_df_list = []

    for patient in patients:
        patient_data = df[df['participant_id'] == patient]
        last_diagnosis = patient_data.iloc[-1]['diagnosis']
        first_data = patient_data.iloc[0].copy()
        first_data['diagnosis'] = last_diagnosis
        new_df_list.append(first_data)

    new_df = pd.concat(new_df_list, axis=1).transpose()
    new_df.sort_values(by=['participant_id'], inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    np.random.seed(1)
    new_df['split'] = np.random.choice(['train', 'val', 'test'], size=len(new_df), p=[0.8, 0.1, 0.1])
    new_df.to_csv('../ADdata/AlzheimerImagingData/prediction_ADNI.csv', index=False)

def train_mci(args):
    device = torch.device("cuda:"+str(args.gpu) if args.gpu is not None and torch.cuda.is_available() else "cpu")
    trainData = PreMCIDataset(img_dir = args.ADNI_dir, split = 'train')
    train_loader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)
    valData = PreMCIDataset(img_dir = args.ADNI_dir, split = 'val')
    val_loader = DataLoader(valData, batch_size=args.batch_size, shuffle=True)
    model = MRIImaging3DConvModel(nClass=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Load checkpoint from epoch {start_epoch -1}")

    for epoch in tqdm(range(start_epoch, args.epochs +1)):
        # train
        model.train()
        for batch_idx, data in tqdm(enumerate(train_loader)):
            images, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        val_pred = []
        val_true = []
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                images, labels = data['image'].to(device), data['label'].to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)

                total_val_loss += loss.item()
                _,predicted = predictions.max(1)
                total_val_correct += predicted.eq(labels).sum().item()
                val_pred.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        val_acc = total_val_correct / len(valData)
        val_f1 = f1_score(val_true, val_pred, average='weighted')

        print(f'Epoch: {epoch}')
        print(f'Val Loss: {total_val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        if not os.path.exists('checkpoint/MCI'):
            os.makedirs('checkpoint/MCI')
        check_path = f'checkpoint/MCI/epoch_{epoch}.pth'
        torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss,
            },check_path)

def test_mci(args, checkpoint):
    # load model
    device = torch.device("cuda:"+str(args.gpu))
    model = MRIImaging3DConvModel(nClass=2).to(device)
    checkpoint = torch.load(checkpoint)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    ADNI_Data = PreMCIDataset(img_dir = args.ADNI_dir, split = 'test')
    ADNI_loader = DataLoader(ADNI_Data, batch_size=args.batch_size, shuffle=True)

    def run_test(loader):
        model.eval()
        total_correct = 0
        test_pred = []
        test_true = []

        with open('failed_cases.txt','w') as file:
            with torch.no_grad():
                for _, data in tqdm(enumerate(loader)):
                    images, labels = data['image'].to(device), data['label'].to(device)
                    identifiers = [f"{a}_{b}" for a, b in zip(data['participant_id'], data['session_id'])]
                    predictions = model(images)
                    _, predicted = predictions.max(1)
                    total_correct += predicted.eq(labels).sum().item()
                    test_pred.extend(predicted.cpu().numpy())
                    test_true.extend(labels.cpu().numpy())
                    
                    preds = predicted.cpu().numpy()
                    trues = labels.cpu().numpy()
                    
                    print(identifiers)
                    for idf, p, t in zip(identifiers, preds, trues):
                        if p != t:
                            file.write(f'Index: {idf}, Predicted: {p}, True: {t}\n')

            test_acc = total_correct / len(loader.dataset)
            test_f1 = f1_score(test_true, test_pred, average='weighted')
            print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

    print("Testing MCI prediction on ADNI data:")
    run_test(ADNI_loader)
    
def predict_one(args, checkpoint):
    device = torch.device("cuda:"+str(args.gpu))
    model = MRIImaging3DConvModel(nClass=2).to(device)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    augmenter = MRIDataAugmentation(imgShape=(169, 208, 179))

    ADNI_Data = PreMCIDataset(img_dir = args.ADNI_dir, split = 'test')
    idx = random.randint(0, len(ADNI_Data) - 1)
    data = ADNI_Data[idx]
    image = torch.from_numpy(data['image']).float().unsqueeze(0).to(device)
    print(image.shape)
    print(torch.load(data['image_path']).shape)
    i2=augmenter.augmentData_single(torch.load(data['image_path']))
    print(i2.shape)
    pa_id, ses_id = data['participant_id'], data['session_id']
    
    model.eval()
    gradients = model.calculate_gradients(image, torch.tensor(1).unsqueeze(0).to(device)).cpu().detach().numpy()
    
    original_image = image[0][0].cpu().detach().numpy() # (169, 208, 179)
    smap = convolve(gradients[0][0], gaussian_kernel(size = 8, sigma = 1)) # (169, 208, 179)
    for dim, middle in zip(["x", "y", "z"], [smap.shape[i] // 2 for i in range(3)]):
        smap_slice = smap.take(indices=middle, axis='xyz'.index(dim))
        smap_slice = threshold_smap(smap_slice, p=0)
        orig_slice = original_image.take(indices=middle, axis='xyz'.index(dim))

        fig, ax = plt.subplots(figsize=(smap_slice.shape[1], smap_slice.shape[0]), dpi=1)
        plt.imshow(smap_slice, cmap='jet', alpha=1)
        plt.imshow(orig_slice, cmap='gray', alpha=0.7)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        output_path = os.path.join(f"demo_{dim}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=1)
        plt.close()
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1).cpu()
        ad_prob = probabilities[0, 1].item()
        print(f"Prediction of developing to AD within 5 years is: {ad_prob*100:.2f}%")
        
    df = pd.read_csv('cognitive/saliency_segmented.csv')

    filtered_rows = df[(df['participant_id'] == data['participant_id']) & (df['session_id'] == data['session_id'])]
    print(data['participant_id'], data['session_id'])
    if not filtered_rows.empty:
        print(filtered_rows)
