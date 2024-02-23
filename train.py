import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import random

from utility import *
from MRIdata import MRIDataset, testDataset
from model import MRIImaging3DConvModel
from dataAugmentation import MRIDataAugmentation
from Saliencymap import *

def train(args):
    """load data"""
    trainData = MRIDataset(img_dir = args.ADNI_dir, split = 'train')
    train_loader = DataLoader(trainData, batch_size = args.batch_size, shuffle = True)
    valData = MRIDataset(img_dir = args.ADNI_dir, split = 'val')
    val_loader = DataLoader(valData, batch_size = args.batch_size, shuffle = True)

    """set up"""
    device = torch.device("cuda:"+str(args.gpu))
    model = MRIImaging3DConvModel(nClass=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    augmenter = MRIDataAugmentation(imgShape=(169, 208, 179))

    """load checkpoint if exists"""
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best', 0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Load checkpoint from epoch {start_epoch -1}")
    else:
        start_epoch = 1
        best_val_acc = 0
                
    """One epoch"""
    for epoch in tqdm(range(start_epoch, args.epochs +1)):
        """======================Train======================"""    
        model.train()
        total_train_loss = 0; total_train_correct = 0
        train_pred = []; train_true = []
        
        """One batch"""
        for batch_idx, data in enumerate(train_loader):
            images, labels = data['image'].to(device), data['label'].to(device)
            images2 = images
            
            """Generate adversarial examples by projected gradient descent (PGD)"""
            if epoch >= args.pgd_start and epoch < args.pgd_stop:
                images2 = images2.cpu().detach().numpy()
                images2 = images2  + (np.random.random(size=images2.shape) * 2 - 1) * args.pgd  # Add random noise
                images2 = torch.from_numpy(images2).float().to(device)
                grad = model.calculate_gradients(images2, labels)
                images2 = images.cpu().detach().numpy()
                for _ in range(5):
                    images2 += (args.pgd / 5) * np.sign(grad.cpu().detach().numpy())
                    images2 = np.clip(images2, images2 - args.pgd, images2 + args.pgd)  # Clip the perturbed images
                    images2 = np.clip(images2, 0, 1)  # Ensure valid pixel range
                images2 = torch.from_numpy(images2).float().to(device)

            # """Apply random masking to the image"""
            if epoch >= args.mask_start:
                if random.random()>=args.epislon:
                    images2 = augmenter.dropblock_random(images2, random.uniform(0.1, args.drop_rate))
                else:
                    grads = model.calculate_gradients(images2, labels)
                    images2 = augmenter.dropblock_grad_guided(images2, grads, random.uniform(0.1, args.drop_rate))
            grad1 = model.calculate_gradients(images2, labels)
                

            """Compute loss"""
            optimizer.zero_grad()
            # Cross-entropy loss (original image)
            predictions = model(images) 
            loss = criterion(predictions, labels)

            """contrastive loss for robustness adverserial"""
            if epoch >= args.pgd_start and epoch < args.pgd_stop:
                # Cross-entropy loss (augmented image)
                predictions1 = model(images2) 
                loss1 = criterion(predictions1, labels)
                # Contrastive loss
                loss = 0.5 * (loss1 + loss) + args.consistency * torch.norm(grad - grad1, p=2)

            """back propagation"""
            loss.backward()
            optimizer.step()

            """compute train accuracy"""
            total_train_loss += loss.item()
            _, predicted = predictions.max(1)
            total_train_correct += predicted.eq(labels).sum().item()
            train_pred.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())
        train_acc = total_train_correct / len(trainData)
        train_f1 = f1_score(train_true, train_pred, average='weighted')

        """======================Validation======================"""
        model.eval()
        total_val_loss = 0; total_val_correct = 0; val_pred = []; val_true = []
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


        """======================save model======================"""
        if not os.path.exists(f'checkpoint/state_{args.state}'):
            os.makedirs(f'checkpoint/state_{args.state}') 
        check_path = f'checkpoint/state_{args.state}/epoch_{epoch}.pth'
        if epoch % 5 ==0 or epoch >= 40:
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss,
                'best':best_val_acc,
            },check_path)


def test(args, checkpath):
    # load model
    device = torch.device("cuda:"+str(args.gpu))
    model = MRIImaging3DConvModel(nClass=2).to(device)
    checkpoint = torch.load(checkpath)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    ADNI_Data = MRIDataset(img_dir = args.ADNI_dir, split = 'test')
    ADNI_loader = DataLoader(ADNI_Data, batch_size=args.batch_size, shuffle=True)

    AIBL_Data = testDataset("aibl")
    AIBL_loader = DataLoader(AIBL_Data, batch_size=args.batch_size, shuffle=True)

    MIRIAD_Data = testDataset("miriad")
    MIRIAD_loader = DataLoader(MIRIAD_Data, batch_size=args.batch_size, shuffle=True)

    OASIS3_Data = testDataset("oasis3")
    OASIS3_loader = DataLoader(OASIS3_Data, batch_size=args.batch_size, shuffle=True)

    def run_test(loader):
        model.eval()
        total_correct = 0
        test_pred = []
        test_true = []

        with torch.no_grad():
            for _, data in enumerate(tqdm(loader)):
                images, labels = data['image'].to(device), data['label'].to(device)
                predictions = model(images)
                _, predicted = predictions.max(1)
                total_correct += predicted.eq(labels).sum().item()
                test_pred.extend(predicted.cpu().numpy())
                test_true.extend(labels.cpu().numpy())
                
        tn, fp, fn, tp = confusion_matrix(test_true, test_pred).ravel()

        SPE = tn / (tn + fp) 
        SEN = tp / (tp + fn)  

        test_acc = total_correct / len(loader.dataset)
        test_f1 = f1_score(test_true, test_pred, average='weighted')
        print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test SPE: {SPE:.4f}, Test SEN: {SEN:.4f}')

    print("Testing on ADNI data:")
    run_test(ADNI_loader)
    
    print("Testing on AIBL data:")
    run_test(AIBL_loader)

    print("Testing on MIRIAD data:")
    run_test(MIRIAD_loader)

    print("Testing on OASIS3 data:")
    run_test(OASIS3_loader)