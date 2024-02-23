import torch
import pandas as pd
import numpy as np
from os.path import join
from torch.utils.data import Dataset
import torch.utils.data
import torch.utils.data.sampler
from dataAugmentation import MRIDataAugmentation

class MRIDataset(Dataset):
    def __init__(self, img_dir, split):  # 'train' or 'val' or 'test'
        self.img_dir = img_dir
        self.split = split
        idx_fold = 0
        self.dim =(169, 208, 179)
        df_path = join('../ADdata/AlzheimerImagingData/ADNI_CAPS', f'split.pretrained.{idx_fold}.csv')
        df = pd.read_csv(df_path)
        df = df[df['split']==split]
        df = df[df['diagnosis'] != 'MCI'] # To skip the MCI data
        self.df = df.reset_index()
        self.diagnosis_code = {'CN': 0, 'AD': 1}
        self.dataAugmentation = MRIDataAugmentation(self.dim, 0.5)

    def __len__(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        img_label = self.df.loc[idx, 'diagnosis']
        image_path = join(self.img_dir, 'subjects', img_name, sess_name, 'deeplearning_prepare_data', 'image_based', 't1_linear', img_name + '_' + sess_name + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')
        if self.img_dir != "../ADdata/AlzheimerImagingData/ADNI_CAPS":
            image_path = join(self.img_dir, img_name + '_' + sess_name + '_augmented.pt')
        # image = self.normalized_image(image_path) # (1, 169, 208, 179)
        if self.split == 'train':
            image = self.dataAugmentation.augmentData_single(torch.load(image_path))
        # print(image.shape)
        image = self.normalized_image(image_path)
        label = self.diagnosis_code[img_label]

        return {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}
        
    def normalized_image(self, image_path):
        "return image into a torch tensor range from [0,1]"
        d = torch.load(image_path).cpu().numpy().astype(np.float32)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        return d

class testDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = f"../ADdata/AlzheimerImagingData/{img_dir.upper()}_CAPS/"
        self.dim =(169, 208, 179)
        df = pd.read_csv(self.img_dir + f'{img_dir}_test_info.csv')
        
        df = df[df['diagnosis'] != 'MCI']
        self.df = df.reset_index()
        self.diagnosis_code = {'CN': 0, 'AD': 1}

    def __len__(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        img_label = self.df.loc[idx, 'diagnosis']

        image_path = join(self.img_dir, 'subjects', img_name, sess_name, 'deeplearning_prepare_data', 'image_based', 't1_linear', img_name + '_' + sess_name + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')
        # if self.img_dir != "../ADdata/AlzheimerImagingData/ADNI_CAPS":
        #     image_path = join(self.img_dir, img_name + '_' + sess_name + '_augmented.pt')
        image = self.normalized_image(image_path)
        label = self.diagnosis_code[img_label]

        return {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}
        
    def normalized_image(self, image_path):
        "return image into a torch tensor range from [0,1]"
        d = torch.load(image_path).cpu().numpy().astype(np.float32)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        return d
    
class PreMCIDataset(Dataset):
    def __init__(self, img_dir, split, 
                dim = (169, 208, 179)
                 ):
        self.img_dir = img_dir
        self.split = split
        df = pd.read_csv('../ADdata/AlzheimerImagingData/MCI_prediction_ADNI.csv')
        df = df[df['split']==split]

        self.df = df.reset_index()
        self.diagnosis_code = {'MCI': 0, 'AD': 1, 'CN': 0}
        self.dataAugmentation = MRIDataAugmentation(dim, 0.5)

    def __len__(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        img_label = self.df.loc[idx, 'diagnosis']
        image_path = join(self.img_dir, 'subjects', img_name, sess_name, 'deeplearning_prepare_data', 'image_based', 't1_linear', img_name + '_' + sess_name + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')
        # image = self.normalized_image(image_path) # (1, 169, 208, 179)
        if self.split == 'train':
            image = self.dataAugmentation.augmentData_single(torch.load(image_path))
        # print(image.shape)
        image = self.normalized_image(image_path)
        label = self.diagnosis_code[img_label]

        return {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}
        
    def normalized_image(self, image_path):
        "return image into a torch tensor range from [0,1]"
        d = torch.load(image_path).cpu().numpy().astype(np.float32)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        return d