import os
import numpy as np
import pandas as pd
import yaml
import argparse
import random
import torch
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

"""align the fastsurfer with original image"""
def crop_center(image, target_shape):
    start_idx = [(original_dim - target_dim) // 2 for original_dim, target_dim in zip(image.shape, target_shape)]
    end_idx = [start + target_dim for start, target_dim in zip(start_idx, target_shape)]
    cropped_image = image[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
    return cropped_image

"""convolution to create Saliency map"""
def gaussian_kernel(size, sigma):
    x, y, z = np.mgrid[-size:size+1, -size:size+1, -size:size+1]
    g = np.exp(-(x**2/float(size) + y**2/float(size) + z**2/float(size)) / (2*sigma**2))
    return g / g.sum()

"""threshold for smap visualization"""
def threshold_smap(smap,p):
    threshold = np.percentile(smap,p)
    smap[smap < threshold] = 0
    return smap

"""noise the masked region"""
def generate_noise(original_data, mask):
    if np.any(mask):
        mean = np.mean(original_data[mask])
        std = np.std(original_data[mask])
        noise = np.random.normal(mean, std, original_data.shape)
        return noise
    else:
        return None

"""random seed for replication"""
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

"""load yaml file to get configs"""
def load_config_to_args(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
    return args

def visualize_npy(path):
    import os
    import matplotlib.pyplot as plt
    array = np.load(path)
    array = array.squeeze()
    
    for dim, slice_array in zip(["x", "y", "z"], [array.take(indices=array.shape[i] // 2, axis=i) for i in range(3)]):
        fig, ax = plt.subplots(figsize=(slice_array.shape[1], slice_array.shape[0]), dpi=1)
        plt.imshow(slice_array, cmap='gray')
        plt.axis('off')
        output_path = f'{os.path.splitext(os.path.basename(path))[0]}_{dim}.png'
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def visualize_smap_npy(path):
    import os
    import matplotlib.pyplot as plt
    from scipy.ndimage import convolve, rotate
    array = np.load(path)
    array = array.squeeze()
    array = convolve(array, gaussian_kernel(size=8, sigma=1))
    
    for dim, slice_array in zip(["x", "y", "z"], [array.take(indices=array.shape[i] // 2, axis=i) for i in range(3)]):
        fig, ax = plt.subplots(figsize=(slice_array.shape[1], slice_array.shape[0]), dpi=1)
        plt.imshow(slice_array, cmap='grey')
        plt.axis('off')
        output_path = f'{os.path.splitext(os.path.basename(path))[0]}_{dim}_smap.png'
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def random_npy(path):
    import os
    import matplotlib.pyplot as plt
    from scipy.ndimage import convolve, rotate
    array = np.load(path)
    array = array.squeeze()
    array = np.random.random(array.shape)
    array = convolve(array, gaussian_kernel(size=8, sigma=1))
    
    for dim, slice_array in zip(["x", "y", "z"], [array.take(indices=array.shape[i] // 2, axis=i) for i in range(3)]):
        fig, ax = plt.subplots(figsize=(slice_array.shape[1], slice_array.shape[0]), dpi=1)
        plt.imshow(slice_array, cmap='grey')
        plt.axis('off')
        output_path = f'{os.path.splitext(os.path.basename(path))[0]}_{dim}_random.png'
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def visualize_pt(path):
    import os
    import matplotlib.pyplot as plt
    array = torch.load(path).cpu().numpy()
    array = array.squeeze()
    
    for dim, slice_array in zip(["x", "y", "z"], [array.take(indices=array.shape[i] // 2, axis=i) for i in range(3)]):
        fig, ax = plt.subplots(figsize=(slice_array.shape[1], slice_array.shape[0]), dpi=1)
        plt.imshow(slice_array, cmap='gray')
        plt.axis('off')
        output_path = f'{os.path.splitext(os.path.basename(path))[0]}_{dim}.png'
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
def visualize_fast(filename):
    from skimage import color
    fast_path = f'fastsurfer_seg/{filename}/mri/aparc.DKTatlas+aseg.deep.mgz'
    array = crop_center(color.label2rgb(nib.load(fast_path).get_fdata().squeeze(), bg_label=0), (169, 179, 208))
    rotateds = {0: lambda img: np.rot90(img, 3), 1: lambda img: np.rot90(img, 2), 2: lambda img: np.rot90(img,3)}

    for dim, slice_array, rotated in zip(["x", "y", "z"], [array.take(indices=array.shape[i] // 2, axis=i) for i in range(3)], [rotateds[j] for j in range(3)]):
        fig, ax = plt.subplots(figsize=(slice_array.shape[1], slice_array.shape[0]), dpi=1)
        slice_array = rotated(slice_array)
        plt.imshow(slice_array, cmap='gray')
        plt.axis('off')
        output_path = f'fast_{filename}_{dim}.png'
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    # middle_x, middle_y, middle_z = fast_data.shape[0] // 2, fast_data.shape[1] // 2, fast_data.shape[2] // 2

    # segs = [fast_data[:, :, middle_z], fast_data[middle_x, :, :], fast_data[:, middle_y, :]]
    # fig, axes = plt.subplots(1, 3, figsize=(20, 16), facecolor='black')
    # for i, seg in enumerate(segs):
    #     # seg = rotated0[i](seg)
    #     axes[i].imshow(seg, cmap='gray')
    #     axes[i].axis('off')
    # plt.subplots_adjust(wspace=1, hspace=200)
    # plt.tight_layout()
    # output_path = f"fast_{filename}.png"
    # plt.savefig(output_path)
    # plt.close(fig)
    
    

def convert_pt_to_nii(dataset):
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        image = item['image']
        participant_id = item['participant_id']
        session_id = item['session_id']
        
        save_dir_path = os.path.join('../ADdata/Data_in_Nii')
        os.makedirs(save_dir_path, exist_ok=True)
        save_path = os.path.join(save_dir_path, participant_id + '_' + session_id + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii')
        
        nii_img = nib.Nifti1Image(image.squeeze(), affine=np.eye(4))
        print(nii_img.shape)
        nib.save(nii_img, save_path)
        
def split_csv_with_label(c,merge=False):
    labels_df = df0 = pd.read_csv('../ADdata/AlzheimerImagingData/ADNI_CAPS/split.pretrained.0.csv')
    features_df = pd.read_csv(c)
    df0 = pd.DataFrame(columns=features_df.columns)
    df1 = pd.DataFrame(columns=features_df.columns)

    for index, row in features_df.iterrows():
        label = labels_df[(labels_df['participant_id'] == row['participant_id']) & (labels_df['session_id'] == row['session_id'])]['diagnosis'].values[0]
        
        if label == 'AD':
            df1 = df1._append(row, ignore_index=True)
        else:
            df0 = df0._append(row, ignore_index=True)
    if merge:
        return lh_rh_merge(df1).drop(columns=['participant_id','session_id']), lh_rh_merge(df0).drop(columns=['participant_id','session_id'])
    else:
        return df1.drop(columns=['participant_id','session_id']), df0.drop(columns=['participant_id','session_id'])

def lh_rh_merge(df):
    df_merged = pd.DataFrame()
    df_merged['participant_id'] = df['participant_id']
    df_merged['session_id'] = df['session_id']
    for column in df.columns:
        if ' (lh)' in column:
            base_column_name = column.replace(' (lh)', '')
            rh_column = column.replace(' (lh)', ' (rh)')
            
            if rh_column in df.columns:
                df_merged[base_column_name] = df[column] + df[rh_column]
            else:
                df_merged[base_column_name + ' (lh)'] = df[column]
        elif ' (rh)' in column:
            continue
        elif column not in ['participant_id', 'session_id']:
            df_merged[column] = df[column]
    return df_merged


"""hash table for fastsurfer labels"""
def label_hash():
    return {
        2: "Cortical white matter (lh)",
        4: "Lateral Ventricle (lh)",
        5: "Inferior Lateral Ventricle (lh)",
        7: "Cerebellar White Matter (lh)",
        8: "Cerebellar Cortex (lh)",
        10: "Thalamus (lh)",
        11: "Caudate (lh)",
        12: "Putamen (lh)",
        13: "Pallidum (lh)",
        14: "3rd-Ventricle",
        15: "4th-Ventricle",
        16: "Brain Stem",
        17: "Hippocampus (lh)",
        18: "Amygdala (lh)",
        24: "CSF",
        26: "Accumbens (lh)",
        28: "Ventral DC (lh)",
        31: "Choroid Plexus (lh)",
        41: "Cortical white matter (rh)",
        43: "Lateral Ventricle (rh)",
        44: "Inferior Lateral Ventricle (rh)",
        46: "Cerebellar White Matter (rh)",
        47: "Cerebellar Cortex (rh)",
        49: "Thalamus (rh)",
        50: "Caudate (rh)",
        51: "Putamen (rh)",
        52: "Pallidum (rh)",
        53: "Hippocampus (rh)",
        54: "Amygdala (rh)",
        58: "Accumbens (rh)",
        60: "Ventral DC (rh)",
        63: "Choroid Plexus (rh)",
        77: "WM-hypointensities",
        1002:"caudalanteriorcingulate (lh)",
        1003: "caudalmiddlefrontal (lh)",
        2003: "caudalmiddlefrontal (rh)",
        1005: "cuneus (lh)",
        1006: "entorhinal (lh)",
        2006: "entorhinal (rh)",
        1007: "fusiform (lh)",
        2007: "fusiform (rh)",
        1008: "inferiorparietal (lh)",
        2008: "inferiorparietal (rh)",
        1009: "inferiortemporal (lh)",
        2009: "inferiortemporal (rh)",
        1010: "isthmuscingulate (lh)",
        1011: "lateraloccipital (lh)",
        2011: "lateraloccipital (rh)",
        1012: "lateralorbitofrontal (lh)",
        1013: "lingual (lh)",
        1014: "medialorbitofrontal (lh)",
        1015: "middletemporal (lh)",
        2015: "middletemporal (rh)",
        1016: "parahippocampal (lh)",
        1017: "paracentral (lh)",
        1018: "parsopercularis (lh)",
        2018: "parsopercularis (rh)",
        1019: "parsorbitalis (lh)",
        2019: "parsorbitalis (rh)",
        1020: "parstriangularis (lh)",
        2020: "parstriangularis (rh)",
        1021: "pericalcarine (lh)",
        1022: "postcentral (lh)",
        1023: "posteriorcingulate (lh)",
        1024: "precentral (lh)",
        1025: "precuneus (lh)",
        1026: "rostralanteriorcingulate (lh)",
        2026: "rostralanteriorcingulate (rh)",
        1027: "rostralmiddlefrontal (lh)",
        2027: "rostralmiddlefrontal (rh)",
        1028: "superiorfrontal (lh)",
        1029: "superiorparietal (lh)",
        2029: "superiorparietal (rh)",
        1030: "superiortemporal (lh)",
        2030: "superiortemporal (rh)",
        1031: "supramarginal (lh)",
        2031: "supramarginal (rh)",
        1034: "transversetemporal (lh)",
        2034: "transversetemporal (rh)",
        1035: "insula (lh)",
        2035: "insula (rh)",
        2002: "caudalanteriorcingulate (rh)",
        2005: "cuneus (rh)",
        2010: "isthmuscingulate (rh)",
        2012: "lateralorbitofrontal (rh)",
        2013: "lingual (rh)",
        2014: "medialorbitofrontal (rh)",
        2016: "parahippocampal (rh)",
        2017: "paracentral (rh)",
        2021: "pericalcarine (rh)",
        2022: "postcentral (rh)",
        2023: "posteriorcingulate (rh)",
        2024: "precentral (rh)",
        2025: "precuneus (rh)",
        2028: "superiorfrontal (rh)"
    }


def move_file():
    import shutil
    original_folder = 'visualization/visualization_MCI' 
    csv_file = '../ADdata/AlzheimerImagingData/MCI_prediction_ADNI.csv'   

    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        pid = row['participant_id']
        sid = row['session_id']
        label = row['diagnosis']
        
        for dim in ['x','y','z']:
            filename = f'{pid}_{sid}_{dim}.png'
            destination_folder = os.path.join(original_folder, str(label))

            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            original_file = os.path.join(original_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            if os.path.exists(original_file):
                shutil.copy(original_file, destination_file)
            else:
                print(f'File not found: {original_file}')
                

def check_diagnosis(csv_file, pid, sid):
    """
    Check if the participant with the given pid and sid has an AD diagnosis.

    Parameters:
    - csv_file: Path to the CSV file.
    - pid: Participant ID.
    - sid: Session ID.

    Returns:
    - True if the participant has an AD diagnosis, False otherwise.
    """
    import csv
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            participant_id = f'{pid}'
            session_id = f'{sid}'
            if row['participant_id'] == participant_id and row['session_id'] == session_id:
                return row['diagnosis'] == 'AD'
    return False