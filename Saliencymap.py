import os
import argparse
import numpy as  np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import convolve, rotate
from skimage import color
from tqdm import tqdm
import torch
mpl.rcParams.update(mpl.rcParamsDefault)

from MRIdata import MRIDataset, PreMCIDataset
from torch.utils.data import DataLoader
from model import MRIImaging3DConvModel
from utility import *

"""Call this to create npy files"""
class saliency(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate(self, dataloader, name):
        self.model.eval()
        for i, data in enumerate(tqdm(dataloader, desc="Processing")):
            images, labels = data['image'].float().to(self.device), data['label'].to(self.device)
            gradients = self.model.calculate_gradients(images, labels).cpu().detach().numpy()
            # gradients = self.model.calculate_gradients(images, torch.ones_like(labels).to(self.device)).cpu().detach().numpy()

            for j, gradient in enumerate(gradients):
                if not os.path.exists(f"smap/smap_{name}"):
                    os.makedirs(f"smap/smap_{name}")
                # original_image = images[j].cpu().detach().numpy()
                subb, sess = data['participant_id'][j], data['session_id'][j]
                # original_image_path = f"original_image/{subb}_{sess}.npy"
                # np.save(original_image_path, original_image)
                gradient_path = f"smap/smap_{name}/{subb}_{sess}.npy"
                np.save(gradient_path, gradient)

def smap_generate(name = '', config='', checkpath = '', MCI = 0):
    if not os.path.exists(f"smap/smap_{name}"):
        os.makedirs(f"smap/smap_{name}")
    if not os.path.exists("original_image"):
        os.makedirs("original_image")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default=config)
    args = parser.parse_args()
    args = load_config_to_args('configs/' + args.config)
    set_seed(args.seed)
    device = torch.device("cuda:"+str(args.gpu))

    # Load trained model
    checkpoint = torch.load(checkpath)
    model = MRIImaging3DConvModel(nClass=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    for split in ['train', 'val', 'test']:
        if MCI !=0:
            ADNI_Data = PreMCIDataset(img_dir = args.ADNI_dir, split = split)
        else:
            ADNI_Data = MRIDataset(img_dir = args.ADNI_dir, split = split)
        ADNI_loader = DataLoader(ADNI_Data, batch_size=args.batch_size, shuffle=True)
        generator = saliency(model, device)
        generator.generate(ADNI_loader, name)

def visualize_slices_and_attention(p, size, sigma, smap_dir, output_dir, original_img_dir ='original_image'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    smap_files = [f for f in os.listdir(smap_dir) if f.endswith(".npy")]
    for smap_file in tqdm(smap_files, desc="Processing saliency maps"):
        output_path = os.path.join(output_dir, smap_file.replace(".npy", f"_x.png"))
        if os.path.exists(output_path):
            continue
        smap_path = os.path.join(smap_dir, smap_file)
        smap = np.load(smap_path)
        smap = np.squeeze(smap)
        smap = convolve(smap, gaussian_kernel(size, sigma))
        if not os.path.exists(os.path.join(original_img_dir, smap_file)):
            print(f"Original image {smap_file} does not exist. Skipping.")
            continue
        original_image = np.load(os.path.join(original_img_dir, smap_file))[0]

        for dim, middle in zip(["x", "y", "z"], [smap.shape[i] // 2 for i in range(3)]):
            smap_slice = smap.take(indices=middle, axis='xyz'.index(dim))
            smap_slice = threshold_smap(smap_slice, p)
            orig_slice = original_image.take(indices=middle, axis='xyz'.index(dim))
            
            fig, ax = plt.subplots(figsize=(smap_slice.shape[1], smap_slice.shape[0]), dpi=1)
            plt.imshow(smap_slice, cmap='jet', alpha=1)
            plt.imshow(orig_slice, cmap='gray', alpha=0.7)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            output_path = os.path.join(output_dir, smap_file.replace(".npy", f"_{dim}.png"))
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=1)
            plt.close()
            
def avg_smap(p, size, sigma, smap_dir):
    csv_file ='../ADdata/AlzheimerImagingData/ADNI_CAPS/split.pretrained.0.csv'

    smap_files = [f for f in os.listdir(smap_dir) if f.endswith(".npy")]
    sum_ad_smap = None
    num_ad_files = 0
    sum_nc_smap = None
    num_nc_files = 0
    
    for smap_file in tqdm(smap_files, desc="Processing saliency maps"):
        smap_path = os.path.join(smap_dir, smap_file)
        pid, sid = smap_file.split('.')[0].split('_')
        smap = np.load(smap_path)
        smap = np.squeeze(smap)
        if sum_ad_smap is None:
                sum_ad_smap = np.zeros_like(smap)
        if sum_nc_smap is None:
            sum_nc_smap = np.zeros_like(smap)
        if check_diagnosis(csv_file,pid,sid):
            sum_ad_smap += smap
            num_ad_files +=1
        else:
            sum_nc_smap += smap
            num_nc_files +=1
        
    print(num_ad_files, num_nc_files)    
    avg_ad_smap = sum_ad_smap / num_ad_files
    avg_nc_smap = sum_nc_smap / num_nc_files
    
    avg_ad_smap = convolve(avg_ad_smap, gaussian_kernel(size, sigma))
    avg_nc_smap = convolve(avg_nc_smap, gaussian_kernel(size, sigma))

    np.save('avg_ad_smap.npy', avg_ad_smap)
    np.save('avg_nc_smap.npy', avg_nc_smap)

def csv_create_segmented_smap(csv_name, name, nomalize=False):
    label_mapping = label_hash()
    overlap_results =[]
    sorted_labels = sorted(label_mapping.keys())
    
    with open(csv_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['participant_id', 'session_id'] + [label_mapping[label] for label in sorted_labels]
        csv_writer.writerow(header)

        for item in tqdm(os.scandir('fastsurfer_seg'), total=sum(1 for _ in os.scandir('fastsurfer_seg')), desc='previous state csv'):
            # get the segmentation and smap
            participant_id, session_id = item.name.split('_')[0], item.name.split('_')[1]
            segmentation_path = f'fastsurfer_seg/{participant_id}_{session_id}/mri/aparc.DKTatlas+aseg.deep.mgz'
            saliency_map = f'smap/smap_{name}/{participant_id}_{session_id}.npy'
            if not os.path.exists(segmentation_path) or os.path.getsize(segmentation_path) == 0 or not os.path.exists(saliency_map):
                continue
            seg_data = nib.load(segmentation_path).get_fdata().squeeze()
            smap_data = rotate(np.load(saliency_map).squeeze(), 90, axes=(1,2))
            seg_data = crop_center(seg_data, smap_data.shape)

            # get the segmentation labels, here the same segmented region has the same value(label)
            label_overlap = {}
            unique_labels, unique_counts = np.unique(seg_data, return_counts=True)

            for label in unique_labels:
                if label == 0: # 0 is background
                    continue
                label_mask = (seg_data == label)
                overlap = np.sum(np.abs(smap_data * label_mask)) if nomalize==False else np.sum(np.abs(smap_data * label_mask))
                label_overlap[label] = overlap

            overlap_results.append(label_overlap)

            # Ensure values are in natural order of labels
            ordered_values = [label_overlap.get(label, 0) for label in sorted_labels]
            row = [participant_id, session_id] + ordered_values
            csv_writer.writerow(row)

        df = pd.read_csv(csv_name)
        df = df.sort_values(by=['participant_id', 'session_id'])
        df.to_csv(csv_name, index=False)

def csv_count_segmented_map(csv_name, name, nomalize=False):
    label_mapping = label_hash()
    overlap_results =[]
    sorted_labels = sorted(label_mapping.keys())
    
    with open(csv_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['participant_id', 'session_id'] + [label_mapping[label] for label in sorted_labels]
        csv_writer.writerow(header)

        for item in tqdm(os.scandir('fastsurfer_seg'), total=sum(1 for _ in os.scandir('fastsurfer_seg')), desc='previous state csv'):
            # get the segmentation
            participant_id, session_id = item.name.split('_')[0], item.name.split('_')[1]
            segmentation_path = f'fastsurfer_seg/{participant_id}_{session_id}/mri/aparc.DKTatlas+aseg.deep.mgz'
            if not os.path.exists(segmentation_path) or os.path.getsize(segmentation_path) == 0:
                continue
            seg_data = nib.load(segmentation_path).get_fdata().squeeze()

            label_count = {}
            unique_labels, unique_counts = np.unique(seg_data, return_counts=True)

            for label, count in zip(unique_labels, unique_counts):
                if label == 0: # 0 is background
                    continue
                label_count[label] = count

            overlap_results.append(label_count)

            # Ensure values are in natural order of labels
            ordered_values = [label_count.get(label, 0) for label in sorted_labels]
            row = [participant_id, session_id] + ordered_values
            csv_writer.writerow(row)

        df = pd.read_csv(csv_name)
        df = df.sort_values(by=['participant_id', 'session_id'])
        df.to_csv(csv_name, index=False)


def bar_line_chart_merged(csv_name, csv_name2, check):
    df = lh_rh_merge(pd.read_csv(csv_name)).drop(columns=['participant_id','session_id'])
    df2 = lh_rh_merge(pd.read_csv(csv_name2)).drop(columns=['participant_id','session_id'])
    
    df_1, df_0 =split_csv_with_label(csv_name, merge=True)
    df2_1, df2_0 =split_csv_with_label(csv_name2, merge=True)

    column_sums_1 = df_1.sum() / df2_1.sum()
    column_proportions_1 = (column_sums_1 / column_sums_1.sum()) * 100
    column_proportions_1 = column_proportions_1.sort_values(ascending=False)
    
    column_sums_0 = df_0.sum() / df2_0.sum()
    column_proportions_0 = (column_sums_0 / column_sums_0.sum()) * 100
    column_proportions_0 = column_proportions_0.sort_values(ascending=False)

    column_sums = df.sum() / df2.sum()
    column_proportions = (column_sums / column_sums.sum()) * 100
    column_proportions = column_proportions.sort_values(ascending=False)

    plt.figure(figsize=(20, 10))
    column_proportions.plot(kind='bar', color='skyblue', alpha=0.6)
    column_proportions.plot(kind='line', color='red', marker='o', linewidth=2, markersize=5)
    plt.title("model's Saliency of different Brain Parts across all the testing data", fontsize=18)
    plt.xlabel('Brain Parts', fontsize=15)
    plt.ylabel('Percentage (%)', fontsize=15)
    plt.xticks(rotation=90)
    plt.savefig(f'cognitive/seg_chart_{check}_merge.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(20, 10))
    column_proportions_1.plot(kind='bar', color='skyblue', alpha=0.6)
    column_proportions_1.plot(kind='line', color='red', marker='o', linewidth=2, markersize=5)
    plt.title("model's Saliency of different Brain Parts across all the AD data", fontsize=18)
    plt.xlabel('Brain Parts', fontsize=15)
    plt.ylabel('Percentage (%)', fontsize=15)
    plt.xticks(rotation=90)
    plt.savefig(f'cognitive/seg_chart_{check}_AD_merge.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(20, 10))
    column_proportions_0.plot(kind='bar', color='skyblue', alpha=0.6)
    column_proportions_0.plot(kind='line', color='red', marker='o', linewidth=2, markersize=5)
    plt.title("model's Saliency of different Brain Parts across all the non-AD data", fontsize=18)
    plt.xlabel('Brain Parts', fontsize=15)
    plt.ylabel('Percentage (%)', fontsize=15)
    plt.xticks(rotation=90)
    plt.savefig(f'cognitive/seg_chart_{check}_CN_merge.png', bbox_inches='tight', dpi=300)
    plt.close()


def bar_line_chart(csv_name, csv_name2, check):
    df = pd.read_csv(csv_name).drop(columns=['participant_id','session_id'])
    df2 = pd.read_csv(csv_name2).drop(columns=['participant_id','session_id'])
    
    df_1, df_0 =split_csv_with_label(csv_name)
    df2_1, df2_0 =split_csv_with_label(csv_name2)
    
    column_sums_1 = df_1.sum() / df2_1.sum()
    column_proportions_1 = (column_sums_1 / column_sums_1.sum()) * 100

    column_sums_0 = df_0.sum() / df2_0.sum()
    column_proportions_0 = (column_sums_0 / column_sums_0.sum()) * 100
    
    # column_sums = df.sum() / df2.sum()
    # column_proportions = (column_sums / column_sums.sum()) * 100
    
    diff = abs(column_proportions_1-column_proportions_0)
    diff = diff.sort_values(ascending=False)
    
    
    # column_proportions_0 = column_proportions_0.sort_values(ascending=False)
    # column_proportions_1 = column_proportions_1.sort_values(ascending=False)
    
    
    
    
    # sorted_index = column_proportions_0.index
    # column_proportions_1 = column_proportions_1.reindex(sorted_index)
    # column_proportions = column_proportions.reindex(sorted_index)
    
    
    plt.figure(figsize=(20, 10))
    diff.plot(kind='bar', color='skyblue', alpha=0.6)
    diff.plot(kind='line', color='red', marker='o', linewidth=2, markersize=5)
    plt.title("Difference of model's Saliency between AD and NC", fontsize=18)
    plt.xlabel('Brain Parts', fontsize=15)
    plt.ylabel('Percentage (%)', fontsize=15)
    plt.xticks(rotation=90)
    plt.savefig(f'cognitive/seg_chart_{check}_diff.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # plt.figure(figsize=(20, 10))
    # column_proportions_1.plot(kind='bar', color='skyblue', alpha=0.6)
    # column_proportions_1.plot(kind='line', color='red', marker='o', linewidth=2, markersize=5)
    # plt.title("model's Saliency of different Brain Parts across all the AD data", fontsize=18)
    # plt.xlabel('Brain Parts', fontsize=15)
    # plt.ylabel('Percentage (%)', fontsize=15)
    # plt.xticks(rotation=90)
    # plt.savefig(f'cognitive/seg_chart_{check}_AD.png', bbox_inches='tight', dpi=300)
    # plt.close()
    
    # plt.figure(figsize=(20, 10))
    # column_proportions_0.plot(kind='bar', color='skyblue', alpha=0.6)
    # column_proportions_0.plot(kind='line', color='red', marker='o', linewidth=2, markersize=5)
    # plt.title("model's Saliency of different Brain Parts across all the non-AD data", fontsize=18)
    # plt.xlabel('Brain Parts', fontsize=15)
    # plt.ylabel('Percentage (%)', fontsize=15)
    # plt.xticks(rotation=90)
    # plt.savefig(f'cognitive/seg_chart_{check}_CN.png', bbox_inches='tight', dpi=300)
    # plt.close()

def visualizations(filename):
    directory_path = 'visualization/combined_visualization'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    fast_path = f'fastsurfer_seg/{filename}/mri/aparc.DKTatlas+aseg.deep.mgz'
    fast_data = crop_center(color.label2rgb(nib.load(fast_path).get_fdata().squeeze(), bg_label=0), (169, 179, 208))
    middle_x, middle_y, middle_z = fast_data.shape[0] // 2, fast_data.shape[1] // 2, fast_data.shape[2] // 2

    segs = [fast_data[:, :, middle_z], fast_data[middle_x, :, :], fast_data[:, middle_y, :]]
    slices = [f"visualization/visualized_slice/{filename}_y.png", f"visualization/visualized_slice/{filename}_x.png", f"visualization/visualized_slice/{filename}_z.png"]
    attens = [f"visualization/visualized_attention/{filename}_y.png", f"visualization/visualized_attention/{filename}_x.png", f"visualization/visualized_attention/{filename}_z.png", ]
    rotated0 = {0: lambda img: np.rot90(img, 3), 1: lambda img: np.rot90(img, 4), 2: lambda img: np.rot90(img,3)}
    rotated1 = {0: lambda img: np.rot90(img, 1), 1: lambda img: np.rot90(img, 1), 2: lambda img: np.rot90(np.flipud(img),3)}

    fig, axes = plt.subplots(3, 3, figsize=(20, 16), facecolor='black')
    for i, seg in enumerate(segs):
        seg = rotated0[i](seg)
        axes[2][i].imshow(seg, cmap='gray')
        axes[2][i].axis('off')

    for i, path_list in enumerate([slices, attens]):
        for j, img_path in enumerate(path_list):
            img = plt.imread(img_path)
            img = rotated1[j](img)
            axes[i][j].imshow(img, cmap='gray')
            axes[i][j].axis('off')

    plt.subplots_adjust(wspace=1, hspace=200)
    plt.tight_layout()
    output_path = f"visualization/combined_visualization/{filename}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def combined_visual_iter_dir(directory_path='smap'):
    all_files = os.listdir(directory_path)
    file_names = [os.path.splitext(file)[0] for file in all_files if os.path.isfile(os.path.join(directory_path, file))]
    for filename in tqdm(file_names):
        if os.path.exists(f"visualization/combined_visualization/{filename}.png"):
            continue
        else:
            try:
                visualizations(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

if __name__ == "__main__":
    "create combined 3 x 3 images for origin_smap_seg"
    combined_visual_iter_dir(directory_path=f'smap')