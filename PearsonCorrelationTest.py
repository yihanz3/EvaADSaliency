import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import pearsonr
from utility import *


def pearson_test_with_segmentation_difference(check):
    csv0 = f'cognitive/sgsm_base.csv'
    csv1 = 'cognitive/segmentation_difference.csv'
    csv2 = f'cognitive/saliency_segmented_diff_base.csv'
    if not os.path.exists(csv1):
        seg_diff(check=check)
    m1, m2 = csv_to_matrix(csv1), csv_to_matrix(csv2)
    correlations0 = compute_pearson(abs(m1), abs(m2))
    
    csv0 = f'cognitive/sgsm_pgd.csv'
    csv2 = f'cognitive/saliency_segmented_diff_pgd.csv'
    if not os.path.exists(csv2):
        csv_merge_filter(csv1, csv0, name='diff', check='pgd')
    m1, m2 = csv_to_matrix(csv1), csv_to_matrix(csv2)
    correlations1 = compute_pearson(abs(m1), abs(m2))
    
    csv0 = f'cognitive/sgsm_{check}.csv'
    csv2 = f'cognitive/saliency_segmented_diff_{check}.csv'
    if not os.path.exists(csv2):
        csv_merge_filter(csv1, csv0, name='diff', check=check)
    m1, m2 = csv_to_matrix(csv1), csv_to_matrix(csv2)
    correlations2 = compute_pearson(abs(m1), abs(m2))
    
    """Box plot"""
    plt.figure(figsize=(10, 6))
    bplot=plt.boxplot([correlations0,correlations1, correlations2], labels=['Base', 'FGSM', 'FGSM+mask'], showfliers=False, patch_artist=False, boxprops=dict(linewidth=1, color='black'))
    plt.ylabel('Pearson Correlation')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('FM3.png', dpi=300)
    plt.show()

def get_variable_name(variable, local_vars):
    for name, value in local_vars.items():
        if value is variable:
            return name

def df_split(column_keep, name):
    df = pd.read_csv('cognitive/ADNIMERGE_15Sep2023.csv', low_memory=False)
    df = df[column_keep]
    df['VISCODE'] = df['VISCODE'].replace('bl', 'm00')
    df['VISCODE'] = df['VISCODE'].str.replace('m', 'ses-M')
    df['PTID'] = 'sub-ADNI' + df['PTID'].str.replace('_', '')
    df = df.rename(columns={'PTID': 'participant_id'})
    df = df.rename(columns={'VISCODE': 'session_id'})
    df.dropna(inplace=True)
    try:
        df = df.rename(columns={'LDELTOTAL_BL': 'LDELTOTAL_bl'})
    except KeyError:
        pass
    df.to_csv(f'cognitive/ADASMERGE_{name}.csv', index=False) 

def compute_difference(seg_map1, seg_map2):
    unique_labels = np.unique(seg_map1)
    differences = {}
    for label in unique_labels:
        if label == 0:
            continue
        size_diff = np.sum(seg_map1 == label) - np.sum(seg_map2 == label)
        differences[label] = size_diff
    return differences

def seg_diff(check):
    print("create csv for segmentation difference")
    with open(f'cognitive/saliency_segmented_{check}.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        with open('cognitive/segmentation_difference.csv', 'w', newline='') as result_file:
            writer = csv.DictWriter(result_file,fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in tqdm(reader):
                participant_id = row['participant_id']
                session_id = row['session_id']
                if session_id != 'ses-M00':
                    seg1_path = f'fastsurfer_seg/{participant_id}_ses-M00/mri/aparc.DKTatlas+aseg.deep.mgz'
                    seg2_path = f'fastsurfer_seg/{participant_id}_{session_id}/mri/aparc.DKTatlas+aseg.deep.mgz'
                    if not os.path.exists(seg1_path) or not os.path.exists(seg2_path) or os.path.getsize(seg1_path) == 0 or os.path.getsize(seg2_path) == 0:
                        continue
                    seg1, seg2 = nib.load(seg1_path).get_fdata().squeeze(), nib.load(seg2_path).get_fdata().squeeze()
                    d = compute_difference(seg1, seg2)
                    for k, v in d.items():
                        label_description = label_hash().get(k)
                        if label_description:
                            row[label_description] = v                            
                    writer.writerow(row)

def compute_pearson(matrix_A, matrix_B):    
    num_patients = matrix_A.shape[0]
    correlations = []
    for i in range(num_patients):
        corr = abs((pearsonr(matrix_A[i], matrix_B[i])[0]))
        correlations.append(corr)
    return correlations       

def csv_merge_filter(csv1, csv2, name, check):
    df1, df2= pd.read_csv(csv1), pd.read_csv(csv2)
    df1 = df1[df1.set_index(['participant_id', 'session_id']).index.isin(df2.set_index(['participant_id', 'session_id']).index)]
    df2 = df2[df2.set_index(['participant_id', 'session_id']).index.isin(df1.set_index(['participant_id', 'session_id']).index)]
    df1 = df1.sort_values(by=['participant_id', 'session_id'])
    df1.to_csv(csv1, index=False)
    df2.to_csv(f'cognitive/saliency_segmented_{name}_{check}.csv', index=False)

def csv_to_matrix(csv_file_path):
    df = pd.read_csv(csv_file_path)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    matrix = numeric_df.values
    return matrix

def corr_heatmap(A, B, x_labels, y_labels):
    norm_A = np.linalg.norm(A, axis=0)
    norm_B = np.linalg.norm(B, axis=0)
    correlation_matrix = np.dot(A.T, B) / np.outer(norm_A, norm_B)
    # Get the indices of the top 10 values in each row
    top_10_indices = np.argsort(-correlation_matrix, axis=1)[:, :10]
    top_10_data = []
    for i, indices in enumerate(top_10_indices):
        top_10_brain_regions = [x_labels[idx] for idx in indices]
        top_10_data.append([y_labels[i]] + top_10_brain_regions)
    top_10_df = pd.DataFrame(top_10_data, columns=['Experiment'] + [f'Top_{j+1}' for j in range(10)])
    top_10_df.to_csv('cognitive/top_10_correlations.csv', index=False)
    plt.figure(figsize=(25, 15))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt="d", cbar=False, square=True, xticklabels=x_labels, yticklabels=y_labels)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('cognitive/corr_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    pearson_test_with_segmentation_difference(check=111)
