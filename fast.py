import os, sys, glob, subprocess

def run_fast_single():
    SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
    FASTSURFER_HOME = SETUP_DIR + "/fastsurfer"
    print(FASTSURFER_HOME)
    NII_DIR = SETUP_DIR
    img_path = SETUP_DIR + "sub-ADNI002S0295_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii"
    
    git_repo_url = 'https://github.com/deep-mi/fastsurfer.git'
    if not os.path.exists(FASTSURFER_HOME):
        print('Clone the repository from github')
        subprocess.run(['git', 'clone', '-q', '--branch', 'stable', git_repo_url], check=True)
    else:
        print('The repository already exists')
    sys.path.append(FASTSURFER_HOME)

    # Setting the environment variable
    env = os.environ.copy()
    env['FASTSURFER_HOME'] = FASTSURFER_HOME
    
    # get the sub-dir name, for example 'sub-ADNI005S0610_ses-M12'
    sid = os.path.basename(img_path).split('_')[0:2]
    sid = '_'.join(sid)

    # Set the parameters for segmentation
    command = [
        os.path.join(FASTSURFER_HOME, 'run_fastsurfer.sh'),
        '--t1', img_path, # the t1-weighted image
        '--sd', os.path.join(SETUP_DIR, 'fastsurfer_seg'), # save dir
        '--sid', sid, # sub-dir under save dir
        '--seg_only',
        '--py', 'python3',
        '--allow_root'
    ]

    # Running the fastsurfer command
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    #Set the input and out dir
    SETUP_DIR = os.path.abspath(__file__) #TODO you need to change this to abosult dir
    FASTSURFER_HOME = SETUP_DIR + "/fastsurfer"
    NII_DIR = SETUP_DIR +"/Data_in_Nii"

    # Clone the FastSurfer repo from github
    git_repo_url = 'https://github.com/deep-mi/fastsurfer.git'
    if not os.path.exists(FASTSURFER_HOME):
        print('Clone the repository from github')
        subprocess.run(['git', 'clone', '-q', '--branch', 'stable', git_repo_url], check=True)
    else:
        print('The repository already exists')
    sys.path.append(FASTSURFER_HOME)

    # Setting the environment variable
    env = os.environ.copy()
    env['FASTSURFER_HOME'] = FASTSURFER_HOME

    # Run Segmentation task over the entire input folder
    for img_path in glob.glob(os.path.join(NII_DIR, "sub-ADNI*.nii")):

        # get the sub-dir name, for example 'sub-ADNI005S0610_ses-M12'
        sid = os.path.basename(img_path).split('_')[0:2]
        sid = '_'.join(sid)

        if os.path.exists(os.path.join(SETUP_DIR, 'fastsurfer_seg', sid)):
            print(f"Skipping {sid}, already exists")
            continue

        # Set the parameters for segmentation
        command = [
            os.path.join(FASTSURFER_HOME, 'run_fastsurfer.sh'),
            '--t1', img_path, # the t1-weighted image
            '--sd', os.path.join(SETUP_DIR, 'fastsurfer_seg'), # save dir
            '--sid', sid, # sub-dir under save dir
            '--seg_only',
            '--py', 'python3',
            '--allow_root'
        ]

        # Running the fastsurfer command
        subprocess.run(command, env=env, check=True)