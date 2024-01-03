import sys;

import easydict
import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
from tqdm import tqdm
import random
import pandas as pd
import argparse
from src import mkdir, chunks

"""
sys.argv.extend(
        [
            '-o', 'data/processed',
            '-i', 'data/BRIXIA/processed/box_data_BX.xlsx',
            '-d', 'BX',
            '-m', 'data/BRIXIA/metadata_global_v2.csv',
            '-cv', '99',
            '-l', 'True'
        ]
    )
# Add the configuration (AFC)
"""
sys.argv.extend(
    [
        '-o', 'data/processed',
        '-i', 'data/AIforCOVID/processed/box_data_AXF1.xlsx',
        '-d', 'AFC',
        '-m', 'data/AIforCOVID',
        '-cv', '99',
        '-l', 'True'
    ]
)

# Configuration file
parser = argparse.ArgumentParser(description="Configuration File")
parser.add_argument("-cv", "--fold", help="Number of folder", type=int, default=99)
parser.add_argument("-l", "--loco", help="", type=bool, default=False)
parser.add_argument("-d", "--dataset_name", help="dataset name (BX, AFC, Multi)", choices=['BX', 'AFC', 'Multi'],
                    type=str)
parser.add_argument("-o", "--output_dir", help="output directory path", default="data/processed", required=True)
parser.add_argument("-i", "--input_data", help="input directory path for data", default="data/processed", required=True)
parser.add_argument("-m", "--metadata", help="input directory path for metadata", default="data/processed",
                    required=True)
args = parser.parse_args()
cv = args.fold


def ValidationCreation():
    """

    Returns:

    """

    global classes
    print()
    data_file = args.input_data
    dataset_name = args.dataset_name
    dest_dir = os.path.join(args.output_dir, dataset_name + '_1R')
    cv = args.fold

    db = pd.read_excel(data_file, header=0, index_col="img")
    db.sort_index(inplace=True)
    label_col = "label"
    classes_centers = []
    if dataset_name == "BX":
        classes = list(db[label_col])
        label_col = "label_dim"
        scores_int = [[eval(val) for val in list(c.replace('[', '').replace(']', ''))] for c in classes]
        scores_global = [sum(s) for s in scores_int]
        classes = ['BIG' if c >= 9 else 'SMALL' for c in scores_global]
        db[label_col] = classes
        classes, classes_counts = np.unique(classes, return_counts=True)
    elif dataset_name == "AFC":
        pass


    # METADATA
    if dataset_name == 'BX':
        metadata = pd.read_csv(args.metadata, sep=';', header=0).set_index('Filename')
        metadata['Manufacturer'] = metadata['Manufacturer'].str.upper()
        metadata['Manufacturer'] = metadata['Manufacturer'].map(
            {'SIEMENS': 'SIEMENS',
             'AGFA': 'AGFA',
             'CARESTREAM HEALTH': 'CARESTREAM',
             'VILLA SISTEMI MEDICALI': 'VSM',
             'AGFA-GEVAERT': 'AGFA',
             'KODAK': 'KODAK',
             'FUJIFILM CORPORATION': 'FUJIFILM',
             'DIGITEC': 'DIGITEC'
             }
        )
        """
        (0) SIEMENS = CR (Computed Radiography)
        (1) AGFA = CR (Computed Radiography)
        (2) CARESTREAM = DX (Computed Radiography)
        (3) VSM = DX (Digital X-ray)
        (4) KODAK = CR (Computed Radiography)
        (5) FUJIFILM = CR (Computed Radiography)
        (6) DIGITEC = DX (Digital X-ray)



        """

        Centers = metadata['Manufacturer'].str.upper().unique()
        # CENTER TO ID
        Centers_to_id = {c: num for c, num in zip(Centers, range(len(Centers)))}
        id_to_Centers = {num: c for c, num in Centers_to_id.items()}
        Labels_centers = [center for center in Centers_to_id.values()]
        type_ = {center: metadata['Modality'][metadata['Manufacturer'] == center].unique() for center in
                 Centers_to_id.keys()}
        # Centers to ID
        Patients_Centers_ids = metadata['Manufacturer'].map(Centers_to_id).to_list()
        classes_centers, counts = np.unique(Patients_Centers_ids, return_counts=True)

        # Recombined Centers
        Centers_to_id['VSM'] = 2
        Centers_to_id['KODAK'] = 0
        Centers_to_id['FUJIFILM'] = 1
        Centers_to_id['DIGITEC'] = 2
        Patients_Centers_ids = metadata['Manufacturer'].map(Centers_to_id).to_list()
        classes_centers, counts = np.unique(Patients_Centers_ids, return_counts=True)
        metadata.sort_index(inplace=True)


    elif dataset_name == 'AFC':
        base_data_folder = args.metadata
        path_to_data_1 = os.path.join(base_data_folder, 'imgs')
        #path_to_data_2 = os.path.join(base_data_folder, 'imgs_r2')
        # CLINICAL DATA
        meta_path = os.path.join(base_data_folder, 'AIforCOVID.xlsx')
        #meta_path_2 = os.path.join(base_data_folder, 'AIforCOVID_r2.xlsx')
        # Clinical Data:
        clinical_meta_ = pd.read_excel(meta_path)
        #clinical_meta_2 = pd.read_excel(meta_path_2)
        #clinical_meta_global = pd.concat([clinical_meta_, clinical_meta_2])
        clinical_meta_global = clinical_meta_
        print('Clinical metadata loaded AFC')
        clinical_meta_global.set_index('ImageFile', inplace=True)
        Centers = clinical_meta_global['Hospital'].str.upper().unique()
        Centers_to_id = {c: num for c, num in zip(Centers, range(len(Centers)))}
        id_to_Centers = {num: c for c, num in Centers_to_id.items()}

        Patients_Centers_ids = clinical_meta_global['Hospital'].map(Centers_to_id)
        classes_centers, counts = np.unique(Patients_Centers_ids, return_counts=True)
        clinical_meta_global.sort_index(inplace=True)

    # LoCo Validation
    if cv == 99:
        div = len(np.unique(list(Centers_to_id.values())))
        test_split = 1
        val_split = 1
        train_split = div - test_split - val_split

    all = []
    # all
    mkdir(dest_dir)
    with open(os.path.join(dest_dir, 'all.txt'), 'w') as file:
        file.write("img label_dim scores center\n") if dataset_name == "BX" else file.write("img label center \n")
        for img in db.index:
            label = db.loc[img, label_col]
            row = str(img) + " " + str(label) + "\n" if dataset_name == "AFC" else str(img) + " " + str(label) + ' ' + \
                                                                                   db.loc[img, 'label'] + "\n"
            file.write(row)
            all.append(row)

    folds_test = [[]] * div
    folds_train_val = [[]] * div


    # Test Folds
    for i, c_c in enumerate(classes_centers):

        # Stratify By Center
        if dataset_name == "BX":
            # TEST FOLDS
            Patient_by_center = metadata['Manufacturer'] == id_to_Centers[c_c]

            Selected_Patient_by_class_center =Patient_by_center

            patient_center = [str(img) + " "+ db.loc[img, label_col] + " " + db.loc[img, 'label'] + " " + metadata.loc[
                str(img) + '.dcm', 'Manufacturer'] + "\n" for img in
                                    db.index[Selected_Patient_by_class_center].to_list()]

            # randomize
            random.seed(0)
            random.shuffle(patient_center)
            # create splits
            fold_center = list(patient_center)

            for j in range(div):
                if j == c_c:
                    folds_test[j] = folds_test[j] + fold_center

            # TRAIN VAL FOLDS
            Patient_by_other_centers = metadata['Manufacturer'] != id_to_Centers[c_c]

            # TRAIN VAL FOLDS
            Patient_by_other_centers = metadata['Manufacturer'] != id_to_Centers[c_c]

            # Select Patient by class and center
            Selected_Patient_by_class_center = Patient_by_other_centers

            patient_class_center_minus_1 = [str(img) + " " + '' + " " + db.loc[img, 'label'] + " " + metadata.loc[
                str(img) + '.dcm', 'Manufacturer'] + "\n" for img in
                                    db.index[Selected_Patient_by_class_center].to_list()]


            # randomize
            random.seed(0)
            random.shuffle(patient_class_center_minus_1)
            # create splits
            cv = div - 1
            fold_train = list(patient_class_center_minus_1)
            if len(fold_train) != cv:
                del fold_train[-1]
            for j in range(div):
                if j == c_c:
                    folds_train_val[j] = folds_train_val[j] + fold_train

        elif dataset_name == "AFC":

            Patient_by_center = clinical_meta_global['Hospital'] == id_to_Centers[c_c]
            # Select Patient by class and center
            Selected_Patient_by_class_center = Patient_by_center

            patient_center = [str(img) + " " + db.loc[img, 'label'] + " " + clinical_meta_global.loc[
                str(img), 'Hospital'] + "\n" for img in
                                    db.index[Selected_Patient_by_class_center].to_list()]

            # randomize
            random.seed(0)
            random.shuffle(patient_center)
            # create splits
            fold_center = list(patient_center)
            for j in range(div):
                if j == c_c:
                    folds_test[j] = folds_test[j] + fold_center

            Patient_by_other_centers = clinical_meta_global['Hospital'] != id_to_Centers[c_c]
            # Select Patient by class and center
            Selected_Patient_by_class_center = Patient_by_other_centers
            patient_class_center_minus_1 = [str(img) + " " + db.loc[img, 'label'] + " " + clinical_meta_global.loc[
                str(img), 'Hospital'] + "\n" for img in
                                    db.index[Selected_Patient_by_class_center].to_list()]
            # randomize
            random.seed(0)
            random.shuffle(patient_class_center_minus_1)
            # create splits
            cv = div - 1
            fold_train = list(patient_class_center_minus_1)
            if len(fold_train) != cv:
                del fold_train[-1]

            for j in range(div):

                if j == c_c:
                    folds_train_val[j] = folds_train_val[j] + fold_train


    # create split dir
    dest_dir = os.path.join(dest_dir, str(cv) if not args.loco else 'loCo')
    mkdir(dest_dir)
    print("Centers" , id_to_Centers )
    for i in range(div):
        print(id_to_Centers)
        dest_dir_cv = os.path.join(dest_dir, str(i))
        mkdir(dest_dir_cv)
        num_samples = int(len(folds_train_val[i]) * 0.1 )
        random.seed(0)
        random.shuffle(folds_train_val[i])
        train = folds_train_val[i][num_samples:]
        val = folds_train_val[i][:num_samples]
        test = folds_test[i]

        # train_CDI.txt
        with open(os.path.join(dest_dir_cv, 'train.txt'), 'w') as file:
            file.write("img label_dim scores center\n") if dataset_name == "BX" else file.write("img label center\n")
            for row in tqdm(train):
                file.write(str(row) if dataset_name == 'BX' else row)
        # val_CDI.txt
        with open(os.path.join(dest_dir_cv, 'val.txt'), 'w') as file:
            file.write("img label_dim scores center\n") if dataset_name == "BX" else file.write("img label center\n")
            for row in tqdm(val):
                file.write(str(row) if dataset_name == 'BX' else row)
        # test_CDI.txt
        with open(os.path.join(dest_dir_cv, 'test.txt'), 'w') as file:
            file.write("img label_dim scores center\n") if dataset_name == "BX" else file.write("img label center\n")
            for row in tqdm(test):
                file.write(str(row) if dataset_name == 'BX' else row)



if __name__ == '__main__':
    ValidationCreation()