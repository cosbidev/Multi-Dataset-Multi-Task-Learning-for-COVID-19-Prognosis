import os
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Configuration File")
parser.add_argument("-n", "--release_number", help="Release Number", type=int, choices=[1,2,3])
args = parser.parse_args()
# Defining the multi-level column structure
exps = ["AFC", "BX_brixia_Lung", "BX_brixia_Global", "BX_regression", "Multi_regression", "Multi_Global", "Multi_Lung"]
folds = ["5", "loCo"]
folds_multi = ["5", "loCo6"]
EXP_NAME = f"{args.release_number}release"


regex = rf"{EXP_NAME}(?!.*32b)"
path_to_search = "/mimer/NOBACKUP/groups/naiss2023-6-336/fruffini/COVID-19-MultiTask/models"

col_names = ["AFC", "brixia_Lung", "brixia_Global", "regression", "Multi_regression", "Multi_Global", "Multi_Lung"]

columns = pd.MultiIndex.from_product([col_names[:4], folds])
columns_multi = pd.MultiIndex.from_product([col_names[4:], folds_multi])


columns = columns.append(columns_multi)
# Defining the model names for the rows
models = ["densenet121_CXR",
 "densenet121",
 "densenet201",
 "densenet161",
 "densenet169",
 "googlenet",
 "mobilenet_v2",
 "resnet18",
 "resnet34",
 "resnet50",
 "resnet50_ChestX-ray14",
 "resnet50_ChexPert",
 "resnet50_ImageNet_ChestX-ray14",
 "resnet50_ImageNet_ChexPert",
 "resnet101",
 "resnext50_32x4d",
 "shufflenet_v2_x0_5",
 "shufflenet_v2_x1_0",
 "wide_resnet50_2",
 "efficientnet_lite0",
 "efficientnet_b0",
 "efficientnet_es",
 "efficientnet_es_pruned",
 "efficientnet_b1_pruned",
 ]

# Creating an empty DataFrame with the specified multiindex columns and models as rows
df = pd.DataFrame(index=models, columns=columns)
for exp in exps:
    name_dir = exp
    if 'brixia' in exp or ('regression' in exp and 'BX' in exp):
        folder_exp = exp

        name_dir = folder_exp[3:]

        print(name_dir)
        exp = 'BX'
        regex = rf"{name_dir}(?!.*32b)"
    elif 'Multi' in exp:
        folder_exp = exp
        name_dir = folder_exp
        exp = 'Multi'
        if 'Global' in name_dir:
            regex = rf"{EXP_NAME + '_brixia_Global'}(?!.*32b)(?!.*AAA)"
        elif 'Lung' in name_dir:
            regex = rf"{EXP_NAME + '_brixia_Lung'}(?!.*32b)(?!.*AAA)"
        elif 'regression' in name_dir:
            regex = rf"{EXP_NAME + '_regression'}(?!.*32b)(?!.*AAA)"
    path_exp = os.path.join(path_to_search, exp)
    for fold in folds:
        if fold == "loCo" and exp == "Multi":
            fold = "loCo6"
            num_of_weights = 6
        elif fold == "loCo" and (name_dir == "brixia_Lung" or name_dir == "brixia_Global" or name_dir == "regression"):
            num_of_weights = 3
        elif fold == "loCo" and name_dir == "AFC":
            num_of_weights = 6
        else:
            num_of_weights = 5
        path_fold = os.path.join(path_exp, fold)

        for d in os.listdir(path_fold):
                match = re.search(regex, d)


                """if 'Multi_regression' in name_dir:
                    print('EXP:', exp, 'REGEX:', regex, 'MATCH:',match, 'path: ', os.path.join(path_fold, d))"""
                if match:
                    regex_d = rf'({""}).*?({re.escape("Entire")})(?!.*Softmax)'

                    exp_fold_dir = os.path.join(path_fold, d)
                    for run in os.listdir(exp_fold_dir):
                        if re.search(regex_d, os.path.join(exp_fold_dir, run)):

                            for model in models:
                                path_model = os.path.join(exp_fold_dir, run, model)
                                if not os.path.exists(path_model):
                                    continue
                                print('\n---------------------------\n EXP:', name_dir,
                                          '\n EXIST',os.path.exists(path_model),
                                          '\n path: ', path_model,
                                          '\n Fold:', fold, '\n Trained?: ', len(os.listdir(path_model)) == num_of_weights,)

                                if os.path.exists(path_model) and len(os.listdir(path_model)) == num_of_weights:

                                    df.loc[model, (name_dir, fold)] = "Trained"

                                    for weight_dir in os.listdir(path_model):
                                        if len(os.listdir(os.path.join(path_model,weight_dir))) == 0:
                                            df.loc[model, (name_dir, fold)] = "Not Trained"



                                    #print(f"Trained: {model} - {name_dir} - {fold}")
                                else:
                                    df.loc[model, (name_dir, fold)] = "Not Trained"

                    continue
#df.drop(columns=[('BX_regression', '5'), ('BX_regression', 'loCo'), ('BX_brixia_Lung', '5'), ('BX_brixia_Lung', 'loCo'), ('BX_brixia_Global', '5'), ('BX_brixia_Global', 'loCo')],
# inplace=True)

print(df)
df.to_excel(f"models_status_EXP_{EXP_NAME}.xlsx")