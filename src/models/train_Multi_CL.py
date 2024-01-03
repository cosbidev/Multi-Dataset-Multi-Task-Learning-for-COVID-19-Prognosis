import argparse
import gc
import glob
import itertools
import shutil
import sys
import os

import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))
print(os.getcwd(), ' the current working directory')
sys.path.extend('./')

from easydict import EasyDict
from src.utils.utils_visualization import plot_training_multi, plot_training_multi_cl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import yaml
from src import (mkdir, seed_all, MultiTaskDataset, seed_worker, get_MultiTaskModel, evaluate_morbidity, is_debug, train_MultiTask, IdentityMultiHeadLoss, Logger, compute_report_metrics,
                 evaluate_regression, CustomSampler, print_CUDA_info)
from src.utils import utils_data


# Configuration file
def main():
    # Configuration file

    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("--cfg_file", help="Number of folder", type=str)
    parser.add_argument("--model_name", help="model_name")
    parser.add_argument("--id_exp", help="seed", default=1)
    parser.add_argument("--checkpointer", "-c", help="seed", action='store_true')
    args = parser.parse_args()

    # Load configuration file
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    with open('configs/common/common_multi_curriculum.yaml') as file_common:
        cfg_common = yaml.load(file_common, Loader=yaml.FullLoader)
    cfg.update(cfg_common)
    cfg['exp_name'] = cfg['exp_name'] + f'_Curriculum'
    del cfg_common
    # Seed everything
    seed_all(cfg['seed'])
    cfg = EasyDict(cfg)
    # Parameters
    batch_size = cfg['trainer']['batch_size']
    # Datasets configs
    morbidity_cfg = cfg.data.modes.morbidity
    severity_cfg = cfg.data.modes.severity

    # BACKBONE
    model_name = args.model_name

    steps = ['train', 'val', 'test']
    cv = cfg['data']['cv']

    # Data config
    data_cfg = cfg['data']
    CV = '_' + str(cfg['data']['cv'])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        Batch = f'_Batch{batch_size // torch.cuda.device_count()}'
    else:
        Batch = f'_Batch{batch_size}'

    # Preprocessing config
    CLAHE = '_Clahe' if data_cfg['preprocess']['clahe'] else ''
    Filter = '_Filter3th' if data_cfg['preprocess']['filter'] else ''
    Clip = '_Clip2-98' if data_cfg['preprocess']['clip'] else ''
    Masked = '_LungMask' if data_cfg['preprocess']['masked'] else '_Entire'

    # Model Config
    model_cfg = cfg['model']
    Drop = f'_Drop{model_cfg["dropout_rate"]}'
    freezing = '_unfreeze' if not model_cfg['freezing'] else ''
    regression = f'_regression_{model_cfg["regression_type"]}' if model_cfg['regression_type'] else ''
    pretrained = '_pretrained_BX' if model_cfg['pretrained'] else ''

    # Trainer Config:
    trainer_cfg = cfg['trainer']
    warming = f'_warmup_' if trainer_cfg['warmup_epochs'] != 0 else ''
    LearningRate = f'_LR{trainer_cfg["optimizer"]["lr"]}'

    # Experiment name
    exp_name = cfg['exp_name'] + CV + Batch + regression + pretrained + LearningRate + warming + Drop + CLAHE + Filter + Clip + Masked + freezing
    print(' ----------| Experiment name: ', exp_name)

    # Device
    device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
    print(device)
    # Directories
    cfg['exp_name'] = cfg['exp_name'] + f'_{args.id_exp}'
    cfg['data']['model_dir'] = os.path.join(cfg['data']['model_dir'], cfg['exp_name'])  # folder to save trained model
    cfg['data']['report_dir'] = os.path.join(cfg['data']['report_dir'], cfg['exp_name'])

    model_dir = os.path.join(cfg['data']['model_dir'] if (cfg['root'] == '.' or is_debug()) else os.path.join(cfg['root'],cfg['data']['model_dir']),
                             exp_name,
                             model_name)  # folder to save model
    print(' ----------| Model directory: ', model_dir)
    if os.path.exists(model_dir) and not args.checkpointer:
        shutil.rmtree(model_dir)
    mkdir(model_dir)


    report_dir = os.path.join(cfg['data']['report_dir'],
                             exp_name,
                             model_name)

    print(' ----------| Report directory: ', report_dir)
    if os.path.exists(report_dir) and not args.checkpointer:
        shutil.rmtree(report_dir)
    mkdir(report_dir)

    logger = Logger(file_name=os.path.join(report_dir, f'log_print_out_{model_name}.txt'), file_mode="w", should_flush=True)
    with open(os.path.join(report_dir, f'config_{model_name}.yaml'), 'w') as file:
        documents = yaml.dump(cfg, file)
    plot_training_dir = os.path.join(report_dir, "training_plot")

    if not args.checkpointer and os.path.exists(plot_training_dir):
        shutil.rmtree(plot_training_dir)
    mkdir(plot_training_dir)



    # Report Curriculum dir
    plot_curriculum_dir = os.path.join(report_dir, "curriculum_results")

    if os.path.exists(plot_curriculum_dir):
        shutil.rmtree(plot_curriculum_dir)
    mkdir(plot_curriculum_dir)



    # Create Fold Array for MORBIDITY TASK and SEVERITY TASK (same folds)
    cv_option = cfg['data']['cv']
    fold_grid, fold_list = utils_data.create_combined_folds(cv_option=cv_option, morbidity_cfg=morbidity_cfg, severity_cfg=severity_cfg)
    mapper_folder = {fold_value: {'M': fold_value.split('_')[0][1] if 'loco' in str(cv_option).lower() else fold_value,
                                  'S': fold_value.split('_')[1][1] if 'loco' in str(cv_option).lower() else fold_value} for fold_value in fold_list}
    if is_debug():
        fold_grid = {fold: fold_grid[fold] for fold in fold_list[:2]}

    # REPORT FINAL:
    # 1) TEST MORBIDITY
    final_results_test_M = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])
    # 2) TEST SEVERITY
    final_results_test_S = pd.DataFrame(columns=['Accuracy L1', 'Accuracy Exp', 'Accuracy Squared', 'Acc_G',
                                                 'Acc_LR', 'Acc_LL', 'LL_L1', 'RL_L1', 'G_L1', 'LL_CC', 'RL_CC', 'G_CC'])
    # 1) TEST MORBIDITY
    final_results_test_M_best = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])
    # 2) TEST SEVERITY
    final_results_test_S_best = pd.DataFrame(columns=['Accuracy L1', 'Accuracy Exp', 'Accuracy Squared', 'Acc_G',
                                                 'Acc_LR', 'Acc_LL', 'LL_L1', 'RL_L1', 'G_L1', 'LL_CC', 'RL_CC', 'G_CC'])
    # Dataframes Reports:
    history = None
    results_classes_test_all_iteration = None
    common_metrics_test_iteration_all = None
    for fold, fold_data in fold_grid.items():

        string_fold = '-----------| Fold ' + str(fold) + ' |----------'
        print(''.center(len(string_fold), '-'))
        print(''.center(len(string_fold), '-'))
        print(string_fold)
        print(''.center(len(string_fold), '-'))
        print(''.center(len(string_fold), '-'))

        # Model fold dir
        model_fold_dir = os.path.join(model_dir, str(fold))
        if not args.checkpointer and os.path.exists(model_fold_dir):
            shutil.rmtree(model_fold_dir)
        mkdir(model_fold_dir)

        plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
        mkdir(plot_training_fold_dir)
        test_results_by_patient = os.path.join(report_dir, 'test_prediction', str(fold))
        if os.path.exists(test_results_by_patient):
            shutil.rmtree(test_results_by_patient)
        mkdir(test_results_by_patient)

        # ------------------- MODEL -------------------
        model = get_MultiTaskModel(kind=cfg['kind'], backbone=model_name, cfg=cfg, device=device)

        if cfg['model']['pretrained']:
            ids_experiment_to_load = cfg['model']['pretrained']

            task = {'AFC': 'morbidity', 'BX': 'severity'}

            selector_exp_folder = {5: lambda name_dataset, id_: f'./models/{name_dataset}/5/{task[name_dataset]}_singletask_{id_}/',
                                   'loCo': lambda name_dataset, id_: f'./models/{name_dataset}/loCo/{task[name_dataset]}_singletask_{id_}/', }

            id_AFC = ids_experiment_to_load[0]
            id_BX = ids_experiment_to_load[1]
            cv_option_loading = 'loCo' if 'loco' in str(cv_option).lower() else cv_option
            exp_folder_morbidity = selector_exp_folder[cv_option_loading](name_dataset='AFC', id_=id_AFC)
            exp_folder_severtity = selector_exp_folder[cv_option_loading](name_dataset='BX', id_=id_BX)

            exp_single_task_selector = lambda folder_singletask: glob.glob(os.path.join(folder_singletask, '*LungMask_LungBbox')) if cfg['data']['preprocess']['masked'] \
                else glob.glob(os.path.join(folder_singletask, '*Entire_LungBbox'))
            print(exp_folder_severtity)
            print(exp_folder_morbidity)

            exp_folder_AFC = exp_single_task_selector(exp_folder_morbidity)
            exp_folder_BX = exp_single_task_selector(exp_folder_severtity)

            print(exp_folder_BX)
            assert len(exp_folder_AFC) > 0, f'Experiment folder for morbidity task not found: {exp_folder_AFC}'
            assert len(exp_folder_BX) > 0, f'Experiment folder for severity task not found: {exp_folder_BX}'

            exp_folder_AFC = exp_folder_AFC[0]
            exp_folder_BX = exp_folder_BX[0]

            if 'loco' in str(cv_option).lower():
                fold_id_s = mapper_folder[fold]['S']
                model_dir_weights_S = os.path.join(exp_folder_BX, model_name, fold_id_s)
                fold_id_m = mapper_folder[fold]['M']
                model_dir_weights_M = os.path.join(exp_folder_AFC, model_name, fold_id_m)
            else:
                fold_id_s = mapper_folder[fold]['S']
                model_dir_weights_S = os.path.join(exp_folder_BX, model_name, str(fold_id_s))
                fold_id_m = mapper_folder[fold]['M']
                model_dir_weights_M = os.path.join(exp_folder_AFC, model_name, str(fold_id_m))

            files_S = list(os.scandir(model_dir_weights_S))
            if len(files_S) > 1:
                times = [os.path.getatime(path) for path in files_S]
                file_S = files_S[times.index(min(times))]
            else:
                file_S = files_S[0]
            dict_Severity = torch.load(file_S.path, map_location=device)

            # Morbidity
            files_M = list(os.scandir(model_dir_weights_M))
            if len(files_M) > 1:
                times = [os.path.getatime(path) for path in files_M]
                file_M = files_M[times.index(min(times))]

            else:
                file_M = files_M[0]
            dict_Morbidity = torch.load(file_M.path, map_location=device)

            # Load weights for the parameters as the mean of the two models pretrained
            Beta = cfg['model']['beta']
            model.load_backbone_average_weights(dict_Morbidity, dict_Severity, Beta=Beta)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, [0, 1])

        # Checkpointer
        model_trained = False
        if args.checkpointer:
            list_saved_model = glob.glob(os.path.join(model_fold_dir, "*.pt"))
            if list_saved_model.__len__() > 0:
                print("Loading model from checkpoint for the fold ", fold)  # load the last checkpoint with the best model
                model_loaded = torch.load(list_saved_model[-1], map_location=device)
                if isinstance(model_loaded, torch.nn.DataParallel):
                    model_dict = model_loaded.module.state_dict()
                elif isinstance(model_loaded, torch.nn.Module):
                    model_dict = model_loaded.state_dict()
                elif isinstance(model_loaded, dict):
                    model_dict = model_loaded
                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(model_dict)
                elif isinstance(model, torch.nn.Module):
                    model.load_state_dict(model_dict)
                model_trained = True
        if args.checkpointer and not model_trained:
            print("No model found for the fold ", fold)
            print("Training from scratch for the fold ", fold)
        model = model.to(device)

        # CURRICULUM LEARNING
        curriculum_cfg = cfg['curriculum']
        dictionary_step = [{'AFC': step, 'BX': 100 - step} for step in curriculum_cfg['steps']]



        quantile_sampling_folder =os.path.join(exp_folder_AFC.replace('models', 'reports'), model_name,'quantile_sampling', str(fold))
        quantile_sampling_file = os.path.join(quantile_sampling_folder, 'q_train_images.csv')




        print(''.center(len(' Curriculum Learning '), '-'))
        # ------------------- DATA -------------------
        if is_debug():
            fold_data['train'] = fold_data['train'][::6]
            fold_data['val'] = fold_data['val'][::6]
            fold_data['test'] = fold_data['test'][::6]
        if True:
            datasets = {step: MultiTaskDataset(data=fold_data[step], cfg_morbidity=morbidity_cfg, cfg_severity=severity_cfg, step=step, cfg=cfg) for step in steps}
            for step in steps:
                print(f'{step} dataset size: {len(datasets[step])}')
                datasets[step].set_normalize_strategy(cfg['trainer']['normalizer'])
        # inter running step performances:
        loss = 1000


        # CREATE STEPS USING QUANTILES:

        if os.path.isfile(quantile_sampling_file):
            quantile_sampling = pd.read_csv(quantile_sampling_file)
        else:
            raise FileNotFoundError(f'Quantile sampling file not found: {quantile_sampling_file}')

        # Possible quantiles values:



        q_values = quantile_sampling['q'].sort_values().unique()


        # Create dictionary with quantile values:
        ids_q_dict  = quantile_sampling.set_index('pz_name')['q'].to_dict()



        # Create dictionary that for each q value correspond a list of IDS:
        selection_by_qs = {q: quantile_sampling[quantile_sampling['q'] == q]['pz_name'].to_list() for q in q_values}

        # Aggregate q values in a new dictionary such that the first iteration is only q = 0, in the second 0 and 1, and in the third 0, 1 and 2ù
        q_values = q_values if cfg['curriculum']['forward'] else q_values[::-1]

        for q in q_values:
            solo_value = 0 if cfg['curriculum']['forward'] else 2
            if q == solo_value:
                continue
            else:
                selection_by_qs[q] = selection_by_qs[q] + selection_by_qs[q-1] if cfg['curriculum']['forward'] else selection_by_qs[q] + selection_by_qs[q+1]


        # Reorder dictionary:
        selection_by_qs = {2-q: selection_by_qs[q] for q in q_values} if not cfg['curriculum']['forward'] else selection_by_qs





        for q, list_ids in selection_by_qs.items():



            # Lambda selection steps:
            lambda_q_steps = lambda q, ids: torch.utils.data.Subset(datasets['train'], datasets['train'].data[datasets['train'].data['dataset_class'] == 'AFC'][(datasets['train'].data[
                                                                                                                                                                     datasets['train'].data['dataset_class'] == 'AFC'].isin(ids))['img']].index.to_list())
            # Create new names for the steps:
            new_names_q_steps = {0: 'easy', 1: 'easy + medium', 2: 'easy + medium + hard'} if cfg['curriculum']['forward'] else {0: 'hard', 1: 'medium + hard', 2: 'easy + medium + hard'}



            print(''.center(len(' Curriculum Learning '), '-'))
            # ------------------- Separate Loaders M and S-------------------





            subset_selector = lambda step, class_d: torch.utils.data.Subset(datasets[step], datasets[step].data[datasets[step].data['dataset_class'] == class_d].index.to_list())

            data_loaders_AFC = {
                'val': DataLoader(subset_selector('val', 'AFC'), batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker),
                'test': DataLoader(subset_selector('test', 'AFC'), batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)}
            data_loaders_BX = {
                'val': DataLoader(subset_selector('val', 'BX'), batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker),
                'test': DataLoader(subset_selector('test', 'BX'), batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)}

            # DEFINE RATIO FOR DATASETS:
            ratio = 40 / 100  # PERC from dataset1, 80% from dataset2


            data_loader_train_curriculum = DataLoader(torch.utils.data.ConcatDataset([lambda_q_steps(q=q, ids=list_ids), subset_selector('train', 'BX')]),
                                                      num_workers=num_workers,
                                                      worker_init_fn=seed_worker,
                                                      batch_size=batch_size,

                                                      )
            # Process your batch here

            data_loaders = {'train': data_loader_train_curriculum,
                            'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker),
                            'test': torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)}
            idx_to_class = datasets['train'].m_idx_to_class
            # Multi Head Identity Loss Handling
            cfg['curriculum']['step_running'] = new_names_q_steps[q]
            criterion = IdentityMultiHeadLoss(cfg=cfg).to(device)
            if not model_trained:
                # Optimizer
                optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=cfg['trainer']['optimizer']['weight_decay'])
                # LR Scheduler
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode=cfg['trainer']['scheduler']['mode'],
                                                           patience=cfg['trainer']['scheduler']['patience'],
                                                           min_lr=float(cfg['trainer']['scheduler']['min_lr']),
                                                           factor=cfg['trainer']['scheduler']['factor'])
                # Train model using curriculum learning strategy
                print_CUDA_info()
                print(''.center(len(' Curriculum Learning '), '-'))
                print('|||     ITERATION:       Q - {0}    |||'.format(new_names_q_steps[q], len(dictionary_step)))
                model, history_partial, best_saved = train_MultiTask(model=model,
                                                                     model_file_name=f'{model_cfg.head}_{model_name}',
                                                                     dataloaders=data_loaders,
                                                                     cfg=cfg,
                                                                     criterion=criterion,
                                                                     optimizer=optimizer,
                                                                     scheduler=scheduler,
                                                                     num_epochs=curriculum_cfg['epochs'],
                                                                     max_epochs_stop=cfg['trainer']['early_stopping'],
                                                                     model_dir=model_fold_dir,
                                                                     device=device,
                                                                     save_model=False)
                # Save Training/Validation Loss History

                # TEST RESULTS FOR THE CURRENT ITERATION:

                results_test_by_image_iteration, results_classes_test_iteration, common_metrics_test_iteration = evaluate_morbidity(
                    model=model,
                    test_loader=data_loaders_AFC['test'],
                    criterion=criterion.get_loss(criterion.loss_1),
                    idx_to_class=idx_to_class,
                    device=device,
                    cfg=cfg,
                    topk=(1,))
                common_metrics_test_iteration = pd.DataFrame(common_metrics_test_iteration, index=[0])
                best_loss = best_saved['best_loss']
                best_acc = best_saved['best_acc']

                if loss > best_loss > 0.0:
                    best_weights_over_iterations = model.state_dict()
                    loss = best_loss
                    accu = best_acc
                    print('-' * 10)
                    print('Best Model Updated between all the iterations of Curriculum Learning')
                    print('Best Model Accuracy: ', accu)
                    print('Best Model Loss: ', loss)
                    print('-' * 10)

                del data_loaders, data_loader_train_curriculum
                gc.collect()
                torch.cuda.empty_cache()

                # Step iteration Name
                results_classes_test_iteration['step'] = np.repeat(f'q = {new_names_q_steps[q]}', len(results_classes_test_iteration))
                history_partial['step'] = np.repeat(f'q = {new_names_q_steps[q]}', len(history_partial))
                common_metrics_test_iteration['step'] = np.repeat(f'q = {new_names_q_steps[q]}', len(common_metrics_test_iteration))
                # Group step iteration
                results_classes_test_iteration['group'] = np.repeat(q, len(results_classes_test_iteration))
                history_partial['group'] = np.repeat(q, len(history_partial))
                common_metrics_test_iteration['group'] = np.repeat(q, len(common_metrics_test_iteration))
                # Fold Number
                results_classes_test_iteration['fold'] = np.repeat(fold, len(results_classes_test_iteration))
                history_partial['fold'] = np.repeat(fold, len(history_partial))
                common_metrics_test_iteration['fold'] = np.repeat(fold, len(common_metrics_test_iteration))

                if q == 0 and history is None and results_classes_test_all_iteration is None and common_metrics_test_iteration_all is None:
                    history = history_partial
                    results_classes_test_all_iteration = results_classes_test_iteration
                    common_metrics_test_iteration_all = common_metrics_test_iteration
                    
                else:

                    history = pd.concat([history, history_partial], axis=0).reset_index(drop=True)
                    results_classes_test_all_iteration = pd.concat([results_classes_test_all_iteration, results_classes_test_iteration], axis=0).reset_index(drop=True)
                    common_metrics_test_iteration_all = pd.concat([common_metrics_test_iteration_all, common_metrics_test_iteration], axis=0).reset_index(drop=True)
                # Save model after curriculum learning
        torch.save(best_weights_over_iterations, os.path.join(model_fold_dir, f'{model_name}' + f'_q_{q}' + '.pt'))
        best_final = model.state_dict()
        torch.save(best_final, os.path.join(model_fold_dir, f'{model_name}' + '_final' '.pt'))



        print('-----------------------------------'
              '\n Best Model-MultiTask Saved in: %s' % (os.path.join(model_dir, f'{model_name}')))

        # Plot Training
        plot_training_multi(history, plot_training_fold_dir)

        loss = IdentityMultiHeadLoss(cfg=cfg)



        # ------------------- TEST -------------------
        # MORBIDITY TEST EVALUATION
        # Evaluate the model on all the test data using the BEST weights between all the iterations
        model.load_state_dict(best_weights_over_iterations)

        results_test_by_image_best, results_classes_test_best, common_metrics_test_best = evaluate_morbidity(
            model=model,
            test_loader=data_loaders_AFC['test'],
            criterion=loss.get_loss(loss.loss_1),
            idx_to_class=idx_to_class,
            device=device,
            cfg=cfg,
            topk=(1,))

        # Save results by patient
        classes = cfg.data.modes.morbidity.classes
        final_results_test_M_best = (
            compute_report_metrics(final_report_folds=final_results_test_M_best,
                                   metrics_report=common_metrics_test_best,
                                   classes_report=results_classes_test_best,
                                   classes=classes,
                                   fold=fold,
                                   results_by_patient=results_test_by_image_best,
                                   model_name=model_name,
                                   report_path=test_results_by_patient,
                                   optional_Dataset='bestABS_AFC'))
        # SEVERITY TEST EVALUATION
        # Evaluate the model on all the test data
        metrics_test_best, loss_test, results_metrics_resume_test_best, results_for_images_test_best = \
            evaluate_regression(model=model,
                                test_loader=data_loaders_BX['test'],
                                criterion=loss.get_loss(loss.loss_2),
                                device=device,
                                regression_type=cfg['model']['regression_type'],
                                cfg=cfg)
        final_results_test_S_best = (
            compute_report_metrics(final_report_folds=final_results_test_S_best,
                                   metrics_report=results_metrics_resume_test_best,
                                   fold=fold,
                                   results_by_patient=results_for_images_test_best,
                                   model_name=model_name,
                                   report_path=test_results_by_patient,
                                   optional_Dataset='BX'))


        # SAVE FINAL RESULTS FOR ALL THE METRICS
        # TEST
        final_results_test_M_best.to_excel(os.path.join(report_dir, f'[all]_test_results_{model_name}_AFC_best.xlsx'))
        final_results_test_S_best.to_excel(os.path.join(report_dir, f'[all]_test_results_{model_name}_BX_best.xlsx'))











        # ------------------- TEST FINAL -------------------
        # MORBIDITY TEST EVALUATION
        # Evaluate the model on all the test data using the FINAL WEIGHTS
        # REPORT FINAL:

        model.load_state_dict(best_final)
        results_test_by_image, results_classes_test, common_metrics_test = evaluate_morbidity(
            model=model,
            test_loader=data_loaders_AFC['test'],
            criterion=loss.get_loss(loss.loss_1),
            idx_to_class=idx_to_class,
            device=device,
            cfg=cfg,
            topk=(1,))
        # Save results by patient
        classes = cfg.data.modes.morbidity.classes
        final_results_test_M = (
            compute_report_metrics(final_report_folds=final_results_test_M,
                                   metrics_report=common_metrics_test,
                                   classes_report=results_classes_test,
                                   classes=classes,
                                   fold=fold,
                                   results_by_patient=results_test_by_image,
                                   model_name=model_name,
                                   report_path=test_results_by_patient,
                                   optional_Dataset='FINAL_AFC'))
        # SEVERITY TEST EVALUATION
        # Evaluate the model on all the test data
        metrics_test, loss_test, results_metrics_resume_test, results_for_images_test = \
            evaluate_regression(model=model,
                                test_loader=data_loaders_BX['test'],
                                criterion=loss.get_loss(loss.loss_2),
                                device=device,
                                regression_type=cfg['model']['regression_type'],
                                cfg=cfg)
        final_results_test_S = (
            compute_report_metrics(final_report_folds=final_results_test_S,
                                   metrics_report=results_metrics_resume_test,
                                   fold=fold,
                                   results_by_patient=results_for_images_test,
                                   model_name=model_name,
                                   report_path=test_results_by_patient,
                                   optional_Dataset='BX'))

    # Save all data for all FOLDS
    history.to_csv(os.path.join(plot_curriculum_dir, f'history_{model_name}_all_fold.csv'))
    results_classes_test_all_iteration.to_csv(os.path.join(plot_curriculum_dir, f'results_classes_test_{model_name}_all_fold.csv'))
    common_metrics_test_iteration_all.to_csv(os.path.join(plot_curriculum_dir, f'common_metrics_test_{model_name}_all_fold.csv'))

    # Plot Training Curriculum
    plot_training_multi_cl(data_history=history, data=common_metrics_test_iteration_all, data_classes=results_classes_test_all_iteration, plot_curriculum_dir=plot_curriculum_dir)

    # SAVE FINAL RESULTS FOR ALL THE METRICS
    # TEST
    final_results_test_M.to_excel(os.path.join(report_dir, f'[all]_test_results_{model_name}_AFC.xlsx'))
    final_results_test_S.to_excel(os.path.join(report_dir, f'[all]_test_results_{model_name}_BX.xlsx'))


if __name__ == '__main__':
    main()
