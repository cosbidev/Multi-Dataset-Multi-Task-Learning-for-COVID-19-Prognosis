import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import collections
import yaml




from src import mkdir, seed_all


with open(args.cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
classes = cfg['data']['classes']
model_name = cfg['model']['model_name']
steps = ['train', 'val', 'test']
cv = cfg['data']['cv']
fold_list = list(range(cv))

# Device
device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)

# Files and Directories
model_dir = os.path.join(cfg['data']['model_dir'], exp_name) # folder to save model
mkdir(model_dir)
report_dir = os.path.join(cfg['data']['report_dir'], exp_name) # folder to save results
mkdir(report_dir)
report_file = os.path.join(report_dir, 'report.xlsx')
plot_training_dir = os.path.join(report_dir, "training_plot")
mkdir(plot_training_dir)
plot_test_dir = os.path.join(report_dir, "test_plot")
mkdir(plot_test_dir)

# Train CV
results = collections.defaultdict(lambda: [])
metric_cols = []
metric_class_cols = collections.defaultdict(lambda: [])
for fold in fold_list:
    # Dir
    model_fold_dir = os.path.join(model_dir, str(fold))
    mkdir(model_fold_dir)
    plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
    mkdir(plot_training_fold_dir)
    plot_test_fold_dir = os.path.join(plot_test_dir, str(fold))
    mkdir(plot_test_fold_dir)

    # Results Frame
    metric_cols.append(str(fold))
    for c in classes[::-1]:
        metric_class_cols[c].append("%s %s" % (str(fold), c))

    # Data Loaders
    fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], str(fold), '%s.txt' % step), delimiter=" ", index_col=0) for step in steps}
    datasets = {step: util_data.DatasetImg(data=fold_data[step], classes=classes, cfg_data=cfg['data']['modes']['img'], step=step) for step in steps}
    data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                    'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                    'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)}

    # Model
    input, _, _ = iter(data_loaders["train"]).next()
    input_dim = input[0].shape[1]
    model = util_model.get_img_autoencoder(model_name=model_name, input_dim=input_dim, h_dim=cfg['model']['h_dim'])
    if cfg['model']['pretrained']:
        model.load_state_dict(torch.load(os.path.join(cfg['model']['pretrained'], str(fold), "model.pt"), map_location=device))
    model = model.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=cfg['trainer']['optimizer']['weight_decay'])
    # LR Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['trainer']['scheduler']['mode'], patience=cfg['trainer']['scheduler']['patience'])
    # Train model
    model, history = util_model.train_autoencoder(model=model, data_loaders=data_loaders,
                                                  criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                                  num_epochs=cfg['trainer']['max_epochs'], early_stopping=cfg['trainer']['early_stopping'],
                                                  model_dir=model_fold_dir, device=device)

    # Plot Training
    util_model.plot_training(history=history, plot_training_dir=plot_training_fold_dir)

    # Test model
    test_results = util_model.evaluate_autoencoder(model=model, data_loader=data_loaders['test'], criterion=criterion, device=device)
    print(test_results)

    # Plot Test
    util_model.plot_evaluate_img_autoencoder(model=model, data_loader=data_loaders['test'], plot_dir=plot_test_fold_dir, device=device)

    # Update report
    results[str(fold)].append(test_results['all'])
    for c in classes:
        results["%s %s" % (str(fold), str(c))].append(test_results[c])

    # Save Results
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    for c in classes[::-1]:
        results_frame.insert(loc=0, column='std %s' % c, value=results_frame[metric_class_cols[c]].std(axis=1))
        results_frame.insert(loc=0, column='mean %s' % c, value=results_frame[metric_class_cols[c]].mean(axis=1))
    results_frame.insert(loc=0, column='std', value=results_frame[metric_cols].std(axis=1))
    results_frame.insert(loc=0, column='mean', value=results_frame[metric_cols].mean(axis=1))
    results_frame.insert(loc=0, column='model', value=model_name)
    results_frame.to_excel(report_file, index=False)
