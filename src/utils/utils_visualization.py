import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.ticker as ticker

def plot_test_results(test_results, plot_test_dir):
    # Test results Loss function
    if 'test_loss' in test_results.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(test_results['test_loss'], label='test_loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Negative Log Likelihood')
        plt.title('Test Loss')
        plt.savefig(os.path.join(plot_test_dir, "Loss"))
        plt.show()
    # Test results Accuracy
    if 'test_acc' in test_results.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(100 * test_results['test_acc'], label='test_acc')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.title('Test Accuracy')
        plt.savefig(os.path.join(plot_test_dir, "Acc"))
        plt.show()

def plot_bbox_on_image(img, box_tot):

    # Create a figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)
    rect = patches.Rectangle((box_tot[0], box_tot[1]), box_tot[2], box_tot[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()
def plot_training_multi(history, plot_training_dir):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'b']
    for i, c in enumerate(['train_loss', 'val_loss']):
        plt.plot(history[c], label=c, color=colors[i])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE (M + S)')
    plt.title('Training and Validation Losses: Morbidity + Severity')
    plt.savefig(os.path.join(plot_training_dir, "Loss"))
    plt.close()
    plt.figure(figsize=(8, 6))
    for i, c in enumerate(['train_acc', 'val_acc']):
        plt.plot(history[c], label=c, color=colors[i])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy : Morbidity')
    plt.savefig(os.path.join(plot_training_dir, "ACC"))
    plt.close()



def plot_training_multi_cl(data_history,  data,  data_classes, plot_curriculum_dir):
    # Let's create a combined bar plot where the x-axis will represent each group and
    # the y-axis will represent the accuracy for each group with separate bars for overall, 'MILD', and 'SEVERE' accuracies.

    # First, we need to prepare the data. We will calculate the mean accuracy for each group for overall, 'MILD', and 'SEVERE'.
    # This will align the data so that each group has an entry for each type of accuracy.

    # Calculate mean accuracies for each group
    overall_accuracy_mean = data.groupby('group')['Accuracy'].mean()
    overall_accuracy_std = data.groupby('group')['Accuracy'].std()
    # MILD CLASS
    mild_accuracy_mean = data_classes[data_classes['class'] == 'MILD'].groupby('group')['top1'].mean()
    mild_accuracy_std = data_classes[data_classes['class'] == 'MILD'].groupby('group')['top1'].std()

    # SEVERE CLASS
    severe_accuracy_mean = data_classes[data_classes['class'] == 'SEVERE'].groupby('group')['top1'].mean()
    severe_accuracy_std = data_classes[data_classes['class'] == 'SEVERE'].groupby('group')['top1'].std()


    dict_encoded = data.loc[:, ['step', 'group']].to_dict()['step']
    # Ensure that we have a matching index for all dataframes before plotting
    overall_accuracy_mean = overall_accuracy_mean.reindex(mild_accuracy_mean.index).fillna(0)
    overall_accuracy_std = overall_accuracy_std.reindex(mild_accuracy_mean.index).fillna(0)
    severe_accuracy_mean = severe_accuracy_mean.reindex(mild_accuracy_mean.index).fillna(0)
    severe_accuracy_std = severe_accuracy_std.reindex(mild_accuracy_mean.index).fillna(0)

    # Define the position of the bars for each group
    fig, ax = plt.subplots(figsize=(15, 10))

    # The positions of the groups on the x-axis
    group_positions = np.arange(len(overall_accuracy_mean))

    # Interpolate the points with a line for each accuracy type

    ax.errorbar(group_positions, overall_accuracy_mean, yerr=overall_accuracy_std, color='blue', marker='^', markersize=15, label='Overall Accuracy', linestyle='-', zorder=3)

    ax.errorbar(group_positions, mild_accuracy_mean, yerr=mild_accuracy_std, color='orange', marker='s', markersize=10, label='MILD Accuracy', linestyle='-.', zorder=2)

    ax.errorbar(group_positions, severe_accuracy_mean, yerr=severe_accuracy_std, color='green', marker='o', markersize=10, label='SEVERE Accuracy', linestyle='-.', zorder=1)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Group', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_title('Accuracy for Overall, MILD, and SEVERE by Group with Interpolated Lines', fontsize=20)
    ax.set_xticks(group_positions)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    ax.set_xticklabels(overall_accuracy_mean.index.map(dict_encoded), rotation=20)
    ax.grid()
    ax.set_ylim(0, 100)  # Assuming accuracy is a percentage
    ax.legend()

    # Save Figure
    plt.savefig(os.path.join(plot_curriculum_dir, "ACC_over_ITERATIONS_TEST.png"))

    # Set up the figure for multiple subplots, assuming we have a reasonable number of groups
    groups = data_history['group'].unique()

    colors = ['salmon', 'skyblue', 'firebrick', 'darkblue']

    group_colors = ['limegreen', 'violet', 'purple', 'magenta', 'yellow', 'orange']
    dict_convert = {i: name for i, name in enumerate(data_classes['step'].unique())}
    # Set up the figure for one horizontal plot containing all groups

    # Plotting training and validation accuracies for each group on the same axis
    for fold in data_history['fold'].unique():

        data_history_fold = data_history[data_history['fold'] == fold].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 5))
        for i, group in enumerate(groups):
            group_data = data_history_fold[data_history_fold['group'] == group]

            # Overall Accuracies
            ax.plot(group_data.index + 1, group_data['train_loss_S'], label=f'Training Loss Severity', marker='o', color=colors[0])
            ax.plot(group_data.index + 1, group_data['val_loss_S'], label=f'Validation Loss Severity', marker='s', color=colors[1])

            ax.plot(group_data.index + 1, group_data['train_loss_M'], label=f'Training Loss Morbidity', marker='o', color=colors[2])
            ax.plot(group_data.index + 1, group_data['val_loss_M'], label=f'Validation Loss Morbidity', marker='s', color=colors[3])

            plt.axvline(x=max(group_data.index) + 1, color=group_colors[i], linestyle='-.', label=dict_convert[group], linewidth=4)
        # Set the title and labels
        ax.set_title('Training vs Validation Losses for all Iterations and AFC/BX losses-- FOLD: {}'.format(fold))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Losses')

        # Ensure the x-axis ticks represent epochs properly
        # Get all unique epochs across all groups to set x-ticks accurately
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 0.50))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        # Shrink current axis by 20% to fit the legend outside of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.grid()

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

        # Save Figure
        plt.savefig(os.path.join(plot_curriculum_dir, f"LOSSES_over_ITERATIONS_train_fold_{fold}.png"))


        group_colors = ['limegreen', 'violet', 'purple', 'magenta', 'yellow', 'orange']
        dict_convert = {i: name for i, name in enumerate(data_classes['step'].unique())}
        # Set up the figure for one horizontal plot containing all groups
        fig, ax = plt.subplots(figsize=(15, 5))
        # Plotting training and validation accuracies for each group on the same axis
        for i, group in enumerate(groups):
            group_data = data_history_fold[data_history_fold['group'] == group]

            # Overall Accuracies
            ax.plot(group_data.index + 1, group_data['train_acc'], label=f'Training Accuracy', marker='o',color=colors[0])
            ax.plot(group_data.index + 1, group_data['val_acc'], label=f'Validation Accuracy', marker='s',color=colors[1])
            plt.axvline(x=max(group_data.index) + 1, color=group_colors[i], linestyle='-.', label=dict_convert[group])
        # Set the title and labels
        ax.set_title('Training vs Validation Accuracy for All Groups -- FOLD: {}'.format(fold))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        # Ensure the x-axis ticks represent epochs properly
        # Get all unique epochs across all groups to set x-ticks accurately

        # Shrink current axis by 20% to fit the legend outside of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.grid()

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(os.path.join(plot_curriculum_dir, f"ACC_over_ITERATIONS_train_fold_{fold}.png"))



def plot_regression(history, plot_training_dir, name_of_accuracies = ['LL', 'RL']):
    # Training results Loss function
    colors = ['r', 'b']
    plt.figure(figsize=(8, 6))
    for i, c in enumerate(['train_loss', 'val_loss']):
        plt.plot(history[c], label=c, color=colors[i])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('BCE error on Global Score')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(plot_training_dir, "Loss"))
    plt.close()

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    list_axis = axs.tolist()
    name_of_accuracies = name_of_accuracies
    for name_acc, ax in zip(name_of_accuracies, list_axis):
        for i, c in enumerate(['train_acc_' + name_acc, 'val_acc_' + name_acc]):
            ax.plot(history[c], label=c, color=colors[i])

            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('ACC on {} score'.format(name_acc), fontsize=15)  # 'Accuracy'')
            ax.legend(fontsize=15)

    plt.suptitle('Train and Validation Accuracy : Severity', fontsize=30)
    plt.savefig(os.path.join(plot_training_dir, "ACC_zones_severity.png"))
    plt.close()



def plot_training(history, plot_training_dir, name=""):
    # Training results Loss function
    if 'train_loss' in history.columns and 'val_loss' in history.columns:
        plt.figure(figsize=(8, 6))
        colors = ['r', 'b']
        for i, c in enumerate(['train_loss', 'val_loss']):
            plt.plot([ep + 1 for ep in list(history.index)], history.loc[:, c], label=c, color=colors[i])
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss Function')
        plt.title('Training and Validation Losses')
        plt.savefig(os.path.join(plot_training_dir, f"Loss{name}"))
        plt.close()
    # Training results Accuracy
    if 'train_acc' in history.columns and 'val_acc' in history.columns:
        plt.figure(figsize=(8, 6))
        colors = ['r', 'b']
        for i,c in enumerate(['train_acc', 'val_acc']):
            plt.plot([ep + 1 for ep in list(history.index)], 100 * history.loc[:, c], label=c, color=colors[i])
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.savefig(os.path.join(plot_training_dir,  f"Acc{name}"))
        plt.close()

def plot_morphos_contours(image, image_th, contour, image_gauss):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].imshow(image, cmap='gray')
    ax[0].imshow(image_gauss > image_th, alpha=0.5, cmap='Reds')
    ax[1].imshow(image, cmap='gray')
    ax[1].plot(contour[0][:, 1], contour[0][:, 0], '-r', linewidth=5)
    ax[1].plot(contour[1][:, 1], contour[1][:, 0], '-r', linewidth=5)


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]  # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns')  # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])  # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10:  # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
