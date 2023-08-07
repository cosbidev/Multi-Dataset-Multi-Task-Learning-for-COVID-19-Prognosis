import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot_regression(history, plot_training_dir):
    # Training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE error on Global Score')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(plot_training_dir, "Loss"))
    plt.close()
    # Training results
    metrics_train_x_epochs = history['train_metrics']
    metrics_val_x_epochs = history['val_metrics']

    # CC coeff.
    epochs = np.arange(len(metrics_train_x_epochs))

    columns = metrics_train_x_epochs[0].columns.to_list()
    indexes = metrics_train_x_epochs[0].index.to_list()
    # Each area score


    # TRAINING SCORES:
    # Plotting the scores over epochs
    list_of_scores = {phase: {index: [] for index in indexes} for phase in ['train', 'val']}

    for phase_metrics, phase in zip([metrics_train_x_epochs, metrics_val_x_epochs], ['train', 'val']):
        for index in indexes:
            score_by_area = {column: [] for column in columns}
            for column in columns:
                score_by_area[column].append(list(df.loc[index, column] for df in phase_metrics))
            list_of_scores[phase][index].append(score_by_area)


    # Plotting the scores over epochs
    dictionary_of_name = {'CC': 'Pearson Correlation Coefficient (CC)', 'MAE': 'Mean Absolute Error (MAE)',
                          'MSE': 'Mean Square Error (MSE)', 'R2': 'R2 Score', 'L1': 'L1 Loss',
                          'SD': 'Standard Deviation (SD)'}

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for index in indexes:
        fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True)  # Adjust the figure size as needed
        counter = 0
        for column, color, ax in zip(columns[:6], colors, fig.get_axes()):
            counter += 1
            for phase in ['train', 'val']:
                score = np.array(list_of_scores[phase][index][0][column][0])
                ax.plot(epochs, score, label=column + '_' + phase, linestyle='--' if phase == 'val' else '-',
                        marker='*' if phase == 'train' else 'o', color=color)

            ax.grid(True)
            ax.legend()

            ax.set_xlabel('' if counter <= 3 else 'Epochs')
            ax.set_title(f'{column}')
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

        plt.suptitle(f'{dictionary_of_name[index]} by Areas Scores, Analysis over Epochs', color = 'r')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"{index}_by_area"))
        plt.close()
    colors = ['b',  'r']
    for index in indexes:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Adjust the figure size as needed
        counter = 0
        for column, color, ax in zip(columns[6:8], colors, fig.get_axes()):
            counter += 1
            for phase in ['train', 'val']:
                score = np.array(list_of_scores[phase][index][0][column][0])
                ax.plot(epochs, score, label=column + '_' + phase, linestyle='--' if phase == 'val' else '-',
                        marker='*' if phase == 'train' else 'o', color=color)

            ax.grid(True)
            ax.legend()

            ax.set_xlabel('' if counter <= 3 else 'Epochs')
            ax.set_title(f'{column}')
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

        plt.suptitle(f'{dictionary_of_name[index]} by Regions Scores, Analysis over Epochs', color = 'r'    )
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"{index}_by_regions"))
        plt.close()
    colors = ['k']
    for index in indexes:
        fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharex=True)  # Adjust the figure size as needed
        counter = 0
        for column, color, ax in zip([columns[-1]], colors, fig.get_axes()):
            counter += 1
            for phase in ['train', 'val']:
                score = np.array(list_of_scores[phase][index][0][column][0])
                ax.plot(epochs, score, label=column + '_' + phase, linestyle='--' if phase == 'val' else '-',
                        marker='*' if phase == 'train' else 'o', color=color)

            ax.grid(True)
            ax.legend()

            ax.set_xlabel('' if counter <= 3 else 'Epochs')
            ax.set_title(f'{column}')
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

        plt.suptitle(f'{dictionary_of_name[index]} on Global Scores, Analysis over Epochs', color = 'r'    )
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"{index}_global"))
        plt.close()
def plot_training(history, plot_training_dir):
    # Training results Loss function
    if 'train_loss' in history.columns and 'val_loss' in history.columns:
        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'val_loss']:
            plt.plot(history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Negative Log Likelihood')
        plt.title('Training and Validation Losses')
        plt.savefig(os.path.join(plot_training_dir, "Loss"))
        plt.show()
    # Training results Accuracy
    if 'train_acc' in history.columns and 'val_acc' in history.columns:
        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'val_acc']:
            plt.plot(100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.savefig(os.path.join(plot_training_dir, "Acc"))
        plt.show()




def plot_morphos_contours(image, image_th, contour, image_gauss):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].imshow(image, cmap='gray')
    ax[0].imshow(image_gauss > image_th, alpha=0.5, cmap='Reds')
    ax[1].imshow(image, cmap='gray')
    ax[1].plot(contour[0][:, 1], contour[0][:, 0], '-r', linewidth=5)
    ax[1].plot(contour[1][:, 1], contour[1][:, 0], '-r', linewidth=5)


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
