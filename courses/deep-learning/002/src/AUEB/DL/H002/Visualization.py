"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Homework: 2
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

__main_title_font_dict = {
    "family": "sans serif", 
    "color":  "black", 
    "weight": "bold", 
    "size": 18
}

__axis_title_fontdict = {
    "family": "sans serif", 
    "color":  "black", 
    "weight": "bold", 
    "size": 14
}

__axis_tick_fontdict = {
    "family": "sans serif", 
    "color":  "black", 
    "weight": "normal", 
    "size": 12
}


"""
"""
def plot_hist(
    data, 
    figsize:tuple = (12, 10), 
    title:str = "Histogram", 
    x_label:str = "class", 
    y_label:str = "frequency", 
    x_tick_labels:[] = None, 
    color:str = "#494949", 
    **kwargs
):
    fig, ax = plt.subplots(figsize = figsize)
    
    labels, counts = np.unique(data, return_counts = True)

    ax.bar(
        labels, 
        counts, 
        width = 0.5, 
        color = color, 
        align = "center"
    )
    
    ax.set_facecolor("#EEEEEE")

    ax.set_xticklabels(x_tick_labels, fontdict = __axis_tick_fontdict)
    ax.set_yticklabels(counts, fontdict = __axis_tick_fontdict)
    
    ax.set_xlabel(x_label, fontdict = __axis_title_fontdict)
    ax.set_ylabel(y_label, fontdict = __axis_title_fontdict)
    
    # Align x-ticks.
    ax.set(xticks = range(len(labels)), xlim = [-1, len(labels)])

    ax.set_title(
        label = title, 
        loc = "center", 
        fontdict = __main_title_font_dict, 
        pad = 20
    )

    return fig


"""
"""
def plot_images(
        data, 
        labels:[] = None, 
        nrows:int = 1, 
        figsize:tuple = (10, 10), 
        title:str = "Images"
):
    assert((labels is None)or (len(data) == len(labels)))

    if labels is None: 
        labels = ["Image (%d)" % i for i in range(1, len(data) + 1)]

    fig = plt.figure(figsize = figsize)

    for n, (image, label) in enumerate(zip(data, labels)):
        subplot = fig.add_subplot(nrows, np.ceil(len(data) / float(nrows)), n + 1)
        subplot.set_axis_off()

        plt.imshow(
            image.reshape(28, 28), 
            cmap = plt.get_cmap("gray")
        )
        
        # subplot.set_title(label)

        subplot.set_title(
            label = label, 
            loc = "center", 
            fontdict = __axis_title_fontdict, 
            pad = 3
        )

    fig.suptitle(
        t = title, 
        fontsize = __main_title_font_dict["size"], 
        fontweight = __main_title_font_dict["weight"]
    )

    return fig

"""
    Plots TensorFlow training history.

    :param histories: a dictionary of histories. Key is model name, value is model class.
    :param metrics: metrics to plot.
    :param epochs: epochs to plot x-axis.
    :param title: plot title.
    :param x_label: x-axis label.
    :param y_label: y-axis label.
    :param log_y: whether to log-transform the y-axis values.
    :param figsize: plot size. 
"""
def plot_training_history(
    histories:{} = None, 
    metrics = ["loss", "val_loss"], 
    epochs = 10, 
    title:str = "", 
    x_label:str = "", 
    y_label:str = "", 
    log_y: bool = False, 
    figsize = (16, 10)
):
    fig, ax = plt.subplots(figsize = figsize)
    
    ax.set_facecolor("#EEEEEE")

    for i, history_label in enumerate(histories):
        for metric in histories[history_label].history:
            if metric in metrics:
                if metric.startswith("val_"):
                    if log_y == True:
                        ax.plot(np.log(histories[history_label].history[metric]), color = f"C{i}", label = f"{history_label} {metric}")
                    else:
                        ax.plot(histories[history_label].history[metric], color = f"C{i}", label = f"{history_label} {metric}")
                else:
                    if log_y == True:
                        ax.plot(np.log(histories[history_label].history[metric]), "--", color = f"C{i}", label = f"{history_label} {metric}")
                    else:
                        ax.plot(histories[history_label].history[metric], "--", color = f"C{i}", label = f"{history_label} {metric}")
    
    ax.set_xlabel(x_label, fontdict = __axis_title_fontdict)
    ax.set_ylabel(y_label, fontdict = __axis_title_fontdict)

    x_ticks = np.arange(1, epochs + 1, epochs / 10)
    # x_ticks [0] += 1

    # y_ticks = np.arange(1, 2)

    ax.set_xticks(x_ticks)
    # ax.set_xticks(y_ticks)
    
    # ax.set_xlim([1, epochs + 1])
    # ax.set_ylim([0, 1])
    
    ax.set_title(
        label = title, 
        loc = "center", 
        fontdict = __main_title_font_dict, 
        pad = 20
    )
    
    plt.legend()
    
    return fig

def plot_auc(
    y_true, 
    y_pred, 
    title: str = None, 
    figsize = (10, 10)
):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    auc = auc(false_positive_rate, true_positive_rate)

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)

    # AUC plot
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].plot(
        false_positive_rate, 
        true_positive_rate, 
        label = f"Model AUC: {auc}"
    )
    axes[0].set_xlabel("False positive rate", fontdict = __axis_title_fontdict)
    axes[0].set_ylabel("True positive rate", fontdict = __axis_title_fontdict)

    # Zoomed AUC plot (top left corner).
    axes[1].xlim(0, 0.2)
    axes[1].ylim(0.8, 1)
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[0].plot(
        false_positive_rate, 
        true_positive_rate, 
        label = f"Model AUC: {auc}"
    )
    axes[0].set_xlabel("False positive rate", fontdict = __axis_title_fontdict)
    axes[0].set_ylabel("True positive rate", fontdict = __axis_title_fontdict)

    fig.set_title(
        label = title, 
        loc = "center", 
        fontdict = __main_title_font_dict, 
        pad = 20
    )

    return fig

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()
    # # Zoom in view of the upper left corner.
    # plt.figure(2)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve (zoomed in at top left)')
    # plt.legend(loc='best')
    # plt.show()

"""
    Plot a classifier confusion matrix.

    :param y_true: true target variables.
    :param y_pred: predicted target variables.
"""
def plot_confusion_matrix(
    y_true, 
    y_pred, 
    labels: [], 
    normalize: str = None, 
    title: str = None, 
    figsize = (10, 10)
):
    confussion_matrix = confusion_matrix(y_true = y_true, y_pred = y_pred, normalize = normalize)
    
    fig, ax = plt.subplots(figsize = figsize)

    sns.heatmap(confussion_matrix, annot = True, fmt = ".2%" if normalize == "all" else "d", ax = ax)

    ax.set_xticklabels(labels, fontdict = __axis_tick_fontdict)
    ax.set_yticklabels(labels, fontdict = __axis_tick_fontdict)
    
    ax.set_xlabel("Predicted", fontdict = __axis_title_fontdict)
    ax.set_ylabel("Actual", fontdict = __axis_title_fontdict)

    ax.set_title(
        label = title, 
        loc = "center", 
        fontdict = __main_title_font_dict, 
        pad = 20
    )

    return fig