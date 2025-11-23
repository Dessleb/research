import matplotlib.pyplot as plt
import time
import seaborn as sns
from tabulate import tabulate
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, f1_score, auc, recall_score, accuracy_score, precision_score, roc_auc_score, roc_curve
import numpy as np
from IPython.display import display



def stacked_bar_chart(models, metric):
    x = np.arange(len(models)) 
    width = 0.5  # Wider bars generally look better for stacked charts
    
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))

    # Initialize the "bottom" of the bars to zero for all models
    bottom = np.zeros(len(models))

    for attribute, measurement in metric.items():
        # Plot the bar starting from the 'bottom' array
        rects = ax.bar(x, measurement, width, label=attribute, bottom=bottom)
        
        # Optional: Add text labels in the center of each segment
        ax.bar_label(rects, label_type='center', fmt='%.0f', padding=0)
        
        # Update the 'bottom' so the next metric sits on top of this one
        bottom += measurement

    ax.set_ylabel('Total Score Sum')
    ax.set_title('Comparison of Models Based on Metrics')
    ax.set_xticks(x, models)
    
    # Move legend outside right so it doesn't cover the bars
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Remove fixed ylim or set it high enough to hold the sum of all metrics
    # ax.set_ylim(0, 500) 

    plt.show()






import matplotlib.colors as colors

def sbar_chart(models, metrics_data, legend_title="Metric", legend_labels=None):
  """Creates a stacked bar chart to compare models based on given metrics.

  Args:
      models: A list of model names.
      metrics_data: A dictionary where keys are metric names and values are lists of
                     metric values for each model.
      legend_title (str, optional): Title for the legend. Defaults to "Metric".
      legend_labels (list, optional): Custom labels for the legend entries. 
                       Defaults to None (uses metric names).
  """

  # Create the plot
  fig, ax = plt.subplots(layout="constrained")

  # Prepare data for plotting
  model_count = len(models)
  metric_count = len(metrics_data)
  width = 0.75  # Width of the bars

  # Create the stacked bars
  bottom = np.zeros(model_count)
  custom_colors = ["#1F77B4", "#2CA02C", "#9467BD"]
  color_map = colors.LinearSegmentedColormap.from_list("", custom_colors)  # Example colormap
  bar_colors = color_map(np.linspace(0, 1, metric_count))  # Assign colors to metrics

  for i, metric in enumerate(metrics_data):
    values = metrics_data[metric]
    p = ax.bar(models, values, width, label=metric, bottom=bottom, color=bar_colors[i])
    bottom += values

    ax.bar_label(p, label_type='center')

  # Customize legend
  if legend_labels:
    ax.legend(title=legend_title, labels=legend_labels)  # Use custom labels
  else:
    ax.legend(title=legend_title)  # Use metric names as labels


 # Adjust legend position
  # Position the legend outside the plot
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')






  # Add labels, title
  ax.set_xlabel('Model')
  ax.set_ylabel('Metric Value')
  ax.set_title('Comparison of Models Based on Metrics')

  plt.show()











def create_stacked_bar_chart(models, metrics_data):
  """Creates a stacked bar chart to compare models based on given metrics.

  Args:
    models: A list of model names.
    metrics_data: A dictionary where keys are metric names and values are lists of
                 metric values for each model.
  """

  # Create the plot
  fig, ax = plt.subplots(layout="constrained")

  # Prepare data for plotting
  model_count = len(models)
  metric_count = len(metrics_data)
  width = 0.75  # Width of the bars

  # Create the stacked bars
  bottom = np.zeros(model_count)
  for i, metric in enumerate(metrics_data):
    values = metrics_data[metric]
    p = ax.bar(models, values, width, label=metric, bottom=bottom)
    bottom += values

    ax.bar_label(p, label_type='center')


  # Position the legend outside the plot
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

  # Add labels, title, and legend
  ax.set_xlabel('Model')
  ax.set_ylabel('Metric Value')
  ax.set_title('Comparison of Models Based on Metrics')
  ax.legend()

  plt.show()



def measure_training_time(model, X_train, y_train):
    """
    Measures the training time of a machine learning model.

    Args:
        model: The machine learning model to train.
        X_train: The training features.
        y_train: The training labels.

    Returns:
        The training time in seconds.
    """

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time
    return round(training_time,2)








def auc_roc(true_prediciton, actual_prediction, model):
    auc_roc = round(roc_auc_score(true_prediciton, actual_prediction),2)*100
    return auc_roc


# Classification Report func
def report(true_pred, model_pred, model_name, output_dict=True):
    # rf_report = round(pd.DataFrame(classification_report(true_pred, model_pred, output_dict=output_dict)).transpose(),2)
    rf_report = classification_report(true_pred, model_pred)
    print(f'------{model_name} model report------------')
    print(rf_report)
    # print(tabulate(rf_report, headers='keys', tablefmt='fancy_grid'))

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
  """
  Plots a confusion matrix with percentage annotations and custom class labels.

  Args:
      y_true: True labels.
      y_pred: Predicted labels.
      model_name: Name of the model.
  """

  class_labels = ["Normal", "Diabetes"]
  cm = confusion_matrix(y_true, y_pred, normalize='true')

  fig, ax = plt.subplots(figsize=(8, 6))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
  disp.plot(cmap='Blues', ax=ax, values_format='.2%')

  plt.title(f'{model_name} Confusion Matrix')
  plt.show()




def plot_auc_roc_curve(model,model_name, X_test, y_true):
  """
  Plots the AUC-ROC curve for a given model.

  Args:
    model: The trained model.
    X_test: The test features.
    y_test: The true labels.
  """

  # Predict probabilities for the positive class
  y_pred_proba = model.predict_proba(X_test)[:, 1]

  # Calculate false positive rate (fpr), true positive rate (tpr), and thresholds
  fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

  # Calculate AUC
  roc_auc = auc(fpr, tpr)

  # Plot the ROC curve
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
  plt.legend(loc='lower right')
  plt.show()


def model_accuracy(y_test, *args):
    models = []
    for model in args:
        models.append(round(accuracy_score(y_test, model)*100))
    return models

def model_precision(y_test, *args):
    models = []
    for model in args:
        models.append(round(precision_score(y_test, model)*100))
    return models

def model_recall(y_test, *args):
    models = []
    for model in args:
        models.append(round(recall_score(y_test, model)*100))
    return models

def model_f1_score(y_test, *args):
    models = []
    for model in args:
        models.append(round(f1_score(y_test, model)*100))
    return models


def specificity(y_true, y_pred):
    """Calculates specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return round(tn / (tn + fp),2)



def print_metrics_table(model_names, metrics_dict):
    # 1. Create a DataFrame first (it's the easiest way to organize the data)
    df = pd.DataFrame(metrics_dict)
    
    # 2. Add the models as the first column (not index, for tabulate printing)
    df.insert(0, "Model Name", model_names)
    
    # 3. Format the numeric columns to strings with '%'
    # We apply this to all columns except 'Model Name'
    for col in df.columns:
        if col != "Model Name":
            df[col] = df[col].apply(lambda x: f"{x:.2f}%")
            
    # 4. Print using tabulate
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

def plot_multimodel_roc(y_test, probas_dict):
 
    plt.figure(figsize=(10, 8))
    
    # Define some line styles/colors to distinguish models
    colors = ['blue', 'green', 'red']
    linestyles = ['-', '--', '-.']
    
    for i, (model_name, probabilities) in enumerate(probas_dict.items()):
        # Calculate FPR and TPR
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        
        # Calculate Area Under Curve (AUC)
        roc_auc = auc(fpr, tpr)
        
        # Plot the curve
        # We cycle through colors/styles to make them distinct
        color = colors[i % len(colors)]
        style = linestyles[i % len(linestyles)]
        
        plt.plot(fpr, tpr, color=color, linestyle=style, lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Plot the "Random Guess" diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random Guess')

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.show()


