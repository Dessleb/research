import matplotlib.pyplot as plt
import time
import seaborn as sns
from tabulate import tabulate
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, f1_score, auc, recall_score, accuracy_score, precision_score, roc_auc_score, roc_curve
import numpy as np



def grouped_bar_chart(models, metric):
    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in metric.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Evaluation Metrics of models for detecting diabetes')
    ax.set_xticks(x + width, models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 150)

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





















# def plot_model_metrics(model, X_test, y_test, class_labels=None, title="Model Performance Metrics"):
    
#     # Args:
#     #     model: Trained scikit-learn model object (must have .predict() and .predict_proba()).
#     #     X_test (np.array or pd.DataFrame): Test features.
#     #     y_test (np.array or pd.Series): True labels for the test data.
#     #     class_labels (list, optional): List of names for the classes. 
#     #                                     Defaults to ['Class 0', 'Class 1'].
#     #     title (str, optional): Main title for the plot.
#     # """
    
#     # --- 1. Setup and Predictions ---
#     print(f"--- Calculating metrics for: {title} ---")
    
#     # Get class predictions
#     y_pred = model.predict(X_test)
    
#     # Get probabilities for the positive class (assuming binary classification)
#     try:
#         # Check if the model has predict_proba
#         if hasattr(model, 'predict_proba'):
#             y_proba = model.predict_proba(X_test)[:, 1]
#         elif hasattr(model, 'decision_function'):
#             # For SVC without probability=True, we can use decision_function
#             # but we won't calculate ROC in this case for simplicity, as it's less reliable
#             print("Warning: Model uses decision_function, but predict_proba is preferred for robust ROC/AUC.")
#             return
#         else:
#             print("Warning: Model does not have predict_proba. Cannot calculate ROC/AUC.")
#             return

#     except AttributeError:
#         print("Error: Model prediction failed.")
#         return

#     # Set default labels if none are provided
#     if class_labels is None:
#         class_labels = ['Class 0', 'Class 1']

#     # Create a figure with two subplots side-by-side
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     plt.suptitle(title, fontsize=16, fontweight='bold')
    
    
#     # --- 2. Confusion Matrix Plot ---
    
#     # Calculate the confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
    
#     # Calculate key metrics for display
#     accuracy = np.mean(y_test == y_pred)
#     # Handle division by zero for precision/recall calculation
#     precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
#     recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     # Plot the matrix using seaborn
#     sns.heatmap(cm, 
#                 annot=True, 
#                 fmt='d', 
#                 cmap='Blues', 
#                 cbar=False,
#                 linewidths=.5,
#                 linecolor='black',
#                 square=True,
#                 ax=axes[0],
#                 xticklabels=class_labels,
#                 yticklabels=class_labels)
    
#     axes[0].set_title('Confusion Matrix', fontsize=14)
#     axes[0].set_xlabel(f'Predicted Label\n\nMetrics: Acc={accuracy:.2f}, Prec={precision:.2f}, Rec={recall:.2f}, F1={f1_score:.2f}')
#     axes[0].set_ylabel('True Label')
    
#     # Adjust rotation for better readability
#     plt.setp(axes[0].get_xticklabels(), rotation=0)
#     plt.setp(axes[0].get_yticklabels(), rotation=90)
    
    
#     # --- 3. AUC-ROC Curve Plot ---
    
#     # Calculate ROC curve metrics (False Positive Rate, True Positive Rate, Thresholds)
#     fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
#     # Calculate the Area Under the Curve (AUC)
#     roc_auc = roc_auc_score(y_test, y_proba)
    
#     # Plot the ROC curve
#     axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
#                  label=f'ROC curve (AUC = {roc_auc:.4f})')
    
#     # Plot the 45-degree diagonal line (random guess line)
#     axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
#     axes[1].set_xlim([0.0, 1.0])
#     axes[1].set_ylim([0.0, 1.05])
#     axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
#     axes[1].set_ylabel('True Positive Rate (Recall/Sensitivity)')
#     axes[1].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
#     axes[1].legend(loc="lower right")
#     axes[1].grid(True)
    
#     # --- 4. Final Display ---
#     plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
#     # plt.show() # NOTE: Commented out for Canvas environment

# # --- New Function for Multi-Model ROC Comparison ---

# def plot_roc_comparison(models, X_test, y_test, title="Model ROC Curve Comparison"):
#     """
#     Plots the ROC curves for multiple models on a single graph for comparison.

#     This function is designed for binary classification problems.

#     Args:
#         models (list): A list of tuples, where each tuple is (model_name, trained_model_object).
#                        The model object must have a .predict_proba() method.
#         X_test (np.array or pd.DataFrame): Test features.
#         y_test (np.array or pd.Series): True labels for the test data.
#         title (str, optional): Title for the plot.
#     """
#     plt.figure(figsize=(10, 8))
    
#     # Plot the 45-degree diagonal line (random guess line)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

#     # Iterate through models and plot ROC curves
#     for name, model in models:
#         try:
#             # Get probabilities for the positive class
#             # Note: If you use XGBoost, ensure you import and use it instead of GradientBoostingClassifier
#             y_proba = model.predict_proba(X_test)[:, 1]
            
#             # Calculate ROC curve metrics
#             fpr, tpr, _ = roc_curve(y_test, y_proba)
#             roc_auc = roc_auc_score(y_test, y_proba)
            
#             # Plot the curve
#             plt.plot(fpr, tpr, lw=2, 
#                      label=f'{name} (AUC = {roc_auc:.4f})')
        
#         except AttributeError:
#             print(f"Warning: Model '{name}' does not have predict_proba and was skipped for ROC comparison.")
#             continue

#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate (1 - Specificity)')
#     plt.ylabel('True Positive Rate (Recall/Sensitivity)')
#     plt.title(title, fontsize=16, fontweight='bold')
#     plt.legend(loc="lower right", title="Models")
#     plt.grid(True)
#     plt.show() # NOTE: For display in the user environment

# # --- Example Usage for Diabetes Detection Models ---

# if __name__ == '__main__':
#     # 1. Generate Synthetic Data for Binary Classification (Simulating Diabetes Detection)
#     # Replace this with your actual diabetes dataset loading and preprocessing!
#     X, y = make_classification(n_samples=500, 
#                                n_features=8, 
#                                n_informative=5, 
#                                n_redundant=1,
#                                n_classes=2,
#                                weights=[0.8, 0.2], # Simulate imbalanced data (more non-diabetic)
#                                flip_y=0.05, 
#                                random_state=42)
    
#     # 2. Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     # 3. Initialize and train the three models (SVC, XGBoost stand-in, RF)
#     # IMPORTANT: Ensure your SVC is trained with probability=True
    
#     # a. Support Vector Classifier (SVC) - Must use probability=True for ROC
#     svc_model = SVC(probability=True, random_state=42)
#     svc_model.fit(X_train, y_train)
    
#     # b. Random Forest (RF)
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)

#     # c. Gradient Boosting Classifier (Used as XGBoost stand-in, as it's a built-in scikit-learn ensemble)
#     # If you have the actual 'xgboost' library installed, replace this with: 
#     # from xgboost import XGBClassifier; gb_model = XGBClassifier(random_state=42); gb_model.fit(X_train, y_train)
#     gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#     gb_model.fit(X_train, y_train)
    
#     # 4. Prepare model list for comparison
#     models_to_compare = [
#         ('SVC', svc_model),
#         ('Random Forest (RF)', rf_model),
#         ('Gradient Boosting (XGB)', gb_model)
#     ]

#     # --- Use Case 1: Individual Model Deep Dive (e.g., Random Forest) ---
#     print("\n[FIRST PLOT: Individual Model Deep Dive (Random Forest)]")
#     # This generates a Confusion Matrix and ROC curve for one model
#     plot_model_metrics(
#         rf_model, 
#         X_test, 
#         y_test, 
#         class_labels=['No Diabetes', 'Diabetes'],
#         title="Random Forest Model Metrics for Diabetes Detection"
#     )

#     # --- Use Case 2: Multi-Model ROC Comparison ---
#     print("\n[SECOND PLOT: Multi-Model ROC Comparison]")
#     # This generates a single plot comparing the ROC curves of all three models
#     plot_roc_comparison(
#         models_to_compare, 
#         X_test, 
#         y_test, 
#         title="Comparative ROC Analysis for Diabetes Detection Models"
#     )