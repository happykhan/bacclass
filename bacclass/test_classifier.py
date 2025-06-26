#!/usr/bin/env python3
"""
Classification Results Evaluation Module

This module provides functions to evaluate classification results, calculate metrics,
and generate visualizations for bacterial biosample classification performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score
)
from typing import Dict
from pathlib import Path
import warnings
from rich.console import Console

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Initialize console for rich output
console = Console()

def load_classification_results(csv_path: str) -> pd.DataFrame:
    """
    Load classification results from CSV file.
    
    Args:
        csv_path (str): Path to the classification results CSV
        
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe
    """
    try:
        df = pd.read_csv(csv_path)
        console.print(f"Loaded {len(df)} classification results from {csv_path}")
        
        # Ensure required columns exist
        required_cols = ['CLASSIFICATION', 'CONFIDENCE', 'REASONING', 'CATEGORY_NUMBER', 'CORRECT_CLASSIFICATION']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            console.print(f"[bold red]Missing columns:[/bold red] [yellow]{missing_cols}[/yellow]")
        return df
    except Exception as e:
        console.print(f"[bold red]Error loading classification results:[/bold red] {e}")
        raise

def clean_classification_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess classification data.
    
    Args:
        df (pd.DataFrame): Raw classification results
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove rows where either classification is missing
    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=['CLASSIFICATION', 'CORRECT_CLASSIFICATION'])
    console.print(f"[blue]Info:[/blue] Removed {initial_len - len(df_clean)} rows with missing classifications")
    
    # Standardize classification labels (remove extra spaces, capitalize)
    df_clean['CLASSIFICATION'] = df_clean['CLASSIFICATION'].astype(str).str.strip().str.title()
    df_clean['CORRECT_CLASSIFICATION'] = df_clean['CORRECT_CLASSIFICATION'].astype(str).str.strip().str.title()
    
    # Handle error classifications
    error_mask = df_clean['CLASSIFICATION'].str.contains('Error', case=False, na=False)
    if error_mask.any():
        console.print(f"[bold yellow]Warning:[/bold yellow] Found {error_mask.sum()} error classifications")
        
    return df_clean

def calculate_classification_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        df (pd.DataFrame): Cleaned classification results
        
    Returns:
        Dict: Dictionary containing various metrics
    """
    y_true = df['CORRECT_CLASSIFICATION']
    y_pred = df['CLASSIFICATION']
    
    # Get unique labels
    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    labels = [label for label in labels if label not in ['Error', 'Unknown']]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'total_samples': len(df),
        'num_classes': len(labels),
        'unique_labels': labels
    }
    
    # Per-class metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics['per_class_metrics'] = class_report
    
    return metrics

def calculate_confidence_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate metrics related to confidence scores.
    
    Args:
        df (pd.DataFrame): Classification results with confidence scores
        
    Returns:
        Dict: Confidence-related metrics
    """
    # Map confidence levels to numeric values
    confidence_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['confidence_numeric'] = df['CONFIDENCE'].map(confidence_map)
    
    # Calculate accuracy by confidence level
    confidence_accuracy = {}
    for conf_level in ['High', 'Medium', 'Low']:
        conf_subset = df[df['CONFIDENCE'] == conf_level]
        if len(conf_subset) > 0:
            acc = accuracy_score(conf_subset['CORRECT_CLASSIFICATION'], conf_subset['CLASSIFICATION'])
            confidence_accuracy[conf_level] = {
                'accuracy': acc,
                'count': len(conf_subset),
                'percentage': len(conf_subset) / len(df) * 100
            }
    
    return {
        'confidence_distribution': df['CONFIDENCE'].value_counts().to_dict(),
        'confidence_accuracy': confidence_accuracy,
        'avg_confidence_correct': df[df['CLASSIFICATION'] == df['CORRECT_CLASSIFICATION']]['confidence_numeric'].mean(),
        'avg_confidence_incorrect': df[df['CLASSIFICATION'] != df['CORRECT_CLASSIFICATION']]['confidence_numeric'].mean()
    }

def plot_confusion_matrix(df: pd.DataFrame, output_dir: str = "plots") -> str:
    """
    Create and save confusion matrix plot.
    
    Args:
        df (pd.DataFrame): Classification results
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to saved plot
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    y_true = df['CORRECT_CLASSIFICATION']
    y_pred = df['CLASSIFICATION']
    
    # Get labels and create confusion matrix
    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Classification Results')
    plt.xlabel('Predicted Classification')
    plt.ylabel('True Classification')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓[/green] Confusion matrix saved to {output_path}")
    return str(output_path)

def plot_classification_performance(metrics: Dict, output_dir: str = "plots") -> str:
    """
    Create performance metrics bar plot.
    
    Args:
        metrics (Dict): Classification metrics
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to saved plot
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract key metrics
    metric_names = ['Accuracy', 'Macro F1', 'Weighted F1', 'Macro Precision', 'Macro Recall']
    metric_values = [
        metrics['accuracy'],
        metrics['macro_f1'],
        metrics['weighted_f1'],
        metrics['macro_precision'],
        metrics['macro_recall']
    ]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Classification Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "performance_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓[/green] Performance metrics plot saved to {output_path}")
    return str(output_path)

def plot_per_class_f1_scores(metrics: Dict, output_dir: str = "plots") -> str:
    """
    Create per-class F1 score plot.
    
    Args:
        metrics (Dict): Classification metrics containing per-class results
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to saved plot
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract per-class F1 scores
    per_class = metrics['per_class_metrics']
    classes = [cls for cls in per_class.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    f1_scores = [per_class[cls]['f1-score'] for cls in classes]
    support = [per_class[cls]['support'] for cls in classes]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(classes, f1_scores, color='lightseagreen')
    
    # Add support count labels on bars
    for bar, f1, sup in zip(bars, f1_scores, support):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'F1: {f1:.3f}\nn={sup}', ha='center', va='bottom', fontsize=9)
    
    plt.title('Per-Class F1 Scores')
    plt.xlabel('Classification Category')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "per_class_f1_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓[/green] Per-class F1 scores plot saved to {output_path}")
    return str(output_path)

def plot_confidence_analysis(df: pd.DataFrame, confidence_metrics: Dict, output_dir: str = "plots") -> str:
    """
    Create confidence level analysis plots.
    
    Args:
        df (pd.DataFrame): Classification results
        confidence_metrics (Dict): Confidence-related metrics
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to saved plot
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Confidence distribution
    conf_dist = confidence_metrics['confidence_distribution']
    ax1.bar(conf_dist.keys(), conf_dist.values(), color=['red', 'orange', 'green'])
    ax1.set_title('Distribution of Confidence Levels')
    ax1.set_ylabel('Count')
    for i, (_, v) in enumerate(conf_dist.items()):
        ax1.text(i, v + max(conf_dist.values()) * 0.01, str(v), ha='center', va='bottom')
    
    # Plot 2: Accuracy by confidence level
    conf_acc = confidence_metrics['confidence_accuracy']
    levels = list(conf_acc.keys())
    accuracies = [conf_acc[level]['accuracy'] for level in levels]
    ax2.bar(levels, accuracies, color=['red', 'orange', 'green'])
    ax2.set_title('Accuracy by Confidence Level')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    for i, acc in enumerate(accuracies):
        ax2.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 3: Confidence vs Correctness
    correct_mask = df['CLASSIFICATION'] == df['CORRECT_CLASSIFICATION']
    confidence_correct = df[correct_mask]['CONFIDENCE'].value_counts()
    confidence_incorrect = df[~correct_mask]['CONFIDENCE'].value_counts()
    
    x = np.arange(len(['High', 'Medium', 'Low']))
    width = 0.35
    
    correct_counts = [confidence_correct.get(level, 0) for level in ['High', 'Medium', 'Low']]
    incorrect_counts = [confidence_incorrect.get(level, 0) for level in ['High', 'Medium', 'Low']]
    
    ax3.bar(x - width/2, correct_counts, width, label='Correct', color='lightgreen')
    ax3.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='lightcoral')
    ax3.set_title('Confidence Distribution by Correctness')
    ax3.set_ylabel('Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['High', 'Medium', 'Low'])
    ax3.legend()
    
    # Plot 4: Error rate by confidence
    error_rates = []
    conf_levels = ['High', 'Medium', 'Low']
    for level in conf_levels:
        subset = df[df['CONFIDENCE'] == level]
        if len(subset) > 0:
            error_rate = 1 - accuracy_score(subset['CORRECT_CLASSIFICATION'], subset['CLASSIFICATION'])
            error_rates.append(error_rate)
        else:
            error_rates.append(0)
    
    ax4.bar(conf_levels, error_rates, color=['darkred', 'darkorange', 'darkgreen'])
    ax4.set_title('Error Rate by Confidence Level')
    ax4.set_ylabel('Error Rate')
    ax4.set_ylim(0, max(error_rates) * 1.1 if error_rates else 1)
    for i, err in enumerate(error_rates):
        ax4.text(i, err + max(error_rates) * 0.02, f'{err:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "confidence_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓[/green] Confidence analysis plot saved to {output_path}")
    return str(output_path)

def generate_evaluation_report(df: pd.DataFrame, metrics: Dict, confidence_metrics: Dict, 
                             output_path: str = "evaluation_report.txt") -> str:
    """
    Generate a comprehensive text report of evaluation results.
    
    Args:
        df (pd.DataFrame): Classification results
        metrics (Dict): Classification metrics
        confidence_metrics (Dict): Confidence metrics
        output_path (str): Path to save the report
        
    Returns:
        str: Path to saved report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Number of Classes: {metrics['num_classes']}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1 Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}\n")
        f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        per_class = metrics['per_class_metrics']
        for cls in sorted(per_class.keys()):
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"\n{cls}:\n")
                f.write(f"  Precision: {per_class[cls]['precision']:.4f}\n")
                f.write(f"  Recall: {per_class[cls]['recall']:.4f}\n")
                f.write(f"  F1-Score: {per_class[cls]['f1-score']:.4f}\n")
                f.write(f"  Support: {per_class[cls]['support']}\n")
        
        # Confidence analysis
        f.write("\n\nCONFIDENCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        conf_dist = confidence_metrics['confidence_distribution']
        for level, count in conf_dist.items():
            percentage = count / metrics['total_samples'] * 100
            f.write(f"{level} Confidence: {count} samples ({percentage:.1f}%)\n")
        
        f.write("\nAccuracy by Confidence Level:\n")
        conf_acc = confidence_metrics['confidence_accuracy']
        for level, data in conf_acc.items():
            f.write(f"  {level}: {data['accuracy']:.4f} (n={data['count']})\n")
        
        # Most common misclassifications
        f.write("\n\nMOST COMMON MISCLASSIFICATIONS\n")
        f.write("-" * 30 + "\n")
        misclassified = df[df['CLASSIFICATION'] != df['CORRECT_CLASSIFICATION']]
        if len(misclassified) > 0:
            error_pairs = misclassified.groupby(['CORRECT_CLASSIFICATION', 'CLASSIFICATION']).size()
            error_pairs = error_pairs.sort_values(ascending=False).head(10)
            for (true_class, pred_class) in error_pairs.index:
                count = error_pairs[(true_class, pred_class)]
                f.write(f"  {true_class} → {pred_class}: {count} cases\n")
        else:
            f.write("  No misclassifications found!\n")
    
    console.print(f"[green]✓[/green] Evaluation report saved to {output_path}")
    return output_path

def evaluate_classification_results(csv_path: str, output_dir: str = "evaluation_results") -> Dict:
    """
    Main function to evaluate classification results and generate all outputs.
    
    Args:
        csv_path (str): Path to classification results CSV
        output_dir (str): Directory to save all outputs
        
    Returns:
        Dict: Complete evaluation results
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    console.print(f"[blue]Starting evaluation[/blue] of classification results from [cyan]{csv_path}[/cyan]")
    
    # Load and clean data
    df = load_classification_results(csv_path)
    df_clean = clean_classification_data(df)
    
    # Calculate metrics
    metrics = calculate_classification_metrics(df_clean)
    confidence_metrics = calculate_confidence_metrics(df_clean)
    
    # Generate plots
    plot_paths = {
        'confusion_matrix': plot_confusion_matrix(df_clean, str(plots_dir)),
        'performance_metrics': plot_classification_performance(metrics, str(plots_dir)),
        'per_class_f1': plot_per_class_f1_scores(metrics, str(plots_dir)),
        'confidence_analysis': plot_confidence_analysis(df_clean, confidence_metrics, str(plots_dir))
    }
    
    # Generate report
    report_path = generate_evaluation_report(
        df_clean, metrics, confidence_metrics, 
        str(Path(output_dir) / "evaluation_report.txt")
    )
    
    # Compile evaluation results
    evaluation_results = {
        'data_summary': {
            'total_samples': len(df),
            'clean_samples': len(df_clean),
            'removed_samples': len(df) - len(df_clean)
        },
        'metrics': metrics,
        'confidence_metrics': confidence_metrics,
        'plot_paths': plot_paths,
        'report_path': report_path,
        'output_directory': output_dir
    }
    
    console.print("✅ Evaluation completed successfully!", style="bold green")
    console.print(f"📁 Results saved to: {output_dir}", style="cyan")
    console.print(f"🎯 Overall Accuracy: {metrics['accuracy']:.4f}", style="bold blue")
    console.print(f"📊 Macro F1 Score: {metrics['macro_f1']:.4f}", style="bold blue")
    
    return evaluation_results
