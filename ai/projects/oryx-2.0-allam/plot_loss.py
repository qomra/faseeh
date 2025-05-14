import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Set the aesthetics for the plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor': 'none'})

# Define color palette - modern and visually appealing with high contrast
colors = sns.color_palette("mako", 10)
# Use a more distinct color palette for reward components
reward_colors = sns.color_palette("tab10", 10)  # More distinct colors

def parse_jsonl(file_path):
    """Parse JSONL file into a list of dictionaries"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Replace single quotes with double quotes for proper JSON parsing
                fixed_line = line.replace("'", '"')
                try:
                    entry = json.loads(fixed_line)
                    data.append(entry)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
    return data

def data_to_dataframe(data):
    """Convert list of dictionaries to pandas DataFrame"""
    df = pd.DataFrame(data)
    
    # Sort by epoch if present
    if 'epoch' in df.columns:
        df = df.sort_values('epoch')
    
    return df

def plot_metrics(df, output_dir=None):
    """Plot the training metrics in visually appealing way"""
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Group metrics by type
    loss_metrics = [col for col in numeric_cols if 'loss' in col.lower()]
    reward_metrics = [col for col in numeric_cols if 'reward' in col.lower() and 'std' not in col.lower()]
    learning_metrics = [col for col in numeric_cols if any(term in col.lower() for term in ['learning', 'grad', 'kl'])]
    other_metrics = [col for col in numeric_cols if col not in loss_metrics + reward_metrics + learning_metrics 
                    and col != 'epoch' and 'length' not in col.lower()]
    
    # Create subplots with consistent styling
    x_col = 'epoch' if 'epoch' in df.columns else df.index
    
    # 1. Plot loss metrics
    if loss_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, metric in enumerate(loss_metrics):
            sns.lineplot(x=x_col, y=metric, data=df, marker='o', markersize=4, 
                        linewidth=2, color=colors[i], label=metric, ax=ax)
            
        ax.set_title('Training Loss', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch' if 'epoch' in df.columns else 'Training Step', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray', 
               loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend outside plot
        
        if output_dir:
            plt.savefig(f"{output_dir}/loss_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 2. Plot reward metrics
    if reward_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, metric in enumerate(reward_metrics):
            sns.lineplot(x=x_col, y=metric, data=df, marker='o', markersize=4, 
                        linewidth=2, color=reward_colors[i % len(reward_colors)], label=metric, ax=ax)
            
        ax.set_title('Reward Metrics', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch' if 'epoch' in df.columns else 'Training Step', fontsize=12)
        ax.set_ylabel('Reward Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray')
        
        if output_dir:
            plt.savefig(f"{output_dir}/reward_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 3. Plot learning rate and related metrics
    if learning_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create a secondary y-axis for different scales
        ax2 = ax.twinx()
        
        # Determine which metrics go on which axis
        primary_metrics = [m for m in learning_metrics if 'learning_rate' not in m]
        secondary_metrics = [m for m in learning_metrics if 'learning_rate' in m]
        
        for i, metric in enumerate(primary_metrics):
            sns.lineplot(x=x_col, y=metric, data=df, marker='o', markersize=4, 
                        linewidth=2, color=colors[i], label=metric, ax=ax)
        
        for i, metric in enumerate(secondary_metrics):
            sns.lineplot(x=x_col, y=metric, data=df, marker='s', markersize=4, 
                        linewidth=2, color=colors[i+len(primary_metrics)], label=metric, ax=ax2)
        
        ax.set_title('Learning Dynamics', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch' if 'epoch' in df.columns else 'Training Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Merge legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, 
                frameon=True, facecolor='white', edgecolor='lightgray')
        
        if output_dir:
            plt.savefig(f"{output_dir}/learning_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 4. Individual reward components
    reward_components = [col for col in reward_metrics if '/' in col]
    if reward_components:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use highly distinct markers and line styles in addition to colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        line_styles = ['-', '--', '-.', ':']
        
        for i, metric in enumerate(reward_components):
            label = metric.split('/')[-1].replace('_reward_func', '')
            sns.lineplot(x=x_col, y=metric, data=df, 
                        marker=markers[i % len(markers)], 
                        markersize=8,  # Larger markers
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=3,  # Thicker lines
                        color=reward_colors[i % len(reward_colors)], 
                        label=label, 
                        ax=ax)
            
        ax.set_title('Reward Components', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch' if 'epoch' in df.columns else 'Training Step', fontsize=12)
        ax.set_ylabel('Component Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray')
        
        if output_dir:
            plt.savefig(f"{output_dir}/reward_components.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 5. Multi-panel overview plot
    key_metrics = []
    if 'loss' in numeric_cols:
        key_metrics.append('loss')
    if 'reward' in numeric_cols:
        key_metrics.append('reward')
    if 'learning_rate' in numeric_cols:
        key_metrics.append('learning_rate')
    if 'kl' in numeric_cols:
        key_metrics.append('kl')
    
    if len(key_metrics) > 1:
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 4*len(key_metrics)), sharex=True)
        
        for i, metric in enumerate(key_metrics):
            ax = axes[i] if len(key_metrics) > 1 else axes
            sns.lineplot(x=x_col, y=metric, data=df, marker='o', markersize=4, 
                        linewidth=2, color=colors[i], ax=ax)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if i == len(key_metrics) - 1:
                ax.set_xlabel('Epoch' if 'epoch' in df.columns else 'Training Step', fontsize=12)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/training_overview.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics from JSONL file')
    parser.add_argument('file', type=str, help='Path to the JSONL file')
    parser.add_argument('--output', type=str, help='Output directory for saving plots', default=None)
    args = parser.parse_args()
    
    # Parse the JSONL file
    data = parse_jsonl(args.file)
    
    if not data:
        print("No valid data found in the file.")
        return
    
    # Convert to DataFrame and plot
    df = data_to_dataframe(data)
    plot_metrics(df, args.output)
    
    print(f"Generated plots for {len(df)} data points across {len(df.columns)} metrics.")
    if args.output:
        print(f"Plots saved to {args.output}")

if __name__ == "__main__":
    main()