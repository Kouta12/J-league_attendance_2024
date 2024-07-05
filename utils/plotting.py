import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, OrderedDict

def plot_learning_curve(
        iterations: List[int], 
        train_scores: List[float], 
        val_scores: List[float], 
        log_dir: str, 
        run_name: str,
) -> str:
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_scores, label='Training RMSE')
    plt.plot(iterations, val_scores, label='Validation RMSE')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    
    # プロットの保存
    plot_path = os.path.join(log_dir, f"{run_name}_learning_curve.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def calculate_average(data_list):
    train_values = []
    valid_values = []
    
    for item in data_list:
        train_values.append(item['train']['rmse'][0])
        valid_values.append(item['valid']['rmse'][0])
    
    # 全ての訓練で同じ数のブースティングラウンドがあることを確認
    min_length = min(len(train) for train in train_values)
    
    # 各ブースティングラウンドごとに平均を計算
    train_avg = np.mean([train[:min_length] for train in train_values], axis=0)
    valid_avg = np.mean([valid[:min_length] for valid in valid_values], axis=0)
    
    return {
        'train': OrderedDict([('rmse', [train_avg.tolist()])]),
        'valid': OrderedDict([('rmse', [valid_avg.tolist()])])
    }

def plot_learning_curve_ensemble(run_name: str, BASE_DIR: str, eval_results_list: List[Dict[str, OrderedDict[str, float]]]):
    eval_average = calculate_average(eval_results_list)
    
    plt.figure(figsize=(10, 6))

    # Plot the RMSE during training
    plt.plot(eval_average['train']['rmse'], label='train')
    plt.plot(eval_average['valid']['rmse'], label='valid')
    
    plt.ylabel('RMSE')
    plt.xlabel('Boosting round')
    plt.title(f'{run_name} - Training performance')
    plt.legend()

    # プロットの保存
    plot_path = os.path.join(BASE_DIR, "logs", run_name, f"{run_name}_learning_curve_ensemble.png")
    plt.savefig(plot_path)
    plt.close()