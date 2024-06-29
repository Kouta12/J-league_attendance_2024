import os
import matplotlib.pyplot as plt
from typing import List

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