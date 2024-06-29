import os
import matplotlib.pyplot as plt

def plot_learning_curve(iterations, train_scores, val_scores, log_dir, run_name):
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