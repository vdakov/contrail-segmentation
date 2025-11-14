import matplotlib.pyplot as plt 
import numpy as np

def plot_aug_comparison(acc_no_aug_list, acc_aug_list, model_no_data_aug, model_data_aug):
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Accuracy comparison
    accuracies = [acc_no_aug_list[-1], acc_aug_list[-1]]
    labels = ['No Data Augmentation', 'With Data Augmentation']

    axs[0].bar(labels, accuracies, color=['red', 'green'])
    axs[0].set_ylim([0, 1])
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Model Accuracy Comparison')
    for i, acc in enumerate(accuracies):
        axs[0].text(i, acc + 0.02, f"{acc:.4f}", ha='center')

    # Feature importance comparison
    fi_no_aug = model_no_data_aug.feature_importances_
    fi_aug = model_data_aug.feature_importances_

    indices = np.arange(len(fi_no_aug))
    axs[1].bar(indices - 0.2, fi_no_aug, width=0.4, label='No Augmentation')
    axs[1].bar(indices + 0.2, fi_aug, width=0.4, label='With Augmentation')
    axs[1].set_xlabel('Feature Index')
    axs[1].set_ylabel('Feature Importance')
    axs[1].set_title('Feature Importances Comparison')
    axs[1].legend()

    plt.show()