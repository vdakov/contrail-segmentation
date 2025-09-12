from matplotlib import pyplot as plt

def compare_two_losses(loss_a, loss_a_val, loss_a_label, loss_b, loss_b_val, loss_b_label):


    plt.plot(loss_a, label=loss_a_label)
    plt.plot(loss_a_val, label=loss_a_label + ' (val)')
    plt.plot(loss_b, label=loss_b_label)
    plt.plot(loss_b_val, label=loss_b_label + ' (val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    plt.title('Loss Comparison')
    plt.show()
    
def plot_accuracy_vs_subset_size(subset_sizes, acc_no_aug, acc_with_aug, 
                               title='Model Accuracy vs Training Subset Size',
                               xlabel='Subset Size', 
                               ylabel='Accuracy on Test Set',
                               figsize=(10, 6),
                               grid=True,
                               save_path=None):
    """
    Plots model accuracy comparison with and without data augmentation across different subset sizes.
    
    Parameters:
    - subset_sizes (list): List of training subset sizes
    - acc_no_aug (list): Accuracy values without data augmentation
    - acc_with_aug (list): Accuracy values with data augmentation
    - title (str): Plot title
    - xlabel (str): X-axis label
    - ylabel (str): Y-axis label
    - figsize (tuple): Figure dimensions
    - grid (bool): Whether to show grid
    - save_path (str): Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.plot(subset_sizes, acc_no_aug, marker='o', label='No Data Augmentation')
    plt.plot(subset_sizes, acc_with_aug, marker='o', label='With Data Augmentation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    if grid:
        plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()
