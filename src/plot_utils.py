import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_losses(train_values, val_values, train_labels, val_labels):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for i in range(len(train_values)):
        ax[0].plot(train_values[i], label=train_labels[i])
    for i in range(len(val_values)):
        ax[1].plot(val_values[i], label=val_labels[i], linestyle='--')
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    return fig
