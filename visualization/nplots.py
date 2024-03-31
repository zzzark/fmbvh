import numpy as np
import matplotlib.pyplot as plt


def visualize_arrays(arrays, cmap="viridis", interpolation="catrom"):
    """
    :param arrays: [..., W, H] or list of [W, H]
    """
    # LIST OR TUPLE
    if isinstance(arrays, list) or isinstance(arrays, tuple):
        if not isinstance(arrays[0], np.ndarray):
            import torch
            assert isinstance(arrays[0], torch.Tensor), "should be list of numpy.ndarray or torch.tensor"
            arrays = [e.detach().cpu().numpy() for e in arrays]
    # NUMPY OR TORCH
    elif not isinstance(arrays, np.ndarray):
        import torch
        assert isinstance(arrays, torch.Tensor), "should numpy.ndarray or torch.tensor"
        arrays = arrays.detach().cpu().numpy()
        if len(arrays.shape) >= 4:
            arrays = arrays.reshape(-1, arrays.shape[-2], arrays.shape[-1])
    
    num_plots = len(arrays)
    num_cols = int(num_plots**0.5+0.9)  # Number of columns in the subplot grid
    num_rows = -(-num_plots // num_cols)  # Ceiling division to determine number of rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))

    for i, array in enumerate(arrays):
        if len(array.shape) == 1:
            array = array[:, None]
            
        row = i // num_cols
        col = i % num_cols

        if num_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]

        ax.imshow(array, cmap=cmap, interpolation=interpolation)
        ax.set_title(f'Array {i+1}')
        ax.axis('off')

    # If the number of plots is not a multiple of num_cols, hide the extra subplots
    for i in range(num_plots, num_rows*num_cols):
        if num_rows > 1:
            axs.flatten()[i].axis('off')
        else:
            axs[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch
    arrays = torch.randn(3, 3, 9, 7)
    visualize_arrays(arrays)
