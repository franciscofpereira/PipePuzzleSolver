import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualizer(current_grid, parent_grid=None):
    # Get the directory path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the path to the images directory
    path_to_images = os.path.join(script_dir, 'images')

    fig, axs = plt.subplots(len(current_grid), len(current_grid[0]), figsize=(5, 5))

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    for ax in axs.flatten():
        ax.axis('off')

    for i, row in enumerate(current_grid):
        for j, img_tuple in enumerate(row):
            img_code = img_tuple[0]
            img_path = os.path.join(path_to_images, f"{img_code}.png")
            img = mpimg.imread(img_path)
            axs[i, j].imshow(img)

            # Check if the parent grid is provided and if the current piece is different from the parent grid
            if parent_grid is not None and current_grid[i][j] != parent_grid[i][j]:
                # Calculate the position of the text
                x_pos = axs[i, j].get_xlim()[0] + 0.5 * (axs[i, j].get_xlim()[1] - axs[i, j].get_xlim()[0])
                y_pos = axs[i, j].get_ylim()[0] + 0.5 * (axs[i, j].get_ylim()[1] - axs[i, j].get_ylim()[0])
                axs[i, j].text(x_pos, y_pos, 'X', horizontalalignment='center', verticalalignment='center', fontsize=27, color='red')


    plt.show()
