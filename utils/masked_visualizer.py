import matplotlib.pyplot as plt
from gymnax.visualize.vis_minatar import init_minatar, update_minatar
import numpy as np

from gymnax.visualize import Visualizer


# Custom visualizer for masked environments
def init_mask(axs, env, state):
    import seaborn as sns
    import matplotlib.colors as colors

    mask = env.get_mask(state)#[:,:,0]

    obs = env.get_obs(state)

    n_channels = env.obs_shape[-1]
    mask_slices = np.zeros((n_channels, 10, 10))

    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 1)
    # numerical_state = (
    #         np.amax(mask * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2)
    #         + 0.5
    # )

    for i in range(n_channels):
        if env.use_cutout:
            mask_slices[i] = mask[1:-1, 1:-1, i] * (i+1) + 0.5
        else:
            mask_slices[i] = mask[1:-1, 1:-1, i] * (i + 1) + 0.5
        axs[i+1].set_title(f"Channel {i+1} Mask")
        axs[i+1].set_xticks([])
        axs[i+1].set_yticks([])
        axs[i+1].imshow(
            mask_slices[i], cmap=cmap, norm=norm, interpolation="none"
        )


class MaskedVisualizer(Visualizer):

    def __init__(self, env, env_params, state_seq, reward_seq=None):

        print("Using Masked Visualizer")
        super().__init__(env, env_params, state_seq, reward_seq)

        n_channels = env.obs_shape[-1]

        self.fig, self.axs = plt.subplots(1, n_channels+1, figsize=(5*(n_channels+1), 5))
        self.ax = self.axs[0]

    def init(self):
        # Plot placeholder points
        if self.env.name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
        ]:
            self.im = init_minatar(self.ax, self.env, self.state_seq[0])
            self.im_mask = init_mask(self.axs, self.env, self.state_seq[0])
        else:
            assert False, "Mask visualization not implemented for non-MinAtar environments."

        # if self.display_mask:
        #      # self.im =
        self.fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])

    def update(self, frame):
        if self.env.name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
        ]:
            update_minatar(self.im, self.env, self.state_seq[frame])
        else:
            assert False, "Mask visualization not implemented for non-MinAtar environments."

        if self.reward_seq is None:
            self.ax.set_title(
                f"{self.env.name} - Step {frame + 1}", fontsize=15
            )
        else:
            self.ax.set_title(
                "{}: Step {:4.0f} - Return {:7.2f}".format(
                    self.env.name, frame + 1, self.reward_seq[frame]
                ),
                fontsize=15,
            )
