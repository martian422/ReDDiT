import torch
import matplotlib.pyplot as plt
import numpy as np


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


# Example grid dimensions
grid_size = 16
half_dim = 64  # Example dimension for the rotary encoding

# Create a 2D frequency grid
t = torch.arange(0, grid_size).float()  # For example, grid size = 16
inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))  # Assuming base=10000
freqs = torch.einsum("i,j->ij", t, inv_freq)
breakpoint()

# Create a 2D frequency grid by concatenating along the second axis (rows and columns)
# freqs_grid = torch.cat([
#     freqs[:, None, :].expand(-1, grid_size, -1),  # Expand for rows
#     freqs[None, :, :].expand(grid_size, -1, -1),  # Expand for columns
# ], dim=-1)
x_grid = freqs[:, None, :].expand(-1, grid_size, -1)
y_grid = freqs[None, :, :].expand(grid_size, -1, -1)
freqs_grid = x_grid + y_grid
# Cosine and sine transformations of the frequencies (We can pick the first frequency component)
cos_vals = freqs_grid[:, :, 0].cos().detach().cpu().numpy()  # Using the first frequency component (cosine)
sin_vals = freqs_grid[:, :, 0].sin().detach().cpu().numpy()  # Using the first frequency component (sine)

# Compute the phase (angle) using atan2
phase_vals = np.arctan2(sin_vals, cos_vals)

# Normalize the phase to the range [0, 1] for better visualization (optional)
phase_vals = (phase_vals + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

# Plotting the phase values of the frequency grid
plt.figure(figsize=(8, 6))

# Plot the Phase Values
plt.imshow(phase_vals, cmap="twilight", aspect='auto')  # Use 'twilight' colormap for phase visualization
plt.colorbar()
plt.title("Phase of First Frequency Channel")

# Save the plot as an image file (e.g., PNG or JPG)
plt.tight_layout()
plt.savefig("rotary_2d_phase_channel2.png")  # Saves the plot to a file
plt.show()  # Show the plot in a window

# Close the plot to free memory
plt.close()

print("Phase plot saved as 'rotary_2d_phase_channel.png'")
