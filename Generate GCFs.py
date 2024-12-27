import numpy as np
from sklearn.decomposition import PCA

# Constants for scaling in the model
sof_x = 100  # Sof for x-axis
sof_y = 100  # Sof for y-axis

# ------------------------------ Helper Functions -----------------------------
def generate_coordinates(x_range, y_range):
    """
    Generates a grid of coordinates based on the given x and y ranges.
    Returns a numpy array of shape (len(x_range) * len(y_range), 2) containing coordinate pairs.
    """
    coordinates = np.empty(shape=(len(x_range) * len(y_range), 2))
    index = 0
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            coordinates[index, 0] = x_range[i]
            coordinates[index, 1] = y_range[j]
            index += 1
    return coordinates

def calculate_correlation_matrix(coordinates1, coordinates2, sof_x, sof_y):
    """
    Calculates the correlation matrix between two sets of coordinates.
    The correlation is calculated based on the distance between points in both sets.
    """
    correlation_matrix = np.empty(shape=(len(coordinates1), len(coordinates2)))
    for i in range(len(coordinates1)):
        # Calculate the absolute distance between points
        distance_x = np.abs(coordinates2[:, 0] - coordinates1[i, 0])
        distance_y = np.abs(coordinates2[:, 1] - coordinates1[i, 1])

        # Compute the correlation using a specified formula
        correlation_matrix[i, :] = (1 + 4 * (distance_x / sof_x)) * (1 + 4 * (distance_y / sof_y)) * \
                                   np.exp(-4 * ((distance_x / sof_x) + (distance_y / sof_y)))
    return correlation_matrix

def apply_pca(correlation_matrix, n_components=300):
    """
    Applies PCA to the correlation matrix for dimensionality reduction.
    Returns the reduced correlation matrix.
    """
    pca = PCA(n_components=n_components, svd_solver='full')
    return pca.fit_transform(correlation_matrix), pca

# ------------------------------ Generate Coordinates -----------------------------
grid_x = np.arange(0.5, 50, 1)  # X coordinates for the full grid with a step of 1
grid_y = np.arange(0.5, 50, 1)  # Y coordinates for the full grid with a step of 1
gcf_x = np.arange(0.5, 50, 2)  # X coordinates for GCF with a step of 2
gcf_y = np.arange(0.5, 50, 2)  # Y coordinates for GCF with a step of 2

# Generate coordinates for the full grid and GCF
full_coordinates = generate_coordinates(grid_x, grid_y)
gcf_coordinates = generate_coordinates(gcf_x, gcf_y)

# Reshape the full_coordinates array for easy manipulation in 3D
reshaped_full_coordinates = full_coordinates.reshape(len(grid_x), len(grid_y), 2)

# ------------------------------ Generate Correlation Matrices -----------------------------
# Calculate the correlation matrix between GCF points
gcf_correlation_matrix = calculate_correlation_matrix(gcf_coordinates, gcf_coordinates, sof_x, sof_y)

# Apply PCA to the GCF correlation matrix
gcf_correlation_matrix_pca, pca = apply_pca(gcf_correlation_matrix)

# Calculate the correlation matrix between full grid points and GCF points
full_to_gcf_correlation_matrix = calculate_correlation_matrix(full_coordinates, gcf_coordinates, sof_x, sof_y)

# Apply PCA transformation to the full-to-GCF correlation matrix using the same PCA model
full_to_gcf_correlation_matrix_pca = pca.transform(full_to_gcf_correlation_matrix)

