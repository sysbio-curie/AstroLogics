There is some part of the script that I think could be added into the later version of the code


# tsne embedding
## Calculating tsne
```{python}
from sklearn.manifold import TSNE

# Calculate the UMAP coordinates
distance_matrix = model.trajectory.distance_matrix
tsne = TSNE(
    n_components=2,
    metric="precomputed",
    init = 'random',
    perplexity=30,      # must be < n/3
    learning_rate="auto",
    random_state=12345,
    n_iter=1000,
    verbose=1
)
coords = tsne.fit_transform(distance_matrix)

# Store the UMAP coordinates in a DataFrame
coords_vis = pd.DataFrame(coords, columns=['x', 'y'], index=distance_matrix.index)
model.trajectory.mds_coords = coords_vis
```

## Visualization of the tsne latent space
```{python}
coords = model.trajectory.mds_coords
plt.figure(figsize=(10,10))
# Plot without cluster labels
sns.scatterplot(x=coords['x'], y=coords['y'],
                alpha=0.5, s=10)

plt.grid(False)
plt.show()
```

## Visualize the attractor group in tsne latent space

```{python}

coords = model.trajectory.mds_coords
coords['attractor_group'] = coords.index.map(attractor_group_dict)

plt.figure(figsize=(8, 8))
sns.scatterplot(
    x=coords['x'],
    y=coords['y'],
    hue=coords['attractor_group'],
    palette='tab10',
    s=100,
    alpha=0.8
)
plt.legend(title="Attractor Group")
plt.grid(False)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Attractor Groups (PCA)')
plt.show()
```
# UMAP
## Calculating the UMAP

```{python}

import umap
distance_matrix = model.trajectory.distance_matrix
# UMAP with precomputed distances
reducer = umap.UMAP(
    metric="precomputed",
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
coords = reducer.fit_transform(distance_matrix)  # shape (n_samples, 2)

# Store the UMAP coordinates in a DataFrame
coords_vis = pd.DataFrame(coords, columns=['x', 'y'], index=distance_matrix.index)
model.trajectory.mds_coords = coords_vis
```

## Visualizing the UMAP space
```{python}
coords = coords_vis
plt.figure(figsize=(10,10))
# Plot without cluster labels
sns.scatterplot(x=coords['x'], y=coords['y'],
                alpha=0.5, s=10)

plt.grid(False)
plt.show()

```

## Visualizing the clustering on to the UMAP projection
```{python}
attractor_group_path = '/home/spankaew/Git/astrologics/data/attractor_group/attractor_group/' + model_name + '_attractor_group.csv'
attractor_group = pd.read_csv(attractor_group_path, index_col = 0)
attractor_group_dict = attractor_group['attractor_group'].to_dict()


coords = model.trajectory.mds_coords
coords['attractor_group'] = coords.index.map(attractor_group_dict)

plt.figure(figsize=(8, 8))
sns.scatterplot(
    x=coords['x'],
    y=coords['y'],
    hue=coords['attractor_group'],
    palette='tab10',
    s=100,
    alpha=0.8
)
plt.legend(title="Attractor Group")
plt.grid(False)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Attractor Groups (PCA)')
plt.show()
```