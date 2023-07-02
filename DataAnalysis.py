import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the Pokémon dataset into a DataFrame
pokemon_data = pd.read_csv('pokemon_dataset.csv')

# Display the first few rows of the dataset
print(pokemon_data.head())

# Check the summary statistics of the dataset
print(pokemon_data.describe())

# Check for missing values
print(pokemon_data.isnull().sum())

# Explore the distribution of variables
print(pokemon_data['Type 1'].value_counts())

# Data Visualization
# Histogram of Base Experience
plt.hist(pokemon_data['BaseExperience'], bins=20)
plt.xlabel('Base Experience')
plt.ylabel('Frequency')
plt.title('Distribution of Pokémon Base Experience')
plt.show()

# Bar plot of Pokémon Types
type_counts = pokemon_data['Type 1'].value_counts().sort_values(ascending=False)
plt.bar(type_counts.index, type_counts.values)
plt.xlabel('Pokémon Type')
plt.ylabel('Count')
plt.title('Count of Pokémon Types')
plt.xticks(rotation=90)
plt.show()

# Scatter plot of Attack vs. Defense
sns.scatterplot(x='Attack', y='Defense', data=pokemon_data)
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Pokémon Attack vs. Defense')
plt.show()

# Correlation Analysis
correlation_matrix = pokemon_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Legendary Pokémon Analysis
legendary_pokemon = pokemon_data[pokemon_data['Legendary'] == True]
print(legendary_pokemon.describe())

legendary_type_counts = legendary_pokemon['Type 1'].value_counts().sort_values(ascending=False)
plt.bar(legendary_type_counts.index, legendary_type_counts.values)
plt.xlabel('Pokémon Type')
plt.ylabel('Count')
plt.title('Count of Pokémon Types (Legendary Pokémon)')
plt.xticks(rotation=90)
plt.show()

# Type Combination Analysis
pokemon_data['Type Combination'] = pokemon_data['Type 1'] + '-' + pokemon_data['Type 2']
type_combination_counts = pokemon_data['Type Combination'].value_counts().sort_values(ascending=False)
plt.bar(type_combination_counts.index, type_combination_counts.values)
plt.xlabel('Type Combination')
plt.ylabel('Count')
plt.title('Count of Pokémon Type Combinations')
plt.xticks(rotation=90)
plt.show()

# Statistical Hypothesis Testing
type1 = pokemon_data[pokemon_data['Type 1'] == 'Water']['Attack']
type2 = pokemon_data[pokemon_data['Type 1'] == 'Fire']['Attack']
t_statistic, p_value = ttest_ind(type1, type2)
print('t-statistic:', t_statistic)
print('p-value:', p_value)

# Dimensionality Reduction with PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pokemon_data.iloc[:, 4:10])  # Selecting numeric columns for PCA
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Pokémon Attributes')
plt.show()

# Dimensionality Reduction with t-SNE
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(pokemon_data.iloc[:, 4:10])  # Selecting numeric columns for t-SNE
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE of Pokémon Attributes')
plt.show()
