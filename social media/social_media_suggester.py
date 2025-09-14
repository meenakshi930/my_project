import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("movies.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Step 2: Handle NaN in 'genres'
df["genres"] = df["genres"].fillna("")

# Step 3: Convert genres into feature vectors
cv = CountVectorizer(tokenizer=lambda x: x.split('|'))  # genres are separated by '|'
count_matrix = cv.fit_transform(df["genres"])

# Step 4: Compute cosine similarity
cosine_sim = cosine_similarity(count_matrix)
similarity_df = pd.DataFrame(cosine_sim, index=df["title"], columns=df["title"])

# Step 5: Recommend movies based on genres
def recommend_movies(movie_title, top_n=5):
    if movie_title not in similarity_df:
        return "Movie not found in dataset!"
    scores = similarity_df[movie_title].sort_values(ascending=False)
    recommended = scores.iloc[1:top_n+1].index.tolist()  # skip same movie
    return recommended

# Example: Recommend for "Toy Story (1995)"
print("\nRecommended Movies for Toy Story (1995):", recommend_movies("Toy Story (1995)"))

# Step 6: Visualization (first 20 movies for clarity)
plt.figure(figsize=(10,8))
sns.heatmap(similarity_df.iloc[:20, :20], cmap="coolwarm")
plt.title("Movie Similarity Matrix (first 20 movies)")
plt.show()

