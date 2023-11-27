# Clustering de documents
## Imports
import collections
import os
import string
import sys

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import nltk

nltk.download('punkt')
# Clustering de documents
## Imports
import collections
import os
import string
import sys

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import nltk

nltk.download('punkt')
data_path = "../data/Rivalités_et_tensions_européennes/"
## Choisir une décennie
DECADE = '1890'
## Charger tous les  fichiers de la décennie et en créer une liste de textes
files = [f for f in sorted(os.listdir(data_path)) if f"_{DECADE[:-1]}" in f]
# Exemple de fichiers
files[:5]
texts = [open(data_path + f).read() for f in files]
# Exemple de textes
texts[0][:400]
## Vectoriser les documents à l'aide de TF-IDF
# Création d'une fonction de pré-traitement
def preprocessing(text, stem=True):
    """ Tokenize text and remove punctuation """
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text)
    return tokens
### Instancier le modèle TF-IDF avec ses arguments
vectorizer = TfidfVectorizer(
    tokenizer=preprocessing,
    stop_words=stopwords.words('french'),
    max_df=0.5,
    min_df=0.1,
    lowercase=True)
### Construire la matrice de vecteurs à l'aide de la fonction `fit_transform`
tfidf_vectors = vectorizer.fit_transform(texts)
# Détail de la matrice
tfidf_vectors
### Imprimer le vecteur tf-IDF du premier document
pd.Series(
    tfidf_vectors[0].toarray()[0],
    index=vectorizer.get_feature_names_out()
    ).sort_values(ascending=False)
## Comprendre les vecteurs et leurs "distances"
cosine([1, 2, 3], [1, 2, 3])
cosine([1, 2, 3], [1, 2, 2])
cosine([1, 2, 3], [2, 2, 2])
### Tests sur nos documents
tfidf_array = tfidf_vectors.toarray()
# Vecteur du document 0
tfidf_array[0]
# Vecteur du document 1
tfidf_array[1]
cosine(tfidf_array[0], tfidf_array[1])
## Appliquer un algorithme de clustering sur les vecteurs TF-IDF des documents
Pour en savoir plus sur le KMeans clustering :
- https://medium.com/dataseries/k-means-clustering-explained-visually-in-5-minutes-b900cc69d175
### Définir un nombre de clusters
N_CLUSTERS = 5
### Instancier le modèle K-Means et ses arguments
km_model = KMeans(n_clusters=N_CLUSTERS)
### Appliquer le clustering à l'aide de la fonction `fit_predict`
clusters = km_model.fit_predict(tfidf_vectors)
clustering = collections.defaultdict(list)

for idx, label in enumerate(clusters):
    clustering[label].append(files[idx])
pprint(dict(clustering))
## Visualiser les clusters
### Réduire les vecteurs à 2 dimensions à l'aide de l'algorithme PCA
Cette étape est nécessaire afin de visualiser les documents dans un espace 2D

https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(tfidf_vectors.toarray())
reduced_vectors[:10]
### Générer le plot
x_axis = reduced_vectors[:, 0]
y_axis = reduced_vectors[:, 1]

plt.figure(figsize=(10,10))
scatter = plt.scatter(x_axis, y_axis, s=100, c=clusters)

# Ajouter les centroïdes
centroids = pca.transform(km_model.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1],  marker = "x", s=100, linewidths = 2, color='black')

# Ajouter la légende
plt.legend(handles=scatter.legend_elements()[0], labels=set(clusters), title="Clusters")