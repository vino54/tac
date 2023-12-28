# Projet final TP4


## Exploration de l'export CAMILLE
# Imports
import os
import textract
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
# Compter le nombre de documents dans le corpus
path = "../data/camille_Seconde_Guerre_mondiale/"
files = sorted(os.listdir(path))
len(files)
# Compter le nombre de journaux par organe de presse dans l'intervalle qui nous interesse
all_years = [str(year) for year in range(1939, 1945)]
# Compter le nombre de journaux par organe de presse
print(f"Il y a {count_newspapers['JB838']} exemplaires du journal Le Soir et {count_newspapers['JB427']} exemplaires de Le Drapeau rouge")
# Visualisation du nombre de document par mois
index = np.arange(len(count_month))
plt.bar(index, count_month.values())
plt.xlabel('Mois')
plt.ylabel('# documents')
plt.xticks(index, count_month.keys(), fontsize=8, rotation=30)
plt.title('TP4 - Nombre de documents par mois')
plt.show()
## Créer un grand fichier 'corpus': commande bash
data_path = '../data/camille_Seconde_Guerre_mondiale'
txt_path = '../data/camille_Seconde_Guerre_mondiale'
with open("../data/camille_Seconde_Guerre_mondiale.txt", "w", encoding="utf-8") as output_file:
    for file in os.listdir(txt_path):
        if file.endswith(".txt"):
            with open(os.path.join(txt_path, file), "r", encoding="utf-8") as f:
                output_file.write(f.read())

# Compter le nombre de mots dans l'ensemble du corpus
!wc ../data/camille_Seconde_Guerre_mondiale.txt

## Analyse de la distribution du vocabulaire
# Imports et dépendances
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# Créer une une liste de stopwords
sw = stopwords.words("french")
sw += ["les", "plus", "cette", "fait", "faire", "être", "deux", "comme", "dont", "tout", 
       "ils", "bien", "sans", "peut", "tous", "après", "ainsi", "donc", "cet", "sous",
       "celle", "entre", "encore", "toutes", "pendant", "moins", "dire", "cela", "non",
       "faut", "trois", "aussi", "dit", "avoir", "doit", "contre", "depuis", "autres",
       "van", "het", "autre", "jusqu", "ceux", "toute", "tel", "ni", "ou", "avoir", 
       "aussi", "pour", "de", "le", "la", "en", "une", "un", "votre", "notre", "leur",
       "trop", "vers", "peu", "ici", "leurs", "pres", "car", "tres", "des", "je"]
sw = set(sw)
print(f"{len(sw)} stopwords:\n {sorted(sw)}")
##### Tokenization
# Récupération du contenu du fichier
path = "../data/camille_Seconde_Guerre_mondiale.txt"
limit = 10**8

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()[:limit]
# Tokenization
words = nltk.wordpunct_tokenize(text)
print(f"{len(words)} words found")
##### Calculer la taille du vocabulaire
# Eliminer les stopwords et les termes non alphabétiques
kept = [w.lower() for w in words if len(w) > 2 and w.isalpha() and w.lower() not in sw]
voc = set(kept)
print(f"{len(kept)} words kept ({len(voc)} different word forms)")
### Récupérer les mots les plus fréquents et en faire un plot
fdist = nltk.FreqDist(kept)
fdist.most_common(10)
# Plot: les n mots les plus fréquents
n = 10
fdist.plot(n, cumulative=False)
### Nuages de mots / Wordcloud
from collections import Counter
from wordcloud import WordCloud
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from IPython.display import Image
# Stopwords
sw = stopwords.words("french")
sw += ["les", "plus", "cette", "fait", "faire", "être", "deux", "comme", "dont", "tout",
       "ils", "bien", "sans", "peut", "tous", "après", "ainsi", "donc", "cet", "sous",
       "celle", "entre", "encore", "toutes", "pendant", "moins", "dire", "cela", "non",
       "faut", "trois", "aussi", "dit", "avoir", "doit", "contre", "depuis", "autres",
       "van", "het", "autre", "jusqu", "ville", "rossel", "dem", "les", "plus", "cette", 
       "fait", "faire", "être", "deux", "ceux", "toute", "tel", "ni", "ou",  "jamais", 
       "aussi", "pour", "de", "le", "la", "en", "une", "un", "votre", "notre", "leur",
       "trop", "vers", "peu", "ici", "leurs", "pres", "car", "tres", "des", "je"]
sw = set(sw)

##### Créer un fichier contenant le texte de tous les journaux d'une année donnée
# Choisir une année
year = 1945
# Lister les fichiers de cette année
data_path = '../data/camille_Seconde_Guerre_mondiale'
txt_path = '../data//camille_Seconde_Guerre_mondiale'
txts = [f for f in os.listdir(txt_path) if os.path.isfile(os.path.join(txt_path, f)) and str(year) in f]
len(txts)
# Stocker le contenu de ces fichiers dans une liste
content_list = []
for txt in txts:
    with open(os.path.join(txt_path, txt), 'r', encoding='utf-8') as f:
        content_list.append(f.read())
# Ecrire tout le contenu dans un fichier temporaire
temp_path = '../data/tmp'
if not os.path.exists(temp_path):
    os.mkdir(temp_path)
with open(os.path.join(temp_path, f'{year}.txt'), 'w', encoding='utf-8') as f:
    f.write(' '.join(content_list))
# Imprimer le contenu du fichier et constater les "déchets"
with open(os.path.join(temp_path, f'{year}.txt'), 'r', encoding='utf-8') as f:
    before = f.read()

before[:500]
##### Nettoyer le fichier à l'aide d'une fonction de nettoyage
### Créer la fonction de nettoyage (à adapter)
def clean_text(year, folder=None):
    if folder is None:
        input_path = f"{year}.txt"
        output_path = f"{year}_clean.txt"
    else:
        input_path = f"{folder}/{year}.txt"
        output_path = f"{folder}/{year}_clean.txt"
    output = open(output_path, "w", encoding='utf-8')
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        words = nltk.wordpunct_tokenize(text)
        kept = [w.upper() for w in words if len(w) > 2 and w.isalpha() and w.lower() not in sw]
        kept_string = " ".join(kept)
        output.write(kept_string)
    return f'Output has been written in {output_path}!'
# Appliquer la fonction sur le fichier complet de l'année
clean_text(year, folder=temp_path)
# Vérifier le résultat
with open(os.path.join(temp_path, f'{year}_clean.txt'), 'r', encoding='utf-8') as f:
    after = f.read()

after[:500]
# Afficher les termes les plus fréquents
frequencies = Counter(after.split())
print(frequencies.most_common(10))
### Créer, stocker et afficher le nuage de mots
cloud = WordCloud(width=2000, height=1000, background_color='white').generate_from_frequencies(frequencies)
cloud.to_file(os.path.join(temp_path, f"{year}.png"))
Image(filename=os.path.join(temp_path, f"{year}.png"))
### Extraction des mots clés d'un document avec Yake
import os
import yake
# Instantier l'extracteur de mots clés
kw_extractor = yake.KeywordExtractor(lan="fr", top=50)
kw_extractor
# Lister les Fichiers
temp_path = '../data/tmp'
files = os.listdir(temp_path)
# Choisir le fichier de l'annéee choisie (1945)
this_file = files[1]
this_file
# Récupérer le texte du fichier de l'annéee choisie (1945)
text = open(os.path.join(temp_path, this_file), 'r', encoding='utf-8').read()
text[:500]
# Extraire les mots clés de ce texte
keywords = kw_extractor.extract_keywords(text)
keywords
print(keywords[:20])
### recherches d'associations de mots
import re

# Define the path to the corpus of text
path = "../data/camille_Seconde_Guerre_mondiale.txt"

# Open the file and read the content into a variable
with open(path, "r", encoding='utf-8') as file:
    corpus = file.read()

# Define the words you want to search for
word1 = "régime"
word2 = "Guerre"

# Use a regular expression to search for the words
pattern = r"\b" + word1 + r"\W+" + word2 + r"\b"
match = re.search(pattern, corpus)

# Print the result
if match:
    print("The words '" + word1 + "' and '" + word2 + "' were found in the corpus")
else:
    print("The words '" + word1 + "' and '" + word2 + "' were not found in the corpus")

# Define the path to the corpus of text
path = "../data/camille_Seconde_Guerre_mondiale.txt"

# Open the file and read the content into a variable
with open(path, "r", encoding='utf-8') as file:
    corpus = file.read()

# Define the words you want to search for
word1 = "angleterre"
word2 = "mort"

# Use a regular expression to search for the words
pattern = r"\b" + word1 + r"\W+" + word2 + r"\b"
match = re.search(pattern, corpus)

# Print the result
if match:
    print("The words '" + word1 + "' and '" + word2 + "' were found in the corpus")
else:
    print("The words '" + word1 + "' and '" + word2 + "' were not found in the corpus")

# Define the path to the corpus of text
path = "../data/camille_Seconde_Guerre_mondiale.txt"

# Open the file and read the content into a variable
with open(path, "r", encoding='utf-8') as file:
    corpus = file.read()

# Define the words you want to search for
word1 = "guerre"
word2 = "allemands"

# Use a regular expression to search for the words
pattern = r"\b" + word1 + r"\W+" + word2 + r"\b"
match = re.search(pattern, corpus)

# Print the result
if match:
    print("The words '" + word1 + "' and '" + word2 + "' were found in the corpus")
else:
    print("The words '" + word1 + "' and '" + word2 + "' were not found in the corpus")

# Define the path to the corpus of text
path = "../data/camille_Seconde_Guerre_mondiale.txt"

# Open the file and read the content into a variable
with open(path, "r", encoding='utf-8') as file:
    corpus = file.read()

# Define the words you want to search for
word1 = "guerre"
word2 = "allemands"

# Use a regular expression to search for the words
pattern = r"\b" + word1 + r"\W+" + word2 + r"\b"
match = re.search(pattern, corpus)

# Print the result
if match:
    print("The words '" + word1 + "' and '" + word2 + "' were found in the corpus")
else:
    print("The words '" + word1 + "' and '" + word2 + "' were not found in the corpus")
##### Segmentation en phrases (tokenization) 
# Fichiers d'inputs et d'outputs
infile = "../data/camille_Seconde_Guerre_mondiale.txt"
outfile = "../data/sents.txt"
# Segmentation en phrases du corpus complet et création d'un nouveau fichier
LIMIT = None
from nltk.tokenize import sent_tokenize  # Make sure to import sent_tokenize from nltk

LIMIT = 100  # Define LIMIT or adjust as needed

with open(outfile, 'w', encoding="utf-8") as output:
    with open(infile, encoding="utf-8", errors="backslashreplace") as f:
        content = f.readlines()
        content = content[:LIMIT] if LIMIT is not None else content
        n_lines = len(content)
        for i, line in enumerate(content):
            if i % 100 == 0:
                print(f'processing line {i}/{n_lines}')
            sentences = sent_tokenize(line)
            for sent in sentences:
                output.write(sent + "\n")
print("Done")

## Clustering de documents
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
from sklearn.datasets import make_blobs
import numpy as np

import nltk

nltk.download('punkt')
data_path = "../data/camille_Seconde_Guerre_mondiale/"
# Choisir une décennie
DECADE = '1945'
# Charger tous les fichiers de la décennie et en créer une liste de textes
files = [f for f in sorted(os.listdir(data_path)) if f"_{DECADE[:-1]}" in f]
# Exemple de fichiers
files[:10]
texts = [open(data_path + f, "r", encoding="utf-8").read() for f in files]
# Exemple de textes
texts[0][:500]
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
##### Vectoriser les documents à l'aide de TF-IDF
# Création d'une fonction de pré-traitement
def preprocessing(text, stem=True):
    """ Tokenize text and remove punctuation """
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text)
    return tokens
# Instancier le modèle TF-IDF avec ses arguments
vectorizer = TfidfVectorizer(
    tokenizer=preprocessing,
    stop_words=stopwords.words('french'),
    max_df=0.5,
    min_df=0.1,
    lowercase=True)
# Construire la matrice de vecteurs à l'aide de la fonction `fit_transform`
tfidf_vectors = vectorizer.fit_transform(texts)
# Détail de la matrice
tfidf_vectors
# Imprimer le vecteur tf-IDF du premier documentpd.Series
pd.Series(
    tfidf_vectors[0].toarray()[0],
    index=vectorizer.get_feature_names_out()
    ).sort_values(ascending=False)
##### Appliquer un algorithme de clustering sur les vecteurs TF-IDF des documents
# Trouver le nombre optimal de clusters (n_clusters) avec la méthode Elbow
X, y = make_blobs(n_samples=1000, n_features=2,random_state=0)
# Within-cluster Sum of Square (WSS)- somme des carrés intra-cluster
WSS = []
# Pour les valeur possible de 'k'
K = range(2, 10)
for n in K:
    algorithm = (KMeans(n_clusters = n) )
    algorithm.fit(X)
    WSS.append(algorithm.inertia_)
fig, (ax1) = plt.subplots(ncols =1)
# fig, (ax1) = plt.subplots(ncols =1)
fig.set_figheight(10)
fig.set_figwidth(20)

ax1.plot(K, WSS, 'bo')
ax1.plot(K, WSS, 'r-', alpha = 1)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Sum_of_squared_distances')
ax1.set_title('TP4 - Elbow Method For Optimal k')
ax1.grid(True)

ax1.axvline(3, color='#F26457', linestyle=':')
# Définir un nombre de clusters
N_CLUSTERS = 3
# Instancier le modèle K-Means et ses arguments
km_model = KMeans(n_clusters=N_CLUSTERS)
# Appliquer le clustering à l'aide de la fonction `fit_predict`
clusters = km_model.fit_predict(tfidf_vectors)
clustering = collections.defaultdict(list)

for idx, label in enumerate(clusters):
    clustering[label].append(files[idx])
pprint(dict(clustering))
##### Visualiser les clusters
# Réduire les vecteurs à 2 dimensions à l'aide de l'algorithme PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(tfidf_vectors.toarray())
reduced_vectors[:10]
# Générer le plot
x_axis = reduced_vectors[:, 0]
y_axis = reduced_vectors[:, 1]

plt.figure(figsize=(10,10))
scatter = plt.scatter(x_axis, y_axis, s=100, c=clusters)

# Ajouter les centroïdes
centroids = pca.transform(km_model.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1],  marker = "x", s=100, linewidths = 2, color='black')

# Ajouter la légende
plt.legend(handles=scatter.legend_elements()[0], labels=set(clusters), title="Clusters")
## Sentiment analysis avec Textblob-FR
import sys
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
## Fonction
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

def get_sentiment(input_text):
    blob = tb(input_text)
    polarity, subjectivity = blob.sentiment
    polarity_perc = f"{100*abs(polarity):.0f}"
    subjectivity_perc = f"{100*subjectivity:.0f}"
    if polarity > 0:
        polarity_str = f"{polarity_perc}% positive"
    elif polarity < 0:
        polarity_str = f"{polarity_perc}% negative"
    else:
        polarity_str = "neutral"
    if subjectivity > 0:
        subjectivity_str = f"{subjectivity}% subjective"
    else:
        subjectivity_str = "perfectly objective"
    print(f"This text is {polarity_str} and {subjectivity_str}.")
##### Analyser le sentiment d'une phrase
get_sentiment("La Seconde Guerre mondiale propulse les États-Unis et l'URSS.")
get_sentiment("les bombardements aériens massifs de civils d'abord par l'Axe en Europe.") 
get_sentiment("Cette guerre oppose de 1939 à 1945 les forces de l'Axe (IIIe Reich allemand, Italie, Japon) aux Alliés (France, Grande-Bretagne, URSS, États-Unis).")
get_sentiment("Le 6 juin 1944, les forces alliées ont lancé le débarquement en Normandie.") 
get_sentiment("La guerre a débuté le 1er septembre 1939, lorsque l'Allemagne a envahi la Pologne")  
get_sentiment("Les causes de la Seconde Guerre mondiale sont multiples, mais les principales comprennent les conséquences du traité de Versailles après la Première Guerre mondiale, l'expansionnisme agressif de l'Allemagne nazie sous le régime d'Adolf Hitler, l'invasion de la Pologne en 1939 et les alliances entre les grandes puissances..")
get_sentiment("En réponse, la France et le Royaume-Uni ont déclaré la guerre à l'Allemagne.")
get_sentiment("En parallèle, les Alliés ont lancé des opérations de débarquement en Afrique du Nord (Opération Torch) et en Italie.")
get_sentiment("Le début de la Seconde Guerre mondiale est marqué par un mélange de peur et d'incertitude, alors que le monde plonge dans l'obscurité de la violence.")
get_sentiment("L'expansionnisme des puissances de l'Axe")
get_sentiment("Le 3 septembre 1939, suite à l'agression de la Pologne, la Grande-Bretagne puis la France déclarent la guerre à l'Allemagne. Les hommes répondent sans joie mais avec détermination à l'ordre de mobilisation.")
## Word Embeddings : le modèle Word2Vec
import sys

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

import nltk
from nltk.tokenize import wordpunct_tokenize
from unidecode import unidecode
#### Chargement et traitement des phrases du corpus
# Création d'un objet qui *streame* les lignes d'un fichier pour économiser de la RAM
class MySentences(object):
    """Tokenize and Lemmatize sentences"""
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, encoding='utf-8', errors="backslashreplace"):
            yield [unidecode(w.lower()) for w in wordpunct_tokenize(line)]
infile = f"../data/camille_Seconde_Guerre_mondiale.txt"
sentences = MySentences(infile)
# Détection des bigrams
bigram_phrases = Phrases(sentences)
type(bigram_phrases.vocab)
len(bigram_phrases.vocab.keys())
# Prenons une clé au hasard
key_ = list(bigram_phrases.vocab.keys())[150]
print(key_)
# Le dictionnaire indique le score de cette coocurrence
bigram_phrases.vocab[key_]
# Conversion des `Phrases` en objet `Phraser`
bigram_phraser = Phraser(phrases_model=bigram_phrases)
# Extraction des trigrams
trigram_phrases = Phrases(bigram_phraser[sentences])
trigram_phraser = Phraser(phrases_model=trigram_phrases)
# Création d'un corpus d'unigrams, bigrams, trigrams
corpus = list(trigram_phraser[bigram_phraser[sentences]])
print(corpus[:10])
#### Entrainement d'un modèle Word2Vec sur ce corpus
%%time
model = Word2Vec(
    corpus, # On passe le corpus de ngrams que nous venons de créer
    vector_size=32, # Le nombre de dimensions dans lesquelles le contexte des mots devra être réduit, aka. vector_size
    window=2, # La taille du "contexte", ici 2 mots avant et après le mot observé
    min_count=10, # On ignore les mots qui n'apparaissent pas au moins 10 fois dans le corpus
    workers=2, # Pas de parallelisation
    epochs=1 # Nombre d'itérations du réseau de neurones sur le jeu de données pour ajuster les paramètres avec la descente de gradient, aka. epochs.
)
# Sauver le modèle dans un fichier
outfile = f"../data/newspapers.model"
model.save(outfile)
##### Explorer le modèle
# Charger le modèle en mémoire
model = Word2Vec.load("../data/newspapers.model")
# Imprimer le vecteur d'un terme
model.wv["allemands"]
# Calculer la similarité entre deux termes: Exemple 1
model.wv.similarity("guerre", "conflit")
# Calculer la similarité entre deux termes: Exemple 2
model.wv.similarity("mondiale", "arme")
# Calculer la similarité entre deux termes: Exemple 3
model.wv.similarity("mondiale", "mondiale")
# Chercher les mots les plus proches d'un terme donné: Exemple 1
model.wv.most_similar("guerre_mondiale", topn=4)
# Chercher les mots les plus proches d'un terme donné: Exemple 2
model.wv.most_similar("seconde", topn=4)
# Faire des recherches complexes à travers l'espace vectoriel : Exemple 1
print(model.wv.most_similar(positive=['seconde', 'mondiale'], negative=['guerre'], topn=1))
# Faire des recherches complexes à travers l'espace vectoriel : Exemple 2
print(model.wv.most_similar(positive=['territoire', 'lituanien'], negative=['violation'], topn=1))
##### end