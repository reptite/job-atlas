# this script trains and pickels a tfidf vectorizer and a umap reducer
# it exports embeddings and models for the app to use later
# run if rebuilding from scratch - python ./train_model/train.py

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import pickle as pkl
import ssl

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

od_data = pd.read_excel(
    'https://www.nationalskillscommission.gov.au/sites/default/files/2021-07/Australian%20Skills%20Classification%2012-03-2021.xlsx'
    ,sheet_name='Occupation Descriptions'
    )
data_prep = od_data.copy()

# concat title and jd as input
data_prep['input'] = data_prep['ANZSCO_Title'] + ' ' + data_prep['ANZSCO_Desc']

# Tokenize by words
data_prep['tokens'] = data_prep.input.apply(lambda x: [w for w in word_tokenize(x.lower()) if w.isalpha()])

# Remove stop words
data_prep['noStops'] = data_prep.tokens.apply(lambda x: [w for w in x if w not in stopwords.words('english')])

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize
data_prep['lemmatized'] = data_prep.noStops.apply(lambda x: [wordnet_lemmatizer.lemmatize(w) for w in x])

# And jam it all back together into a single string
data_prep['lemmatizedStr'] = data_prep['lemmatized'].apply(lambda x: ' '.join(x))

# Vectorize
tfidf_vectorizer = TfidfVectorizer()
list_corpus = data_prep.lemmatizedStr.to_list()

# train vectorizer
word_vectors = tfidf_vectorizer.fit_transform(list_corpus)

# train reducer
reducer = umap.UMAP()
word_embeddings = reducer.fit_transform(word_vectors)

# generate data for export
plot_data = pd.DataFrame(data=word_embeddings, columns=['umap_x','umap_y'])
plot_data['job_title'] = data_prep['ANZSCO_Title']
plot_data['job_description'] = data_prep['ANZSCO_Desc']
plot_data['source'] = 'government'
plot_data.head()

# export data and models
plot_data.to_csv('./static/job_embeddings.csv',index=False)

tfidf_vectorizer_fn = './static/tfidf_vectorizer.sav'
reducer_fn = './static/reducer.sav'
pkl.dump(tfidf_vectorizer, open(tfidf_vectorizer_fn,'wb'))
pkl.dump(reducer, open(reducer_fn,'wb'))