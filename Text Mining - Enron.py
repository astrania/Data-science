'''
This text mining project focuses on analyzing Enron corporation's raw executive email data set and accomplishes:

- Identify email contents’ similarity, and highlight mostly discussed topics.
- Cluster executives by email contents, and analyze leadership level relations and hierarchy based on topics discussed.
- Determine emails’ underlying sentiment evolvement, and link to any potential association to bankruptcy.


'''

import os, email, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import print_function
import nltk
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

###############
## Cleaning
###############
# Read the data into a DataFrame
os.chdir('/Desktop/MSA/Text Mining')
emails_df = pd.read_csv('emails.csv')
print(emails_df.shape)
emails_df.head()

# A single message looks like this
print(emails_df['message'][0])

# filter out only those that contain 'inbox' in the file name
emails_df = emails_df[emails_df['file'].str.contains('inbox')]
emails_df.head()

# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)

# Get fields(keys) from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
print(keys)

# Parse content from emails
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload().lower())
    return ''.join(parts)
emails_df['content'] = list(map(get_text_from_email, messages))

# Split multiple email addresses
def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs
emails_df['From'] = emails_df['From'].map(split_email_addresses)
emails_df['To'] = emails_df['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
emails_df['user'] = emails_df['file'].map(lambda x: x.split('/')[0])

# Set index to be user and drop meaningless columns with too few values
emails_df = emails_df.set_index('user')\
    .drop(['file', 'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding']
    , axis=1)

# Parse datetime
emails_df['Date'] = pd.to_datetime(emails_df['Date'], 
infer_datetime_format=True)

# remove special characters in the content column
emails_df['content'] = emails_df['content'].apply(lambda x: 
    re.sub('[^a-zA-Z0-9-_*.]', ' ', x))
emails_df['content']

emails_df.head()
####################################
##tokenization, stemming, stop words removal
####################################  
# Romove stop words
#define stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('enron', 'image', 'subject', 're', 'ect', 'hou', 'dear', 
                        'corp','let','know', 'new york', 'please', 'e-mail'))
all_stopwords = default_stopwords | custom_stopwords

#define a function that combines tokenization and stemming
stemmer = nltk.stem.porter.PorterStemmer()

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is 
    # caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in 
    nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

###################
####### TF-IDF
###################
#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2, max_features=50,
                                   stop_words=list(all_stopwords), 
                                   tokenizer=tokenize_and_stem, 
                                   use_idf=True, ngram_range=(1,3))

# Learn the text and return term-document matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(emails_df['content'][:200])
print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
print(terms)

# view results
pd.DataFrame(tfidf_matrix.toarray(), index = emails_df['content'][:200], 
columns = terms).head(10)

###################
###### LSA
###################
lsa = TruncatedSVD(2, algorithm = 'arpack')
tfidf_matrix_lsa = lsa.fit_transform(tfidf_matrix)
tfidf_matrix_lsa_norm = Normalizer(copy=False).fit_transform(tfidf_matrix_lsa)

# view results
pd.DataFrame(lsa.components_, index = ["component_1", "component_2"], 
    columns = terms)
    
pd.DataFrame(tfidf_matrix_lsa_norm, index = emails_df['content'][:200], 
columns = ["component_1", "component_2"])

# PCA scatter plot
%pylab inline
import matplotlib.pyplot as plt

xs = [w[0] for w in tfidf_matrix_lsa_norm]
ys = [w[1] for w in tfidf_matrix_lsa_norm]
figure()
plt.scatter(xs, ys)
xlabel('First PC')
ylabel('Second PC')
title('Plot of points against LSA PCs(scatter)')
show()

# PCA vector geometric plot
plt.figure()
ax = plt.gca()
ax.quiver(0, 0, xs, ys, angles = 'xy', scale_units = 'xy', scale = 1, 
linewidth = 0.01)
ax.set_xlim([-1, 1])
ax.set_ylim([-1,1])
xlabel('First PC')
ylabel('Second PC')
title('Plot of points against LSA PCs(geometric)')
plt.draw()
plt.show()

# document silimarity using lsa
dist = 1 - cosine_similarity(tfidf_matrix_lsa)
print(np.amax(dist[dist<0.9]))

###################    
######## K-Means clustering on words similarity
###################
num_clusters = 5
km = KMeans(n_clusters = num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :3]:
        print(' %s' % terms[ind], end='')
    print()
print()
print()

###################    
######## K Means execs clustering based on words similarity
###################
#group execs by email contents
exec_df = emails_df.groupby('user')['content'].apply(lambda x: x.sum())
print(exec_df.head())

# apply tfidf                                   
exec_tfidf_matrix = tfidf_vectorizer.fit_transform(exec_df)

exec_terms = tfidf_vectorizer.get_feature_names()
print(exec_terms)

exec_names = exec_df.index.tolist()

# view results
pd.DataFrame(exec_tfidf_matrix.toarray(), index = exec_names[], 
columns = exec_terms).head(10)
 
# kmeans clustering               
num_clusters = 5
km = KMeans(n_clusters = num_clusters)
km.fit(exec_tfidf_matrix)
clusters = km.labels_.tolist()

# view results
print("Execs clustering on email content similarity:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print()
    print("Cluster %d:" % i, end='')
    print()
    for ind in order_centroids[i, :6]: #replace 5 with 3 words per cluster
        print(' %s  %s' % (exec_names[ind], exec_terms[ind]), end='')
        print()
print()
print()

############################ 
####### Hierarchical clustering
############################   
dist_exec = 1 - cosine_similarity(exec_tfidf_matrix)

linkage_matrix = ward(dist_exec) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=exec_names);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
#plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters                 