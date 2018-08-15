import os

import sys
# print(sys.path)
# path = "C:/Users/anreed/Documents/Initiatives/Context Edge/ContextEdge Code/ctxe-pylib"
# sys.path.append(path)
import ContextEdge as ctxe

"""
-----------------------------------------------------------
                    PDF Extraction
-----------------------------------------------------------
"""
# PDF text extractions
# print ("Writing text from PDFs to .txt files...")
# location of the inputs / pdf files
file_dir = r'C:\Users\anreed\Documents\Initiatives\Context Edge\OPM Analysis\pdfs'
file_paths = os.listdir(file_dir)  #creates list of pdf filenames
# where the output .txt files will go
output_path = r'C:\Users\anreed\Documents\Initiatives\Context Edge\OPM Analysis\opm_txt_files'
file_paths

for filename in file_paths:
    textfile_name = filename.rsplit('.')[0] + '_py.txt'   #get new textfile name
    filepath = os.path.join(output_path, textfile_name)
    print (filepath)
    with open(filepath, 'wb') as f:
        # uses contextedge extractor module to get text from pdf
        pdf_text = ctxe.extractors.pdf2txt(os.path.join(file_dir, filename)).encode('utf-8')
        # print (pdf_text)
        f.write(pdf_text)
    print ("Extracted text from '{0}' to '{1}'.".format(filename, textfile_name))

# print ("Finished extracting text from PDFs")


"""
-----------------------------------------------------------
                        Processing
-----------------------------------------------------------
"""
# Now that context is extracted from pdfs, we can preprocess and analyze
# using additional modules of ContextEdge

# Create corpus from newly extract .txt files
pdf_docs = ctxe.universe.Corpus()
# read documents into corpus via directory path
# replace with your personal path to the folder that holds the txt documents
pdf_docs.ingestion(directory=output_path)
len(pdf_docs.corpus)
pdf_docs.corpus
doc1 = pdf_docs.corpus[0]
doc7 = pdf_docs.corpus[1]
doc7.text

pdf_docs.corpus[:]

testing_doc = pdf_docs.corpus[0]

testing_corp.top_n()

#Build Corpus of 25 text files for comparison
output_path_25 = r'C:\Users\anreed\Documents\Initiatives\Context Edge\OPM Analysis\pdfs_25'
corp25 = ctxe.universe.Corpus()
corp25.ingestion(directory=output_path_25)

"""
---------------------------------------------------------------
                        Frequency Analysis
---------------------------------------------------------------
"""
import string
import imp
import preprocessor
imp.reload(preprocessor)
from preprocessor import preprocessor
from sklearn.metrics.pairwise import cosine_similarity
from ctxe.preprocessor

pdf_docs.top_n()
13, 17

words = ['a', '5', 'like', 'to', '8888', '&']
words = [word for word in words if len(word)>1]
words

len('g')

preprocessor(corp25.corpus[17].text)
corp25.corpus[17].tokens
corp25.frequency_analysis()


"""
---------------------------------------------------------------
                        Document Similarity
---------------------------------------------------------------
"""

#Full Corpus
fullvec = pdf_docs.vectorize(vec_options=vec_dict)
dsm = cosine_similarity(pdf_docs.vec_matrix)
dsm.shape

fullvec.get_feature_names()

dsm = pdf_docs.document_similarity()
pdf_docs.plot_dsm(dsm)

dsm.shape

dsm


#Corpus 25

##custom vectorizer
vec_dict = dict(stop_words='english', max_df=0.8, ngram_range=(1, 3), token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
vec = corp25.vectorize(vec_options=vec_dict)
vec
corp25.vector
corp25.vec_matrix
vec.get_feature_names()

##custom dsm
#dsm25 = corp25.document_similarity()

dsm25 = cosine_similarity(corp25.vec_matrix)
dsm25.shape
dsm25


"""Visualize cosine similarity matrix"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

###Full Corpus
# Generate a mask for the upper triangle
mask = np.zeros_like(dsm25, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(20, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
plt.rcParams['figure.figsize']=(15,15)
cos_sim_plot = sns.heatmap(dsm25, mask=mask, cmap=cmap, center=0,
            square=True, xticklabels=1,yticklabels =1)
cos_sim_plot_fig = cos_sim_plot.get_figure()

#write to png
#cos_sim_plot_fig.savefig("cosinesimilarity_heatmap.png")

#--------------------------------------------------------------------
"""List out doc-doc pairings sorted by strongest cosine similarity score"""

def get_most_similar_docs(dsm):
    dsm = np.tril(dsm, k=0)
    row_number = []
    column_number = []
    score = []

    for index, x in np.ndenumerate(dsm):
        row_number.append(index[0])
        column_number.append(index[1])
        score.append(round(x,5))

    similarity_scores_df = pd.DataFrame(
        {'doc_1': row_number,
         'doc_2': column_number,
         'similarity_score': score
        })

    similarity_scores_df = similarity_scores_df[similarity_scores_df['similarity_score'].between(0,1, inclusive = False)]

    return similarity_scores_df.sort_values(by = 'similarity_score', ascending = False).head(25)

get_most_similar_docs(dsm25)
get_most_similar_docs(dsm)

cos_sim_full_list = get_most_similar_docs(dsm)
cos_sim_full_list.to_csv('ALL_cos_sim_full_list', encoding='utf-8', index=False)


#---------------------------------------------------------------------------------------------------
""" function to get main features for a document from TFIDF Vectorizer"""
from sklearn.feature_extraction.text import TfidfVectorizer

fullvec = pdf_docs.vectorize(vec_options=vec_dict)                         #need to streamline the intake of a tfidf vector (Sparse) matrix

def get_main_features(ctxe_corpus, doc_index):
    #this function takes in a corpus a desired document number and outputs the top 50 most important features by tfidf weighting
    tfidf_matrix = ctxe_corpus.vec_matrix                                  #need to improve line of inhertance here
    feature_names = fullvec.get_feature_names()                            #need to improve line of inhertance here

    doc_num = doc_index
    feature_index = tfidf_matrix[doc_num,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc_num, x] for x in feature_index])

    doc_features = []
    doc_feature_scores = []

    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        doc_features.append(w)
        doc_feature_scores.append(s)

    doc_main_features_df = pd.DataFrame(
        {'feature': doc_features,
         'tfidf_weighting': doc_feature_scores
        })

    return doc_main_features_df.sort_values(by = 'tfidf_weighting', ascending = False).head(25)


### Highest cosine similarity score: NURSING JOBS
#Get main features from TFIDF matrix for Doc GS-0620 (index of 78)
get_main_features(pdf_docs, 78).to_csv('features_0620.csv', encoding='utf-8', index=False)
#Get main features from TFIDF matrix for Doc GS-0621 (index of 79)
get_main_features(pdf_docs, 79).to_csv('features_0621.csv', encoding='utf-8', index=False)

### Second highest cosine similarity score:
#Get main features from TFIDF matrix for Doc GS-0260 (index of 40)
get_main_features(pdf_docs, 40).to_csv('features_0260.csv', encoding='utf-8', index=False)
#Get main features from TFIDF matrix for Doc GS-036 (index of 64)
get_main_features(pdf_docs, 64).to_csv('features_036.csv', encoding='utf-8', index=False)

### Third highest cosine similarity score:
#Get main features from TFIDF matrix for Doc GS-1104 (index of 127)
get_main_features(pdf_docs, 127).to_csv('features_1104.csv', encoding='utf-8', index=False)
#Get main features from TFIDF matrix for Doc GS-1107 (index of 130)
get_main_features(pdf_docs, 130).to_csv('features_1107.csv', encoding='utf-8', index=False)

get_main_features(pdf_docs, 115)
len(doc78_features)
len(doc78_feature_scores)

doc78
tfidf_matrix =  tf.fit_transform(corpus)


pdf_docs.corpus[24].text
pdf_docs.corpus[42].text
pdf_docs.corpus[44].text
pdf_docs.corpus[67].text
pdf_docs.corpus[211].text
pdf_docs.corpus[182].text

#Get main features from TFIDF of Doc GS-0621 (index of 79)
pdf_docs.corpus[40].text

"""
---------------------------------------------------------------
                        K-Means Clustering
---------------------------------------------------------------
"""
from sklearn.cluster import KMeans
from __future__ import print_function

#define distance metrics
dist = 1 - dsm
dist

#define vector matrix
fullvec_matrix = pdf_docs.vec_matrix


num_clusters = 23
km = KMeans(n_clusters=num_clusters)

%time km.fit(fullvec_matrix)

#Create document index to use at document title
doc_index = []
a=0
for i in pdf_docs.corpus:
    doc_index.append(a)
    a +=1


clusters = km.labels_.tolist()
all_tokens = fullvec.get_feature_names()
vocab_frame = pd.DataFrame({'words': all_tokens}, index = all_tokens)

films = { 'doc_index': doc_index, 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters] , columns = ['doc_index', 'cluster'])


print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[all_tokens[ind].split(',')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace

    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['doc_index'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

print()
print()
