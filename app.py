import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similariity

student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(__file ,encoding='utf-8').read() for_file in student_files]

def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(doc1,doc2):
    return  cosine_similariity([doc1,doc2])

    
