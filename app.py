import os from sklearn.feature_extraction.text
import TfidfVectorizer from sklearn.metrics.pairwise
import cosine_similariity

student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file ,encoding='utf-8').read()]
