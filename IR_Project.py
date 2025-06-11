# ------------------------------------------------------------
# PHASE 1: DATASET DEFINITION, TEXT PREPROCESSING AND INDEX CREATION AND SAVING

# DOWNLOAD DATASET
from sklearn.datasets import fetch_20newsgroups

categories = ['sci.space', 'rec.autos', 'comp.graphics', 'talk.politics.mideast']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data
labels = newsgroups.target
category_names = newsgroups.target_names

# TEXT PREPROCESSING
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text.lower()) 
    words = [word for word in words if word not in stop_words and word.isalnum()]
    return words

# sample_text = documents[0]
# print("Original Text:", sample_text, "\n")
# print("Processed Tokens:", preprocess_text(sample_text))

# INVERTED INDEX CONSTRUCTION
import pandas as pd
from collections import defaultdict

inverted_index = defaultdict(set)
for doc_id, text in enumerate(documents):
    words = preprocess_text(text)
    for word in words:
        inverted_index[word].add(doc_id) 

df_index = pd.DataFrame([(term, list(docs)) for term, docs in inverted_index.items()], columns=["Term", "Docs"])
# print(df_index.head(100))

# SAVING INDEX
import json

with open("inverted_index.json", "w") as f:
    json.dump({k: list(v) for k, v in inverted_index.items()}, f)
    
    
# ------------------------------------------------------------
# PHASE 2: SEARCHING ENGINE IMPLEMENTATION WITH TF-IDF
    
# LOADING INDEX
with open("inverted_index.json", "r") as f:
    inverted_index = json.load(f)

N = len(documents)

# PREPROCESSING QUERY
def preprocess_query(query):
    return preprocess_text(query)

# TF-IDF COMPUTING FOR EACH DOCUMENT
import math
from collections import defaultdict, Counter

def compute_scores(query_terms, documents):
    scores = defaultdict(float)

    for term in query_terms:
        if term in inverted_index:
            doc_list = inverted_index[term]
            # IDF CALCULATING
            df = len(doc_list)
            idf = math.log(N / (1 + df))

            for doc_id in doc_list:
                # TF CALCULATING
                tf = preprocess_text(documents[int(doc_id)]).count(term)
                tf_norm = tf / len(preprocess_text(documents[int(doc_id)]))
                # TF-IDF CALCULATING
                scores[doc_id] += tf_norm * idf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# SEARCHING (QUERYING)
def search(query, top_k=5):
    query_terms = preprocess_query(query)
    scores = compute_scores(query_terms, documents)
    
    print(f"Results for the query: '{query}'\n")
    for i, (doc_id, score) in enumerate(scores[:top_k]):
        print(f"{i+1}. Document {doc_id} (score: {score:.4f})")
        # print(documents[int(doc_id)][:300])
        print("-" * 80)

# SEARCH LAUNCHING - INSERTING STRING IN THE SYSTEM
# search("man")


# ------------------------------------------------------------
# PHASE 3: LTR IMPLEMENTATION

# DEFINING QUERY EXAMPLES FOR LTR
query_examples = {
    "space mission": {84: 1, 1739: 1, 248: 1, 2: 0, 10: 0},
    "computer graphics": {2090: 1, 1786: 1, 1972: 1, 1: 0, 8: 0},
    "middle east politics": {16: 1, 71: 1, 526: 1, 3: 0, 6: 0},
}

# GENERATING TRAINING DATASET
import numpy as np

def extract_features(query, doc_id, documents):
    doc_text = documents[doc_id]
    doc_tokens = preprocess_text(doc_text)
    query_tokens = preprocess_query(query)

    common_terms = len(set(query_tokens).intersection(doc_tokens))
    doc_len = len(doc_tokens)

    tfidf_sum = 0
    for term in query_tokens:
        tf = doc_tokens.count(term) / doc_len if doc_len else 0
        df = len(inverted_index.get(term, []))
        idf = math.log(N / (1 + df)) if df else 0
        tfidf_sum += tf * idf

    tfidf_avg = tfidf_sum / len(query_tokens) if query_tokens else 0

    return [tfidf_avg, common_terms, doc_len]

# DATASET BUILDING
X = []
y = []

for query, docs in query_examples.items():
    for doc_id, label in docs.items():
        features = extract_features(query, doc_id, documents)
        X.append(features)
        y.append(label)
        
# TRAINING MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# NEW SEARCHING FUNCTION WITH TRAINED DATA
def ranked_search(query):
    candidate_docs = set()
    query_tokens = preprocess_query(query)
    
    # All docs containing at least 1 term
    for term in query_tokens:
        candidate_docs.update(inverted_index.get(term, []))
    
    scored_docs = []
    for doc_id in candidate_docs:
        features = extract_features(query, int(doc_id), documents)
        features_scaled = scaler.transform([features])
        score = model.predict_proba(features_scaled)[0][1]
        scored_docs.append((int(doc_id), score))

    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 results (doc_id, score) for '{query}':")
    for doc_id, score in ranked[:20]:
        print(f"Doc {doc_id} → score: {score:.4f}")

# NEW SEARCH LAUNCHING
ranked_search("space mission")


# ------------------------------------------------------------
# PHASE 4: LTR PERSONALIZATION

# DEFINING USER PREFERENCES
user_profiles = {
    "marco": {
        "interests": ["space", "mission", "satellite"]
    },
    "stefano": {
        "interests": ["middle", "east", "politic"]
    },
    "carmen": {
        "interests": ["graphics", "computer", "render"]
    }
}

# PERSONALIZING QUERY BASED ON USER PREFERENCES
def personalize_query(query, user_id):
    query_tokens = set(preprocess_query(query))
    extended_tokens = list(query_tokens.union(user_profiles[user_id]["interests"]))
    return " ".join(extended_tokens)


# PERSONALIZING SEARCH RANKING
def personalized_ranked_search(query, user_id, top_k=5):
    full_query = personalize_query(query, user_id)
    ranked_search(full_query)


# PERSONALIZED SEARCH LAUNCHING
personalized_ranked_search("space mission", "marco")
personalized_ranked_search("middle east", "stefano")
personalized_ranked_search("computer graphics", "carmen")


# ------------------------------------------------------------
# PHASE 5: PERFORMANCE EVALUATION

# CALCULATING PRECISION@K METRIC
def precision_at_k(true_labels, predicted_ids, k=5):
    relevant = set([doc_id for doc_id, rel in true_labels.items() if rel > 0])
    retrieved = predicted_ids[:k]
    relevant_retrieved = [doc_id for doc_id in retrieved if doc_id in relevant]
    return len(relevant_retrieved) / k

# CALCULATING MAP (MEAN AVERAGE PRECISION) METRIC
def average_precision(true_labels, predicted_ids):
    relevant = set([doc_id for doc_id, rel in true_labels.items() if rel > 0])
    if not relevant:
        return 0
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(predicted_ids):
        if doc_id in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant)

def mean_average_precision(test_queries, rank_function):
    ap_list = []
    for query, relevant_docs in test_queries.items():
        predicted = rank_function(query)
        ap = average_precision(relevant_docs, predicted)
        ap_list.append(ap)
    return sum(ap_list) / len(ap_list)

# TEST QUERIES FOR EVALUATION
def rank_tf_idf(query):
    query_terms = preprocess_query(query)
    scores = compute_scores(query_terms, documents)
    return [int(doc_id) for doc_id, _ in scores]

def rank_ltr(query):
    query_terms = preprocess_query(query)
    candidate_docs = set()
    for term in query_terms:
        candidate_docs.update(inverted_index.get(term, []))

    scored_docs = []
    for doc_id in candidate_docs:
        features = extract_features(query, int(doc_id), documents)
        features_scaled = scaler.transform([features])
        score = model.predict_proba(features_scaled)[0][1]
        scored_docs.append((int(doc_id), score))

    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]

# PERFORMANCE EVALUATION
for name, rank_func in [("TF-IDF", rank_tf_idf), ("LTR", rank_ltr)]:
    print(f"== {name} ==")
    for query, rel_docs in query_examples.items():
        predicted = rank_func(query)
        prec = precision_at_k(rel_docs, predicted, k=5)
        print(f"{query} → Precision@5: {prec:.2f}")
        
for name, rank_func in [("TF-IDF", rank_tf_idf), ("LTR", rank_ltr)]:
    map_score = mean_average_precision(query_examples, rank_func)
    print(f"{name} → MAP: {map_score:.4f}")

# EVALUATION FOR PARTICULAR USER
def rank_ltr_personalized(query, user_id):
    full_query = personalize_query(query, user_id)
    return rank_ltr(full_query)

print("== LTR Custom ==")
for user in user_profiles.keys():
    for query, rel_docs in query_examples.items():
        predicted = rank_ltr_personalized(query, user)
        prec = precision_at_k(rel_docs, predicted)
        print(f"[{user}] {query} → Precision@5: {prec:.2f}")
