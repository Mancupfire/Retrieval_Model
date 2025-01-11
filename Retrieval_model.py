import json
import numpy as np
import torch
import csv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

with open('singapore-keywords_train.json', 'r') as f:
    train_data = json.load(f)

keywords = list(train_data['np2count'].keys())

keyword_set = set(keywords)

def extract_users(info):
    l_user, user2kw = [], []
    for ii in info:
        lus = info[ii]
        for u in lus:
            if u not in l_user:
                l_user.append(u)
                user2kw.append([])
            idx = l_user.index(u)
            user2kw[idx].append(ii)
    return l_user, user2kw

train_users, train_users2kw = extract_users(train_data['np2users'])

restaurant_set = set()
listres = []
for kw in train_data['np2rests'].keys():
    listres.extend(train_data['np2rests'][kw].keys())
restaurant_set = set(listres)

keyword_set = list(keyword_set)
restaurant_set = list(restaurant_set)
restaurants = len(listres)
num_keywords = len(keyword_set)
num_restaurants = len(restaurant_set)
a = np.zeros((num_keywords, num_restaurants))

for kw in train_data['np2rests'].keys():
    for res in train_data['np2rests'][kw].keys():
        idx_kw = keyword_set.index(kw)
        idx_res = restaurant_set.index(res)
        a[idx_kw][idx_res] = 1

keyword_embeddings = model.encode(list(keyword_set))

with open('singapore-keywords_test.json', 'r') as r:
    test_data = json.load(r)

user_keywords = list(test_data['np2reviews'].keys())
user_keywords_list = list(user_keywords)

test_users, test_users2kw = extract_users(test_data['np2users'])

test_keywords = [kw for sublist in test_users2kw for kw in sublist]
test_keyword_embeddings = model.encode(test_keywords)

similarity_scores = cosine_similarity(test_keyword_embeddings, keyword_embeddings)

filtered_keywords = []
for i, user_kw in enumerate(test_users2kw):
    updated_user_kw = []
    for kw in user_kw:
        if kw not in keyword_set:
            test_idx = test_keywords.index(kw)
            sim_scores = similarity_scores[test_idx]

            best_match_idx = np.argmax(sim_scores)
            best_match_keyword = keyword_set[best_match_idx]

            updated_user_kw.append(best_match_keyword)
        else:
            updated_user_kw.append(kw)

    filtered_keywords.append(updated_user_kw)

test_users2kw = filtered_keywords

results = []
for kw in test_users2kw:
    t = np.zeros((1, len(keyword_set)))
    keywords = kw[:10]
    for keys in keywords:
        if keys in keyword_set:
            idx_kw = keyword_set.index(keys)
            t[0][idx_kw] = 1
    R = np.dot(t, a)
    result = np.argsort(R[0])[::-1][:10]
    results.append(result)

if __name__ == "__main__":
        for i, result in enumerate(results):
            restaurant_names = [restaurant_set[idx] for idx in result]
            print(f"The result for user {i} is: {restaurant_names}")

csv_file_path = "./result/results.csv"

with open(csv_file_path, mode="w", newline="") as file:
    fieldnames = ['number', 'user', 'restaurant_name']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    number = 1
    for user, restaurant_indices in zip(test_users, results):
        for idx in restaurant_indices:
            restaurant_name = restaurant_set[idx]
            writer.writerow({'number': number, 'user': user, 'restaurant_name': restaurant_name})
            number += 1

print(f"\nResults saved to: {csv_file_path}")