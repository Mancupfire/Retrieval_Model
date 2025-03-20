import json
import numpy as np
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import pipeline

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load training data
with open('edinburgh-keywords_train.json', 'r') as f:
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

# Load test data
with open('edinburgh-keywords_test.json', 'r') as r:
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

llm_pipeline = pipeline("text-generation", model="gpt2")

def re_rank_candidates(user_id, candidate_restaurants, user_keywords):
    prompt = (
        f"Người dùng có sở thích: {', '.join(user_keywords[:5])}. "  
        f"Với các candidate restaurants sau: {', '.join(candidate_restaurants[:5])}. "  
        "Hãy xếp hạng lại các restaurant theo mức độ phù hợp với sở thích của người dùng theo thứ tự giảm dần và chỉ in ra danh sách tên các restaurant, cách nhau bằng dấu phẩy."
    )
    generated = llm_pipeline(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
    output = generated[len(prompt):].strip()
    re_ranked = [restaurant.strip() for restaurant in output.split(',')]
    valid_re_ranked = [r for r in re_ranked if r in candidate_restaurants]
    if len(valid_re_ranked) == 0:
        valid_re_ranked = candidate_restaurants
    return valid_re_ranked

final_results = []
for idx, (user, candidate_indices) in enumerate(zip(test_users, results)):
    candidate_restaurants = [restaurant_set[i] for i in candidate_indices]
    user_kw = test_users2kw[idx]
    re_ranked = re_rank_candidates(user, candidate_restaurants, user_kw)
    final_results.append(re_ranked)
    print(f"Re-ranked results for user {user}: {re_ranked}")


# Function return candidate restaurant in dictionary type to use for the System
def generate_results(test_users, results, test_users2kw, restaurant_set, re_ranked):
    output_data = {}

    for idx, (user, restaurant_indices) in enumerate(zip(test_users, results)):
        user_data = {}

        user_keywords = test_users2kw[idx]

        candidate_restaurants = [restaurant_set[i] for i in restaurant_indices]

        re_ranked_restaurants = re_ranked[idx]

        positions = [str(i) for i in restaurant_indices]  

        user_data["kw"] = user_keywords[:10]  
        user_data["candidate"] = re_ranked_restaurants[:10] 
        user_data["positions"] = positions[:10]  
        output_data[user] = user_data

    return output_data 


result_dict = generate_results(test_users, results, test_users2kw, restaurant_set, final_results)
print(result_dict)  


# Function to save the result to a json file for quick checking
def save_results_to_json(test_users, results, test_users2kw, restaurant_set, re_ranked, file_path='./result/results(afterLLM).json'):
    output_data = {}
    for idx, (user, restaurant_indices) in enumerate(zip(test_users, results)):
        user_data = {}
        user_keywords = test_users2kw[idx]
        candidate_restaurants = [restaurant_set[i] for i in restaurant_indices]
        re_ranked_restaurants = re_ranked[idx]
        positions = [str(i) for i in restaurant_indices]
        user_data["kw"] = user_keywords[:10]
        user_data["candidate"] = re_ranked_restaurants[:10]
        user_data["positions"] = positions[:10]
        output_data[user] = user_data
    with open(file_path, mode="w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to: {file_path}")

save_results_to_json(test_users, results, test_users2kw, restaurant_set, final_results)
