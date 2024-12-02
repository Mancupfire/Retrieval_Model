import json
import numpy as np
import torch
from torch_geometric.data import Data
     

with open('edinburgh-keywords_train.json', 'r') as f:
     train_data = json.load(f)

keywords = list(train_data['np2count'].keys())
     
# Loại bỏ những từ bị trùng
keyword_set = set(keywords)
     
def extract_users(info):
    '''
    input:
        info: data['np2users']
    output:
        list_user, user2kw
    '''
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

# Danh sách các nhà hàng trong tập train
restaurants = list(train_data['np2rests'])
restaurant_set = set(restaurants)
listres = []
for kw in train_data['np2rests'].keys():
      listres.extend(train_data['np2rests'][kw].keys())

restaurant_set = set(listres)
listres.extend(train_data['np2rests'][kw].keys())
     
# Tạo ma trận liên kết từ keywords và restaurant
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

# Ma trận liên kết giữa keywords và restaurants
with open('edinburgh-keywords_test.json', 'r') as r:
    test_data = json.load(r)

user_keywords = list(test_data['np2reviews'].keys())
user_keywords_list = list(user_keywords)

test_users, test_users2kw = extract_users(test_data['np2users'])
    
for kw in test_users2kw:
    t  = np.zeros((1, num_keywords))
    keywords = kw[:10]
    for keys in keywords:
        if keys in keyword_set:
           idx_kw = keyword_set.index(keys)
           t[0][idx_kw] = 1
    R = np.dot(t, a)
    result = np.argsort(R[0])[::-1][:10]

print(result)
