import json

with open('home_status_method_hierarchical.json', 'r') as f:
    home = json.load(f)

lens = []
lens2 = set()
for d in home :
    d_ = home[d]['capabilities']
    for k in d_ : 
        for v in d_[k] :
            lens2.add(v)
print(lens2)
print(len(lens2))
# print(sum(lens) / len(lens))
exit()
home40_capa = home['40']['capabilities']
home40_stat = home['40']['home_status']

with open('home_40_status.json', 'w') as f :
    json.dump({'capabilities': home40_capa, 'status': home40_stat}, f)