"""Keep the metadata of the Video Games items that survive the filters of synapse_v5.py.
Usage: python filter_meta.py Video_Games.csv meta_Video_Games.jsonl meta_needed.jsonl"""
import sys, json
import pandas as pd
r, meta, out = sys.argv[1:4]
d = pd.read_csv(r); d = d[d.rating >= 4]
while True:
    n0 = len(d)
    uc = d.groupby('user_id').size(); d = d[d.user_id.isin(uc[uc >= 10].index)]
    ic = d.groupby('parent_asin').size(); d = d[d.parent_asin.isin(ic[ic >= 5].index)]
    if len(d) == n0: break
need = set(d.parent_asin)
with open(out, 'w') as o:
    for line in open(meta):
        j = json.loads(line)
        if j.get('parent_asin') in need:
            o.write(json.dumps({x: j.get(x) for x in ['parent_asin', 'title', 'main_category', 'categories', 'store', 'details']}) + '\n')
