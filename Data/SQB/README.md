# SimpleQuestion-Balance

Please use following script in ipython for detail.
```
import pickle as pkl
test= pkl.load(open("fold-0.test.pickle","rb"))
test[0].__dict__
```
The "relation" for each [item](Item.py) is the id in [relation vocabulary](../Embedding/rel.voc.pickle).