# Prerequisite knowledge for Recommendation System

## Basic usage on pandas
```
import pandas as pd

# Create DataFrame
df = pd.DataFrame({'col1': [1, 4, 7, 10],'col2': [18, 16, 18, 19],'col3':[2,2,2,9],'col4':[99,99,99,99]})
'''
The result is like that
   col1  col2  col3  col4
0     1    18     2    99
1     4    16     2    99
2     7    18     2    99
3    10    19     9    99
Pay attention to index [0,1,2,3]
'''

# Change index by df.set_index('col2')
col2Indexed = df.set_index('col2')
'''
The index change from [0,1,2,3] to [18,16,18,19]
      col1  col3  col4
col2
18       1     2    99
16       4     2    99
18       7     2    99
19      10     9    99
'''

# Get unique index
list(col2Indexed.index.unique().values) 
'''
The output is [18, 16, 19]
'''

# Enumerate index
for idx, col2 in enumerate(list(col2Indexed.index.unique().values)):
    print(idx)
    print(col2)
'''
The output is showed below
0
18
1
16
2
19
'''

# Use loc
f = col2Indexed.loc[19]
'''
col1    10
col3     9
col4    99
Name: 19, dtype: int64
'''

type(f)
'''
<class 'pandas.core.series.Series'>
'''

f['col1']
'''
10
'''

f = col2Indexed.loc[18]
'''
      col1  col3  col4
col2
18       1     2    99
18       7     2    99
'''

type(f)
'''
<class 'pandas.core.frame.DataFrame'>
'''

# Group by
col2Indexed.groupby('col3')['col1'].sum()
'''
col3
2    12
9    10
Name: col1, dtype: int64
'''

col2Indexed.groupby('col3')['col1'].sum().reset_index()
'''
   col3  col1
0     2    12
1     9    10
'''

col2Indexed.groupby('col3')['col1'].sum().sort_values(ascending=True).reset_index()
'''
   col3  col1
0     9    10
1     2    12
'''

# Reset index to int type index
col2Indexed.reset_index()
'''
The output is showed below
   col2  col1  col3  col4
0    18     1     2    99
1    16     4     2    99
2    18     7     2    99
3    19    10     9    99
'''

# DataFrame isin & ~ operation
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
'''
   A  B
0  1  a
1  2  b
2  3  f
'''
df.isin([1, 3, 12, 'a'])
'''
       A      B
0   True   True
1  False  False
2   True  False
'''
~df.isin([1, 3, 12, 'a'])
'''
       A      B
0  False  False
1   True   True
2  False   True
'''

# Get values from Series
df['B'].values
'''
array(['a', 'b', 'f'], dtype=object)
'''
# Select element by isin
df['A'].isin([1, 3, 12, 'a'])
'''
0     True
1    False
2     True
Name: A, dtype: bool
'''

df['A'][df['A'].isin([1, 3, 12, 'a'])]
'''
0    1
2    3
Name: A, dtype: int64
'''
```

# Top-N accuracy

Top-1 accuracy is the conventional accuracy: the model answer (the one with highest probability) must be exactly the expected answer.

Top-5 accuracy means that any of your model 5 highest probability answers must match the expected answer.

For instance, let's say you're applying machine learning to object recognition using a neural network. A picture of a cat is shown, and these are the outputs of your neural network:

- Tiger: 0.4
- Dog: 0.3
- Cat: 0.1
- Lynx: 0.09
- Lion: 0.08
- Bird: 0.02
- Bear: 0.01

Using top-1 accuracy, you count this output as wrong, because it predicted a tiger.

Using top-5 accuracy, you count this output as correct, because cat is among the top-5 guesses.

```
def _verify_hit_top_n(item_id, recommended_items, topn):
    index = -1
    for i, c in enumerate(recommended_items):
        if c == item_id:
            index = i
            break
    hit = int(index in range(0, topn))
    return hit, index

recommended_items = ['Tiger', 'Dog', 'Cat', 'Lynx', 'Lion', 'Bird', 'Bear']
_verify_hit_top_n('Cat',recommended_items, 1)
'''
(0, -1)
'''

recommended_items = ['Tiger', 'Dog', 'Cat', 'Lynx', 'Lion', 'Bird', 'Bear']
_verify_hit_top_n('Cat',recommended_items, 5)
'''
(1, 2)
'''

# Another version to implement Top-N
def _verify_hit_top_n(item_id, recommended_items, topn):
    try:
        index = next(i for i, c in enumerate(recommended_items) if c == item_id)
    except:
        index = -1
    hit = int(index in range(0, topn))
    return hit, index
```
