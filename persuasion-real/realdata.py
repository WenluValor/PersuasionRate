import nltk
import numpy as np
import pandas as pd
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import bisect


def read_json():
    # df = pd.concat(pd.read_json("clean-yelp/user.json", lines=True, chunksize=10000))
    # df.to_csv("clean-yelp/user.csv", index=False)
    df = pd.read_csv("clean-yelp/review.csv", nrows=100)
    df.to_csv("clean-yelp/review-100.csv", index=False)


def clean_review():
    id1 = 'GXFMD0Z4jEVZBCsbPf4CTQ' # 4.5
    id2 = '_ab50qdWOk0DdB6XOrBitw' # 4.0
    id3 = 'iSRTaT9WngzB8JJ2YKJUig' # 3.5
    id_list = [id1, id2, id3]
    df = pd.concat(pd.read_csv("clean-yelp/review.csv", chunksize=10000))
    for i in range(3):
        id = id_list[i]
        df1 = df[df['business_id'] == id]
        df1.to_csv('clean-yelp/' + str(i) + 'review.csv', index=False)


def get_X():
    df = pd.concat(pd.read_csv("clean-yelp/user.csv", chunksize=10000))
    for i in range(3):
        rev = pd.read_csv('clean-yelp/' + str(i) + 'review.csv')
        result = df[df["user_id"].isin(rev["user_id"])]
        result.to_csv('clean-yelp/' + str(i) + 'user.csv', index=False)


def get_senti():
    # nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    for i in range(3):
        df = pd.read_csv('clean-yelp/' + str(i) + 'review.csv')
        df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        df = df[['review_id', 'user_id', 'text', 'date', 'sentiment', 'stars']]
        df.to_csv('clean-yelp/' + str(i) + 'shortreview.csv', index=False)


def get_TYX(win, t, s):
    for i in range(3):
        df = pd.read_csv('clean-yelp/' + str(i) + 'shortreview.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['prev5_avg'] = (df['sentiment'].shift(1).rolling(window=win, min_periods=1).mean())
        # df['count_neg_prev5'] = ((df['sentiment'] <= 0.2).astype(int).shift(1).rolling(window=win, min_periods=1).sum())
        df['label_T'] = (df['prev5_avg'] >= t).astype(int)
        # df['label_T'] = (df['count_neg_prev5'] >= t).astype(int)
        df['label_Y'] = (df['stars'] >= s).astype(int)

        user = pd.read_csv('clean-yelp/' + str(i) + 'user.csv')
        df['order'] = range(df.shape[0])
        result = (df.merge(user, on='user_id', how='left').sort_values('order').drop(columns='order'))

        result.to_csv('clean-yelp/' + str(i) + 'shortuser.csv', index=False)
        df = df.reset_index(drop=True)
        df[['label_T']].to_csv('clean-yelp/' + str(i) + 'T.csv', index=True)
        df[['label_Y']].to_csv('clean-yelp/' + str(i) + 'Y.csv', index=True)


def get_X():
    for i in range(3):
        df = pd.read_csv('clean-yelp/' + str(i) + 'shortuser.csv')
        df['yelping_since'] = pd.to_datetime(df['yelping_since'])
        now = pd.Timestamp.now()
        delta = now - df['yelping_since']
        df['hours_since'] = delta.dt.total_seconds() / 3600
        df['elite_count'] = (df['elite'].fillna('').str.findall(r'\d{4}').str.len())
        df['id_count'] = (df['friends'].fillna('').str.findall(r'[^,\s]+').str.len())

        X_count = df[['review_count', 'useful', 'funny', 'cool', 'id_count', 'fans',
                      'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
                'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny',
                'compliment_writer', 'compliment_photos']]
        X_count = np.log1p(X_count)
        X_other = df[['hours_since', 'elite_count', 'average_stars']]
        X = pd.concat([X_other, X_count], axis=1)
        X = (X - X.mean()) / X.std()
        X.to_csv('clean-yelp/' + str(i) + 'X.csv', index=True)


def make_data(win, t, s):
    get_TYX(win, t, s)
    get_X()


def get_visited_friends(id):
    df = pd.read_csv('clean-yelp/' + str(id) + 'shortuser.csv')
    visited_col = []
    seen = defaultdict(list)  # user_id -> list of indices

    for i, row in df.iterrows():
        user = row['user_id']

        # parse friends
        if isinstance(row['friends'], str):
            friends = [f.strip() for f in row['friends'].split(',') if f.strip()]
        else:
            friends = row['friends'] if row['friends'] is not None else []

        visited = []

        # collect ALL previous occurrences of friends
        for f in friends:
            if f in seen:
                visited.extend(seen[f])  # <-- key difference

        # optional: sort indices
        visited.sort()

        visited_col.append(visited)

        # update AFTER computing visited
        seen[user].append(i)

    df['visited'] = visited_col

    df['visited'] = df['visited'].apply(lambda x: ', '.join(map(str, x)) if x else '')
    df['index'] = df.index
    df = df[['index', 'visited']]
    df.to_csv('clean-yelp/' + str(id) + 'visiteduser.csv', index=True)


def parse_visited(x):
    if pd.isna(x) or x == '':
        return []
    return [int(v.strip()) for v in str(x).split(',') if v.strip()]


def longest_path_from_visited(id):
    """
    df has:
    - 'index' (0,1,...)
    - 'visited' (list of previous indices)
    """
    df = pd.read_csv('clean-yelp/' + str(id) + 'visiteduser.csv')
    n = len(df)
    df['visited'] = df['visited'].apply(parse_visited)

    dp = [0] * n          # longest path length ending at i
    parent = [-1] * n     # for reconstruction

    for i in range(n):
        visited = df.loc[i, 'visited']

        if not visited:  # no incoming edges
            dp[i] = 1
            parent[i] = -1
        else:
            best_j = max(visited, key=lambda j: dp[j])
            dp[i] = dp[best_j] + 1
            parent[i] = best_j

    # find endpoint of longest path
    end = max(range(n), key=lambda i: dp[i])

    # reconstruct path
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = parent[cur]

    path.reverse()

    return path, dp[end]


def construct_chain(id):
    get_visited_friends(id)
    path, end = longest_path_from_visited(id)
    df = pd.read_csv('clean-yelp/' + str(id) + 'shortuser.csv')
    df['index'] = df.index
    df_subset = df.iloc[path]

    df = pd.read_csv('clean-yelp/' + str(id) + 'visiteduser.csv')
    df_subset['vis_index'] = df['visited']
    df_subset.to_csv('clean-yelp/' + str(id) + 'shortvisitor.csv', index=False)


def construct_TYX(t, s):
    for i in range(0, 3):
        construct_chain(id=i)
        df = pd.read_csv('clean-yelp/' + str(i) + 'shortvisitor.csv')
        df_full = pd.read_csv('clean-yelp/' + str(i) + 'shortuser.csv')

        df['avg_value'] = df['vis_index'].apply(lambda x: compute_avg(x, df_full['sentiment']))
        df['label_T'] = (df['avg_value'] >= t).astype(int)
        df['label_Y'] = (df['stars'] >= s).astype(int)
        df[['label_T']].to_csv('clean-yelp/' + str(i) + 'T.csv', index=True)
        df[['label_Y']].to_csv('clean-yelp/' + str(i) + 'Y.csv', index=True)

        df['yelping_since'] = pd.to_datetime(df['yelping_since'])
        now = pd.Timestamp.now()
        delta = now - df['yelping_since']
        df['hours_since'] = delta.dt.total_seconds() / 3600
        df['elite_count'] = (df['elite'].fillna('').str.findall(r'\d{4}').str.len())
        df['id_count'] = (df['friends'].fillna('').str.findall(r'[^,\s]+').str.len())

        X_count = df[['review_count', 'useful', 'funny', 'cool', 'id_count', 'fans',
                      'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
                      'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny',
                      'compliment_writer', 'compliment_photos']]
        X_count = np.log1p(X_count)
        X_other = df[['hours_since', 'elite_count', 'average_stars']]
        X = pd.concat([X_other, X_count], axis=1)
        X = (X - X.mean()) / X.std()
        X.to_csv('clean-yelp/' + str(i) + 'X.csv', index=True)


def compute_avg(x, value_map):
    if pd.isna(x) or x == '':
        return 0   # or np.nan if you prefer
    indices = [int(v.strip()) for v in x.split(',') if v.strip()]
    values = [value_map[i] for i in indices if i in value_map]
    if len(values) == 0:
        return 0
    return sum(values) / len(values)


if __name__ == '__main__':
    # read_json()
    # clean_review()
    # get_X()
    # get_senti()

    # construct_chain(id=0)
    construct_TYX(t=0.9, s=5.0)

    exit()