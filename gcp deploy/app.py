 
import pandas as pd
import sklearn 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("ngo_details.csv")

tfidf = TfidfVectorizer(analyzer='word', min_df=0)

vec_arr = tfidf.fit_transform(df['Description'])
vec_features = tfidf.get_feature_names()

sim_mat = cosine_similarity(vec_arr, vec_arr)

titles = df['Name']
indices = pd.Series(df.index, index=df['Name'])

def get_recommendations(name):
    idx = indices[name]
    sim_scores = list(enumerate(sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    ngo_indices = [i[0] for i in sim_scores]
    print(sim_scores)
    print(ngo_indices)
    return titles.iloc[ngo_indices].values

print(get_recommendations('DEV GOVIND GRAMIN VIKAS SANSTHAN'))
