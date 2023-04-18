import gensim.models
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from model_to_vectors import load_model
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from bias_utils import *

matplotlib.use('pdf')
#os.chdir('..')

# Load HW
histwords_dir = '../Replication-Garg-2018/data/coha-word'
histwords = load_coha_HistWords(input_dir=histwords_dir, only_nonzero=True)

# Load UK_OG_11
bbb_vecs = load_BBB_nonzero(
    input_dir=os.path.join(Path(__file__).parent, f'data/UK/results'), file_stamp='UK',
    run_id='UK_OG_11', only_nonzero=False, match_vectors=None)

# BBB global embeddings
model = load_model(
    f"data/UK/results/model_best_UK_UK_OG_11.pth.tar",
    f"data/UK/processed/vocabUK_freq.npy")
bbb_global_emb = model.word_input_embeddings

with open("TEMPORARYFILE.txt", "w") as f:
    for word, emb in bbb_global_emb.items():
        f.write(f"{word} {' '.join(map(str, emb))}\n")
global_vecs = gensim.models.KeyedVectors.load_word2vec_format("TEMPORARYFILE.txt", binary=False, no_header=True)
os.remove("TEMPORARYFILE.txt")

""""
# Explore cosine similarities
cs_df = pd.DataFrame()
for decade, model in histwords.items():
    w = model.vectors
    w = pd.DataFrame(w)
    cs = cosine_similarity(w)
    cs_df = pd.concat([cs_df, pd.DataFrame.from_dict(
        {'vectors': ['HW'], 'decade': [int(decade)], 'median_cs': [np.median(cs)]})])

for decade, model in bbb_vecs.items():
    w = model.vectors
    w = pd.DataFrame(w)
    cs = cosine_similarity(w)
    cs_df = pd.concat([cs_df, pd.DataFrame.from_dict(
        {'vectors': ['BBB-Decadal'], 'decade': [int(decade)], 'median_cs': [np.median(cs)]})])

emb_df = pd.DataFrame()
for w, emb in bbb_global_emb.items():
    w_df = pd.DataFrame(emb.reshape(1, -1))
    emb_df = pd.concat([emb_df, w_df])
cs = cosine_similarity(emb_df)
med_cs = np.median(cs)
for decade in bbb_vecs.keys():
    cs_df = pd.concat([cs_df, pd.DataFrame.from_dict(
            {'vectors': ['BBB-Global'], 'decade': [int(decade)], 'median_cs': [med_cs]})])

ax = sns.lineplot(data=cs_df, x='decade', y='median_cs', hue='vectors')
ax.set(xlabel='Decade', ylabel='Median cosine similarity between words')
"""

# Cosine similarities: specific decade (understand distribution)
cs_decade = '2000'
embeddings = {
    'HistWords': histwords[cs_decade].vectors,
    'BBB-Decadal': bbb_vecs[cs_decade].vectors,
    'BBB-Global': global_vecs.vectors
}

def plot_cs(embeddings):
    cs_df = pd.DataFrame()
    for name, embedding in tqdm(embeddings.items()):
        decade_cs = cosine_similarity(pd.DataFrame(embedding))
        mask = np.zeros(decade_cs.shape, dtype='bool')
        mask[np.triu_indices(len(decade_cs))] = True
        decade_cs = pd.DataFrame(decade_cs).mask(mask, None)
        #decade_cs = decade_cs.values
        decade_cs = pd.melt(decade_cs)
        decade_cs = decade_cs.loc[~decade_cs['value'].isna()]
        decade_cs['model'] = name
        cs_df = pd.concat([cs_df, decade_cs])

    g = sns.FacetGrid(cs_df, row='model', sharex=True)
    g.map_dataframe(sns.kdeplot, x='value')
    g.set_titles(row_template="{row_name}")
    g.fig.suptitle('')
    g.figure.savefig(os.path.join('results/embeddings/cosine_sim1990s.png'), dpi=800)

plot_cs(embeddings)

"""
# Measure performance of the global vectors
eval_dir = Path(__file__).parent / "data" / "COHA" / "evaluation"



eval_score = pd.DataFrame()
score, sections = global_vecs.evaluate_word_analogies(str(eval_dir / 'questions-words.txt'))
for section_dict in sections:
    if len(section_dict['correct']) + len(section_dict['incorrect']) == 0:
        accuracy = None
    else:
        accuracy = len(section_dict['correct']) / (len(section_dict['correct']) + len(section_dict['incorrect']))
    eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
        {'task': ['analogy'], 'section': [section_dict['section']], 'accuracy': [accuracy],
         'vectors': ['BBB-Global']})])

# Word similarity (Bruni et al 2012 -- used in HistWords)
pearson, spearman, oov = global_vecs.evaluate_word_pairs(str(eval_dir / 'MEN_dataset_natural_form_full.txt'))
eval_score = pd.concat(
    [eval_score, pd.DataFrame.from_dict(
        {'task': ['Bruni'], 'section': ['pearson_stat'], 'accuracy': [pearson.statistic],
         'vectors': ['BBB-Global']})])

# Visualize HW
nonzero_df = pd.DataFrame()
for decade_str in histwords.keys():
    w = histwords[decade_str].vectors
    # nonzero = nonzero = np.abs(w) > 1e-6
    nonzero = w
    nonzero = pd.DataFrame(nonzero)
    nonzero['decade'] = int(decade_str)
    nonzero['word'] = histwords[decade_str].key_to_index.keys()
    nonzero_df = pd.concat([nonzero_df, nonzero])
embed_df = nonzero_df.copy()
nonzero_df = pd.melt(nonzero_df, id_vars=['word', 'decade'], var_name='dim', value_name='element')
"""
"""
def facet_heatmap(data, color, **kws):
    data = data.pivot_table(values='element', index='word', columns='dim')
    sns.heatmap(data, cbar=True)

g = sns.FacetGrid(nonzero_df, col='decade', col_wrap=5)
g.map_dataframe(facet_heatmap)
g.set_titles(row_template="{row_name}", col_template='{col_name}')
g.fig.suptitle('')
g.figure.savefig(os.path.join("test.png"), dpi=800)
"""