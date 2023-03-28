import glob
import os
import gensim
import numpy as np


def load_coha_HistWords(input_dir, only_nonzero):
    vectors_list = glob.glob(f'{input_dir}/*vectors.txt')
    vectors = {}
    for file_name in vectors_list:
        file_decade = file_name.split(os.path.sep)[-1][:4]

        if only_nonzero:
            temp_file_name = 'vectors.txt'
            with open(temp_file_name, 'w') as wf:
                with open(file_name, 'r') as rf:
                    for line in rf:
                        w, vec = line.split(' ', maxsplit=1)
                        npvec = np.fromstring(vec, sep=' ')
                        if np.linalg.norm(npvec) > 1e-6:
                            wf.write(f"{w} {vec}")
            file_name = temp_file_name

        vectors[file_decade] = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False, no_header=True)

        if only_nonzero:
            os.remove(temp_file_name)

    return vectors


def load_BBB_nonzero(input_dir, file_stamp, run_id, only_nonzero, match_vectors=None):
    bbb_vecs = {}
    for decade in range(181, 201):
        decade_str = str(decade) + '0'
        file_name = os.path.join(input_dir, f"decade_embeddings_{file_stamp}_{run_id}_{decade}.txt")
        if only_nonzero:
            assert match_vectors is not None
            temp_file_name = 'vectors.txt'
            with open(temp_file_name, 'w') as wf:
                with open(file_name, 'r') as rf:
                    for line in rf:
                        w, vec = line.split(' ', maxsplit=1)
                        if w in list(match_vectors[decade_str].key_to_index.keys()):
                            wf.write(f"{w} {vec}")
            file_name = temp_file_name

        bbb_vecs[decade_str] = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False, no_header=True)

        if only_nonzero:
            os.remove(temp_file_name)

    return bbb_vecs
