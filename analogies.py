#
# Adapt gensim's code on evaluate_word_analogies to allow for the other
# definitions from Levy & Goldberg
#

from gensim import utils, matutils, logging
import itertools
from six import string_types
from numpy import ndarray, float32 as REAL, array, dot, sqrt, newaxis
from numpy.random import normal as np_normal

logger = logging.getLogger(__name__)


def most_similar_pairdirection(
        model, negative_shift, positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None):
    """
    Modifies gensim's most_similar function to implement the PAIRDIRECTION analogy method.
    """
    model.init_sims()

    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in positive
    ]
    negative = [
        (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in negative
    ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        elif word in model.key_to_index:
            mean.append(weight * model.vectors[model.key_to_index[word]])
            all_words.add(model.key_to_index[word])
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    if indexer is not None:
        return indexer.most_similar(mean, topn)

    limited = model.vectors.copy() if restrict_vocab is None else model.vectors[:restrict_vocab].copy()

    # Shift limited by b to conserve the direction -- we need to shift and re-normalize
    if isinstance(negative_shift, ndarray):
        negative_shift_vec = negative_shift
    elif negative_shift in model.key_to_index:
        negative_shift_vec = model.vectors[model.key_to_index[negative_shift]]
    else:
        raise KeyError("word '%s' not in vocabulary" % negative_shift)
    limited -= negative_shift_vec
    limited_norms = sqrt((limited ** 2).sum(-1))[..., newaxis]
    limited_norms[model.key_to_index[negative_shift]] = 1.
    limited = (limited / limited_norms).astype(REAL)


    dists = dot(limited, mean)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    result = [(model.index_to_key[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]


def evaluate_word_analogies_multiple(
        model, analogies, restrict_vocab=300000, case_insensitive=True,
        dummy4unknown=False, method='3COSADD', top_threshold=5, sds=None, scaling=None):
    """Compute performance of the model on an analogy test set.

    The accuracy is reported (printed to log and returned as a score) for each section separately,
    plus there's one aggregate summary at the end.

    This method corresponds to the `compute-accuracy` script of the original C word2vec.
    See also `Analogy (State of the art) <https://aclweb.org/aclwiki/Analogy_(State_of_the_art)>`_.

    Parameters
    ----------
    analogies : str
        Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.
        See `gensim/test/test_data/questions-words.txt` as example.
    restrict_vocab : int, optional
        Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
        This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
        in modern word embedding models).
    case_insensitive : bool, optional
        If True - convert all words to their uppercase form before evaluating the performance.
        Useful to handle case-mismatch between training tokens and words in the test set.
        In case of multiple case variants of a single word, the vector for the first occurrence
        (also the most frequent if vocabulary is sorted) is taken.
    dummy4unknown : bool, optional
        If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
        Otherwise, these tuples are skipped entirely and not used in the evaluation.
    similarity_function : str, optional
        Function name used for similarity calculation.

    Returns
    -------
    score : float
        The overall evaluation score on the entire evaluation set
    sections : list of dict of {str : str or list of tuple of (str, str, str, str)}
        Results broken down by each section of the evaluation set. Each dict contains the name of the section
        under the key 'section', and lists of correctly and incorrectly predicted 4-tuples of words under the
        keys 'correct' and 'incorrect'.

    """

    if top_threshold > 100:
        raise Exception('Threshold should be <= 100, or adjust number of pairs returned in sims.')
    if restrict_vocab:
        raise Exception('Not implemented')

    ok_keys = model.index_to_key[:restrict_vocab]
    #ok_keys = [k for k in ok_keys if k is not None]
    if case_insensitive:
        ok_vocab = {k.upper(): model.get_index(k) for k in reversed(ok_keys)}
    else:
        ok_vocab = {k: model.get_index(k) for k in reversed(ok_keys)}
    oov = 0
    logger.info("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)
    sections, section = [], None
    quadruplets_no = 0
    with utils.open(analogies, 'rb') as fin:
        for line_no, line in enumerate(fin):
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    model._log_evaluate_word_analogies(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': [], 'MeanReciprocalRank': []}
            else:
                if not section:
                    raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except ValueError:
                    logger.info("Skipping invalid line #%i in %s", line_no, analogies)
                    continue
                quadruplets_no += 1
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    oov += 1
                    if dummy4unknown:
                        logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                        section['incorrect'].append((a, b, c, expected))
                    else:
                        logger.debug("Skipping line #%i with OOV words: %s", line_no, line.strip())
                    continue
                original_key_to_index = model.key_to_index
                original_vectors = model.vectors
                model.key_to_index = ok_vocab
                ignore = {a, b, c}  # input words to be ignored
                predicted_list = []

                # Sample embeddings if standard deviations are provided
                if sds:
                    # ok_keys are the keys included in the decade's vocab
                    sds_idx = [sds.key_to_index[w] for w in ok_keys]
                    sds_reordered = sds.vectors[sds_idx, :]  # sd vectors ordered per model.vectors

                    # Sample (V, D) of N(0, 1)
                    sampled_posteriors = np_normal(size=sds_reordered.shape)
                    sampled_posteriors *= sds_reordered
                    sampled_posteriors *= scaling
                    sampled_posteriors += model.vectors
                    model.vectors = sampled_posteriors
                    model.fill_norms(force=True)

                if method == '3COSADD':
                    sims = model.most_similar(
                        positive=[b, c], negative=[a], topn=100, restrict_vocab=restrict_vocab)
                elif method == 'PAIRDIRECTION':
                    sims = most_similar_pairdirection(
                        model, negative_shift=c, positive=[b], negative=[a], topn=100, restrict_vocab=restrict_vocab)
                elif method == '3COSMUL':
                    sims = model.most_similar_cosmul(
                        positive=[b, c], negative=[a], topn=100, restrict_vocab=restrict_vocab)
                else:
                    raise Exception('Check analogy method.')

                model.key_to_index = original_key_to_index
                model.vectors = original_vectors
                model.fill_norms(force=True)

                # Compute accuracy
                for element in sims:
                    predicted = element[0].upper() if case_insensitive else element[0]
                    if predicted in ok_vocab and predicted not in ignore:
                        predicted_list.append(predicted)
                        if expected not in predicted_list and len(predicted_list) >= top_threshold:
                            logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                            break
                        if expected in predicted_list:
                            break
                if expected in predicted_list:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))

                # Compute Mean Reciprocal Rank
                word_elements = [w.upper() if case_insensitive else w for (w, score) in sims]
                RR = 0
                for i, w in enumerate(word_elements):
                    if w == expected:
                        RR = 1 / (i + 1)
                section['MeanReciprocalRank'].append(RR)
    if section:
        # store the last section, too
        sections.append(section)
        model._log_evaluate_word_analogies(section)

    total_MRR = list(itertools.chain.from_iterable(s['MeanReciprocalRank'] for s in sections))
    total_MRR = sum(total_MRR) / len(total_MRR) if len(total_MRR) > 0 else None

    total = {
        'section': 'Total accuracy',
        'correct': list(itertools.chain.from_iterable(s['correct'] for s in sections)),
        'incorrect': list(itertools.chain.from_iterable(s['incorrect'] for s in sections)),
        'MeanReciprocalRank': total_MRR
    }

    # Compute MRR for each section
    for section in sections:
        sec_MRR = section['MeanReciprocalRank']
        section['MeanReciprocalRank'] = sum(sec_MRR) / len(sec_MRR) if len(sec_MRR) > 0 else None

    oov_ratio = float(oov) / quadruplets_no * 100
    logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
    if not dummy4unknown:
        logger.info(
            'NB: analogies containing OOV words were skipped from evaluation! '
            'To change this behavior, use "dummy4unknown=True"'
        )
    analogies_score = model._log_evaluate_word_analogies(total)
    sections.append(total)
    # Return the overall score and the full lists of correct and incorrect analogies
    return analogies_score, sections
