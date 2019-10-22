import numpy as np

def comp_bert_sim(bert_client, sent1, sent2):
    """
    Return the pair-wise sentence similarity based on BERT Vectors
    Reference for usage: https://github.com/hanxiao/bert-as-service
    """

    bert_vecs = bert_client.encode([sent1, sent2])
    cosine_sim = (np.dot(bert_vecs[0], bert_vecs[1])) / (np.linalg.norm(bert_vecs[0]) * np.linalg.norm(bert_vecs[1]))
    return cosine_sim