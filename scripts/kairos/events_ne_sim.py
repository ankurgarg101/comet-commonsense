"""
Computes the causal chains based on the similarity between the need and effect of a predicate pair.
"""
import numpy as np
from kutils import *
from graphviz import Digraph

def create_causal_graph(bert_client, chunk_predicates, comet_out, use_order):

    """
    Create the causal graph using the output from the COMET model
    """

    print('Causal Graph')

    causal_graph = {}
    sent_ids = list(comet_out.keys())

    for sent_id in comet_out:
        causal_graph[sent_id] = []

        max_score = 0.0
        max_n_beam = None
        max_sent_id2 = None
        max_e_beam = None
        max_cosine_sim = None

        for beam_id, beam in enumerate(comet_out[sent_id]['xNeed']['beams']):

            beam1_score = np.exp(comet_out[sent_id]['xNeed']['beam_losses'][beam_id])

            for sent_id2 in sent_ids:
                if sent_id2 == sent_id:
                    continue

                if use_order:
                    if sent_id2 > sent_id:
                        continue

                for beam_id2, beam2 in enumerate(comet_out[sent_id2]['Effect']['beams']):

                    cosine_sim = comp_bert_sim(bert_client, beam, beam2)

                    if cosine_sim < 0.5:
                        continue

                    beam2_score = np.exp(comet_out[sent_id2]['Effect']['beam_losses'][beam_id2])

                    score = cosine_sim * beam1_score * beam2_score

                    if max_score < score:
                        max_score = score
                        max_n_beam = beam
                        max_e_beam = beam2
                        max_sent_id2 = sent_id2
                        max_cosine_sim = cosine_sim

        if max_n_beam is not None:

            causal_graph[sent_id].append({
                    'need': max_n_beam,
                    'effect': max_e_beam,
                    'max_sent_id2': max_sent_id2,
                    'sim_score': max_cosine_sim,
                    'score': max_score,
                })

    return causal_graph

def create_digraph(causal_graph, chunk_predicates, fname):

    """'
    Create a digraph using Graphviz and store it using chunk_id as the filename.
    """

    print('Computing Digraph')

    digraph = Digraph('causal_graph', filename=fname, format='png')

    # Create all nodes in the graph
    for sent_id in causal_graph:
        digraph.node(str(sent_id), '{}. {}'.format(sent_id, chunk_predicates[sent_id]['sentence']))

    # Create the edges by traversing the causal graph
    for sent_id in causal_graph:

        for relation in causal_graph[sent_id]:

            digraph.edge(str(relation['max_sent_id2']), relation['effect'], label='effect')

            digraph.edge(relation['effect'], relation['need'], label=str(relation['sim_score']))

            digraph.edge(relation['need'], str(sent_id), label='need')

    digraph.render()
