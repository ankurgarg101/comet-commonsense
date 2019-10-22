"""
Implements one way of computing the causal graph and the related digraph generation
"""

from kutils import *
from graphviz import Digraph

def create_causal_graph(bert_client, chunk, comet_out, partial_order, use_order):

    """
    Create the causal graph using the output from the COMET model
    """

    print('Causal Graph')

    causal_graph = {}
    sent_ids = list(comet_out.keys())

    for sent_id in comet_out:
        causal_graph[sent_id] = []

        for beam_id, beam in enumerate(comet_out[sent_id]['xNeed']['beams']):

            max_sim = 0.0
            max_beam = None
            max_sent_id2= None

            for sent_id2 in sent_ids:
                if sent_id2 == sent_id:
                    continue

                if use_order:
                    if sent_id2 not in partial_order['reverse'][sent_id]:
                        continue

                for beam_id2, beam2 in enumerate(comet_out[sent_id2]['Effect']['beams']):

                    cosine_sim = comp_bert_sim(bert_client, beam, beam2)

                    if max_sim < cosine_sim:
                        max_sim = cosine_sim
                        max_beam = beam2
                        max_sent_id2 = sent_id2

            if max_beam is not None:

                causal_graph[sent_id].append({
                        'need': beam,
                        'effect': max_beam,
                        'max_sent_id2': max_sent_id2,
                        'sim_score': max_sim
                    })

    #print('===============')
    #print('causal_graph', causal_graph)
    return causal_graph

def create_digraph(causal_graph, text_chunk, fname):

    """'
    Create a digraph using Graphviz and store it using chunk_id as the filename.
    """

    print('Computing Digraph')

    digraph = Digraph('causal_graph', filename=fname, format='png')

    # Create all nodes in the graph
    for sent_id in causal_graph:
        digraph.node(str(sent_id), '{}. {}'.format(sent_id, text_chunk[sent_id]))

    # Create the edges by traversing the causal graph
    for sent_id in causal_graph:

        for relation in causal_graph[sent_id]:

            digraph.edge(str(relation['max_sent_id2']), relation['effect'], label='effect')

            digraph.edge(relation['effect'], relation['need'], label=str(relation['sim_score']))

            digraph.edge(relation['need'], str(sent_id), label='need')

    digraph.render()
