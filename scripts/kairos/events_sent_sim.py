"""
Computes the causal chains based on the similarity between the need and effect of a predicate pair.
"""

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

        for rel_name in comet_out[sent_id]:

            if rel_name == 'Effect':
                continue

            max_score = 0.0
            max_sent_id2 = None
            max_n_beam = None
            max_cosine_sim = None


            for beam_id, beam in enumerate(comet_out[sent_id][rel_name]['beams']):

                beam_score = np.exp(comet_out[sent_id][rel_name]['beam_losses'][beam_id])
                for sent_id2 in sent_ids:
                    if sent_id2 == sent_id:
                        continue

                    if use_order:
                        if rel_name == 'xNeed':
                            if sent_id2 > sent_id:
                                continue
                        elif rel_name == 'Effect':
                            if sent_id2 < sent_id:
                                continue

                    cosine_sim = comp_bert_sim(bert_client, beam, chunk_predicates[sent_id2]['sentence'])

                    if cosine_sim < 0.5:
                        continue

                    score = cosine_sim * beam_score

                    if max_score < score:
                        max_score = score
                        max_cosine_sim = cosine_sim
                        max_n_beam = beam
                        max_sent_id2 = sent_id2

            if max_n_beam is not None:
                causal_graph[sent_id].append({
                        'rel_name': rel_name,
                        'beam': max_n_beam,
                        'max_sent_id2': max_sent_id2,
                        'max_score': score,
                        'max_sim_score': max_cosine_sim
                    })

    #print('===============')
    #print('causal_graph', causal_graph)
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
            if relation['rel_name'] == 'xNeed':
                digraph.edge(str(relation['max_sent_id2']), relation['beam'], label=str(relation['max_sim_score']))
                digraph.edge(relation['beam'], str(sent_id), label='need')

            elif relation['rel_name'] == 'Effect':
                digraph.edge(str(sent_id), relation['beam'], label='effect')
                digraph.edge(relation['beam'], str(relation['max_sent_id2']), label=str(relation['max_sim_score']))

    digraph.render()
