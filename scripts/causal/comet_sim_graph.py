import argparse
import numpy as np
import os
import simplejson
import pickle

import atomic_predict
from bert_serving.client import BertClient
from sentence_transformers import SentenceTransformer

def prune_graph(args, comet_model, text_chunk, chunk_id):

	"""
	Create the causal graph for a given text chunk
	@param text_chunk: List of sentences.
	"""

	folder_name = 'chunk_comet_out_{}'.format(args.sampling_algorithm)
	if not os.path.exists(os.path.join(args.data_dir, folder_name)):
		os.makedirs(os.path.join(args.data_dir, folder_name))
	elif os.path.exists(os.path.join(args.data_dir, folder_name, '{}.json'.format(chunk_id))):
		with open(os.path.join(args.data_dir, folder_name, '{}.json'.format(chunk_id)), 'rb') as f:
			comet_out_pruned = pickle.load(f)
			return comet_out_pruned

	print('Computing Pruned Comet Output Graph')

	comet_output = atomic_predict.gen_preds(comet_model, text_chunk)

	# Dictionary to store the pruned graph required for computing the
	comet_out_pruned = {}
	for sent_idx in comet_output:

		comet_out_pruned[sent_idx] = {}
		output = comet_output[sent_idx]

		# For each sentence merge all sentences without 'none' in the oEffect and xEffect category.
		# Also, keep the xNeed Category
		# TODO: Think about other categories if they can fit the use case.

		if 'xNeed' in output:
			comet_out_pruned[sent_idx]['xNeed'] = {
				'beams': [],
				'beam_losses': []
			}

			for beam_id in range(len(output['xNeed']['beams'])):

				if not (output['xNeed']['beams'][beam_id] == 'none'):
					comet_out_pruned[sent_idx]['xNeed']['beams'].append(output['xNeed']['beams'][beam_id])
					comet_out_pruned[sent_idx]['xNeed']['beam_losses'].append(output['xNeed']['beam_losses'][beam_id])

		if 'oEffect' in output:
			comet_out_pruned[sent_idx]['Effect'] = {
				'beams': [],
				'beam_losses': []
			}

			for beam_id in range(len(output['oEffect']['beams'])):

				if not (output['oEffect']['beams'][beam_id] == 'none'):
					comet_out_pruned[sent_idx]['Effect']['beams'].append(output['oEffect']['beams'][beam_id])
					comet_out_pruned[sent_idx]['Effect']['beam_losses'].append(output['oEffect']['beam_losses'][beam_id])

		if 'xEffect' in output:
			if 'Effect' not in comet_out_pruned[sent_idx]:
				comet_out_pruned[sent_idx]['Effect'] = {
					'beams': [],
					'beam_losses': []
				}

			for beam_id in range(len(output['xEffect']['beams'])):

				if not (output['xEffect']['beams'][beam_id] == 'none'):
					comet_out_pruned[sent_idx]['Effect']['beams'].append(output['xEffect']['beams'][beam_id])
					comet_out_pruned[sent_idx]['Effect']['beam_losses'].append(output['xEffect']['beam_losses'][beam_id])

	with open(os.path.join(args.data_dir, folder_name, '{}.json'.format(chunk_id)), 'wb') as f:
		pickle.dump(comet_out_pruned, f)

	return comet_out_pruned

def get_partial_order(args):

	# Process the events and relations and constructs a partial order of sentences in the text chunk
	partial_order = {}

	graph = {}
	with open(os.path.join(args.data_dir, 'relations.json'), 'r') as f:
		relations = simplejson.load(f)

	for rel_type in relations:
		for rel in relations[rel_type]:
			if rel['chunk_id'] not in graph:
				graph[rel['chunk_id']] = {}
				for i in range(5):
					graph[rel['chunk_id']][i] = set()

			arg1_sent_idx = rel['arg1_sent_idx'] - rel['chunk_id'] * 5
			arg2_sent_idx = rel['arg2_sent_idx'] - rel['chunk_id'] * 5

			if not (arg1_sent_idx == arg2_sent_idx):
				graph[rel['chunk_id']][arg1_sent_idx].add(arg2_sent_idx)

	# Implement the topological sort.
	for chunk_id in graph:
		partial_order[chunk_id] = {
			'fwd': {},
			'reverse': {}
		}

		# Compute dfs from each node to get list of all nodes that follow this node
		for idx in range(5):

			partial_order[chunk_id]['fwd'][idx] = set()
			partial_order[chunk_id]['reverse'][idx] = set()

			visited = [False for i in range(5)]

			stack = [idx]
			visited[idx] = True

			while(stack):
				root = stack.pop()
				for node in graph[chunk_id][root]:
					if not visited[node]:
						partial_order[chunk_id]['fwd'][idx].add(node)
						stack.append(node)
						visited[node] = True

		for idx in range(5):
			for node in partial_order[chunk_id]['fwd'][idx]:
				partial_order[chunk_id]['reverse'][node].add(idx)

	return partial_order

def main(args):

	args = parser.parse_args()
	comet_model = atomic_predict.fetch_model(args)
	if args.sim_client == 'bert':
		bert_client = BertClient()
	elif args.sim_client == 'sbert':
		bert_client = SentenceTransformer('bert-base-nli-mean-tokens')

	if args.sim_method == 'ne':
		from need_effect_sim import create_causal_graph, create_digraph
	elif args.sim_method == 'sent':
		from sentence_sim import create_causal_graph, create_digraph

	with open(os.path.join(args.data_dir, 'text_chunks.json'), 'r') as f:
		text_chunks = simplejson.load(f)

	partial_order = get_partial_order(args)

	if not os.path.exists(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'causal_graph')):
		os.makedirs(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'causal_graph'))

	if not os.path.exists(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'comet_out')):
		os.makedirs(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'comet_out'))

	if not os.path.exists(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'digraphs')):
		os.makedirs(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'digraphs'))

	with open(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'partial_order.json', ), 'wb') as f:
		pickle.dump(partial_order, f)

	for chunk_id in text_chunks:
		print("Processing Chunk: ", chunk_id)
		chunk = text_chunks[chunk_id]
		comet_out_pruned = prune_graph(args, comet_model, chunk, chunk_id)
		causal_graph = create_causal_graph(bert_client, chunk, comet_out_pruned, partial_order[int(chunk_id)], args.use_order)
		create_digraph(causal_graph, chunk, os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'digraphs', str(chunk_id)))

		# Save the causal graph, pruned comet output
		with open(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'causal_graph', '{}.json'.format(chunk_id)), 'wb') as f:
			pickle.dump(causal_graph, f)

		with open(os.path.join(args.data_dir, '{}_{}'.format(args.sim_method, int(args.use_order)), 'comet_out', '{}.json'.format(chunk_id)), 'wb') as f:
			pickle.dump(comet_out_pruned, f)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--model_file", type=str, required=True)
	parser.add_argument("--sampling_algorithm", type=str, default="greedy")
	parser.add_argument('--data_dir', type=str, required=True, default="Directory to read the Dictionary of text chunks from")
	parser.add_argument('--sim_method', type=str, default='ne', help="The similarity method to compute the causal graph")
	parser.add_argument('--use_order', type=bool, default=True, help="Whether to use temporal order of sentences or not")
	parser.add_argument('--sim_client', type=str, default='sbert', help="Which BERT embedding to use (bert, sbert)")
	args = parser.parse_args()
	main(args)
