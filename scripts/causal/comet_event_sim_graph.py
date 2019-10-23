"""
Module that first extract predicates from the sentences using Allen NLP OpenIE library and then computes a causal chain on that.
"""

import argparse
import numpy as np
import os
import simplejson
import pickle
import collections

import atomic_predict
from bert_serving.client import BertClient
from sentence_transformers import SentenceTransformer
from allennlp.predictors.predictor import Predictor

def fetch_openie_events(args, openie_predictor, chunk):

	"""
	Fetch OpenIE events using AllenNLP library
	"""
	predicates = []
	for sent in chunk:
		out = openie_predictor.predict(sentence=sent)

		words = out['words']
		verbs = out['verbs']
		pred_sent_set = set()
		for vb in verbs:
			args = {}
			for idx, tag in enumerate(vb['tags']):
				if not tag == 'O':
					tag_name = tag.split('-')[1]
					if tag_name not in args:
						args[tag_name] = words[idx]
					else:
						args[tag_name] += ' ' + words[idx]

			sent_objs = []
			if 'BV(ARG0' in args:
				sent_objs.append(args['BV(ARG0'])
			if 'ARG0' in args:
				sent_objs.append(args['ARG0'])
			if 'BV' in args:
				sent_objs.append(args['BV'])
			if 'V' in args:
				sent_objs.append(args['V'])
			if 'ARG1' in args:
				sent_objs.append(args['ARG1'])
			if 'AV' in args:
				sent_objs.append(args['AV'])
			if 'ARG2' in args:
				sent_objs.append(args['ARG2'])
			if 'ARG3' in args:
				sent_objs.append(args['ARG3'])
			pred_sent = " ".join(sent_objs)

			if pred_sent not in pred_sent_set:
				pred_obj = {
					'sentence': pred_sent,
					'args': args,
					'tags': vb['tags'],
				}
				predicates.append(pred_obj)
				pred_sent_set.add(pred_sent)

	return predicates

def prune_graph(args, comet_model, chunk_predicates, chunk_id):

	"""
	Create the causal graph for a given text chunk
	@param text_chunk: List of sentences.
	"""

	folder_name = 'events_comet_out_{}'.format(args.sampling_algorithm)
	if not os.path.exists(os.path.join(args.data_dir, folder_name)):
		os.makedirs(os.path.join(args.data_dir, folder_name))
	elif os.path.exists(os.path.join(args.data_dir, folder_name, '{}.json'.format(chunk_id))):
		with open(os.path.join(args.data_dir, folder_name, '{}.json'.format(chunk_id)), 'rb') as f:
			comet_out_pruned = pickle.load(f)
			return comet_out_pruned

	print('Computing Pruned Comet Output Graph')

	text_chunk = [x['sentence'] for x in chunk_predicates]
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

def main(args):

	args = parser.parse_args()
	comet_model = atomic_predict.fetch_model(args)

	if args.sim_client == 'bert':
		bert_client = BertClient()
	elif args.sim_client == 'sbert':
		bert_client = SentenceTransformer('bert-base-nli-mean-tokens')

	openie_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")

	if args.sim_method == 'ne':
		from events_ne_sim import create_causal_graph, create_digraph
	elif args.sim_method == 'sent':
		from events_sent_sim import create_causal_graph, create_digraph

	# Set the results folder name based on the parameters
	res_folder = "events_" + args.sim_method + '_' + str(args.use_order) + '_' + args.sim_client

	with open(os.path.join(args.data_dir, 'text_chunks.json'), 'r') as f:
		text_chunks = simplejson.load(f)

	if not os.path.exists(os.path.join(args.data_dir, res_folder, 'causal_graph')):
		os.makedirs(os.path.join(args.data_dir, res_folder, 'causal_graph'))

	if not os.path.exists(os.path.join(args.data_dir, res_folder, 'digraphs')):
		os.makedirs(os.path.join(args.data_dir, res_folder, 'digraphs'))

	for chunk_id in text_chunks:
		print("Processing Chunk: ", chunk_id)
		chunk = text_chunks[chunk_id]

		# Fetch OpenIE events using AllenNLP library
		chunk_predicates = fetch_openie_events(args, openie_predictor, chunk)

		comet_out_pruned = prune_graph(args, comet_model, chunk_predicates, chunk_id)

		causal_graph = create_causal_graph(bert_client, chunk_predicates, comet_out_pruned, args.use_order)

		create_digraph(causal_graph, chunk_predicates, os.path.join(args.data_dir, res_folder, 'digraphs', str(chunk_id)))

		# Save the causal graph, pruned comet output
		with open(os.path.join(args.data_dir, res_folder, 'causal_graph', '{}.json'.format(chunk_id)), 'wb') as f:
			pickle.dump(causal_graph, f)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--model_file", type=str, required=True)
	parser.add_argument("--sampling_algorithm", type=str, default="beam-10")
	parser.add_argument('--data_dir', type=str, required=True, default="Directory to read the Dictionary of text chunks from")
	parser.add_argument('--sim_method', type=str, default='ne', help="The similarity method to compute the causal graph")
	parser.add_argument('--use_order', type=bool, default=True, help="Whether to use temporal order of sentences or not")
	parser.add_argument('--sim_client', type=str, default='sbert', help="Which BERT embedding to use (bert, sbert)")
	args = parser.parse_args()
	main(args)
