import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

def fetch_model(args):

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    # Set the sampling algorithm
    sampling_algorithm = args.sampling_algorithm
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    return model, sampler, data_loader, text_encoder

def gen_preds(model_obj, text_chunk):
    model, sampler, data_loader, text_encoder = model_obj

    category_list = ['oEffect', 'xEffect', 'xNeed']

    atomic_output = {}
    for sent_idx, sentence in enumerate(text_chunk):
        atomic_output[sent_idx] = {}
        for category in category_list:
            outputs = interactive.get_atomic_sequence(sentence, model, sampler, data_loader, text_encoder, category)

            atomic_output[sent_idx][category] = outputs[category]

    return atomic_output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--sampling_algorithm", type=str, default="greedy")
    args = parser.parse_args()

    model_obj = fetch_model(args)
    gen_preds(model_obj, ['howdy how are you?'])
