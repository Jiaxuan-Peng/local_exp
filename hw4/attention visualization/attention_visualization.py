'''
Code source (with some changes):
https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234
https://gist.githubusercontent.com/theDestI/fe9ea0d89386cf00a12e60dd346f2109/raw/15c992f43ddecb0f0f857cea9f61cd22d59393ab/explain.py
'''
#/uufs/chpc.utah.edu/common/home/u1413911/micromamba/envs/hw/bin/python /uufs/chpc.utah.edu/common/home/u1413911/local_exp/hw3/assignment_3.py

import torch
import pandas as pd
import numpy as np

from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

import matplotlib.pyplot as plt

import argparse 
import jsonlines
import os 

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, tokenizer, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__tokenizer = tokenizer
        self.__device = device
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
        
       
    def explain(self, text: str, outfile_path: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        indices = inputs[0].detach().tolist()
        all_tokens = self.__tokenizer.convert_ids_to_tokens(indices)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        outputs = self.__pipeline.model(inputs, output_attentions = True)
        attentions = outputs['attentions']
        output_attentions_all = torch.stack(attentions)
        #print(output_attentions_all.shape)
        # Give a path to save
        #self.visualize(inputs, attributes, outfile_path)
        layer = 11

        scores_mat = output_attentions_all[layer].squeeze().detach().cpu().numpy()
        x_label_name='Head'
        fig = plt.figure(figsize=(20, 20))

        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(4, 3, idx+1)
            # append the attention weights
            im = ax.imshow(scores, cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(all_tokens)))
            ax.set_yticks(range(len(all_tokens)))

            ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(all_tokens, fontdict=fontdict)
            ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(outfile_path)
        plt.show()


    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device = self.__device).unsqueeze(0)
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device
                                )
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, tokenizer, device)

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], os.path.join(args.output_dir,f'example_{idx}'))
            output_file_path = os.path.join(args.output_dir, f'example_{idx}')
            print(f"Saving output to: {output_file_path}")
            idx+=1
            print (f"Example {idx} done")

if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--analsis_dir', default='/home/u1413911/local_exp/hw4/out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='incorrect_predictions.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='/home/u1413911/local_exp/hw4/out', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()
    main(args)