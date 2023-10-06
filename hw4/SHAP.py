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
import shap

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
        
    def visualize(self, inputs: list, attributes: list, outfile_path: str):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        #import pdb; pdb.set_trace()
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        a = pd.Series(attr.cpu().numpy()[0][::-1], 
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1])
        '''
        a.plot.barh(figsize=(10,20))
        plt.savefig(outfile_path)        
        
        '''
        max_len = 130
        if len(a) > max_len:
            num_chunks = len(a) // max_len + 1 if len(a) % max_len != 0 else len(a) // max_len
            for i in range(num_chunks):
                start_idx = i * max_len
                end_idx = (i + 1) * max_len if (i + 1) * max_len < len(a) else len(a)
                chunk_a = a[start_idx:end_idx]
                chunk_a.plot.barh(figsize=(10, 20))
                outfile_path = f"{outfile_path}_{i+1}.png"
                plt.savefig(outfile_path)
                plt.clf()
        else:
            a.plot.barh(figsize=(10, 20))
            plt.savefig(outfile_path)
        
                    

    def explain_shap(self, text: str, outfile_path: str):
        # Convert the input text to input features (tokens)
        inputs = self.generate_inputs(text)

        # Create a background dataset (you can use random or empty data)
        background_dataset = torch.randn(10, inputs.shape[1])  # Adjust the size as needed

        # Create a SHAP explainer with the KernelExplainer
        explainer = shap.KernelExplainer(self.forward_func, background_dataset)

        # Compute SHAP values for the input
        shap_values = explainer.shap_values(inputs)

        # Plot the SHAP values
        shap.summary_plot(shap_values, inputs, feature_names=self.__pipeline.tokenizer.convert_ids_to_tokens(inputs[0]))

        # Save the plot
        plt.savefig(outfile_path)

    
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
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain_shap(obj["review"], os.path.join(args.output_dir,f'example_{idx}'))
            output_file_path = os.path.join(args.output_dir, f'example_{idx}')
            print(f"Saving output to: {output_file_path}")
            idx+=1
            print (f"Example {idx} done")


if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--analsis_dir', default='out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='incorrect_predictions.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()
    main(args)

    