import os
import argparse
import jsonlines
import torch
import pandas as pd
import numpy as np
from torch import tensor
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import matplotlib.pyplot as plt
from captum.concept import TCAV
from captum.concept._core.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
import torchtext 
from torchtext.vocab import Vocab
from captum.concept._utils.common import concepts_to_str


class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    
    def forward_func(self, inputs: tensor, position=0):
        """
        Wrapper around the prediction method of the pipeline
        """
        pred = self.__pipeline.model(inputs, attention_mask=torch.ones_like(inputs))
        return pred[position]

    #############TCAV##############
    def format_float(f):
        return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

    def visualize(self, experimental_sets, tcav_scores, outfile_path: str, layers = ['classifier'], score_type='sign_count'):
        fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

        barWidth = 1 / (len(experimental_sets[0]) + 1)

        for idx_es, concepts in enumerate(experimental_sets):
            concepts = experimental_sets[idx_es]
            concepts_key = concepts_to_str(concepts)
            
            layers = tcav_scores[concepts_key].keys()
            pos = [np.arange(len(layers))]
            for i in range(1, len(concepts)):
                pos.append([(x + barWidth) for x in pos[i-1]])
            _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
            for i in range(len(concepts)):
                val = [self.format_float(scores[score_type][i]) for layer, scores in tcav_scores[concepts_key].items()]
                _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

            # Add xticks on the middle of the group bars
            _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
            _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
            _ax.set_xticklabels(layers, fontsize=16)

            # Create legend & Show graphic
            _ax.legend(fontsize=16)
        plt.savefig(outfile_path)
        plt.show()

    def get_tensor_from_filename(filename):
        ds = torchtext.data.TabularDataset(path=filename,
                                        fields=[('text', torchtext.data.Field()),
                                                ('label', torchtext.data.Field())],
                                        format='csv')

    def assemble_concept(self, name, id, concepts_path="/uufs/chpc.utah.edu/common/home/u1413911/local_exp/hw4"):
        dataset = CustomIterableDataset(self.get_tensor_from_filename, concepts_path)
        concept_iter = dataset_to_dataloader(dataset, batch_size=1)
        return Concept(id=id, name=name, data_iter=concept_iter)

    def explain(self, text: str, concepts: list, outfile_path: str):
        """
        Implement TCAV evaluation for concepts and generate visualizations
        """
        # Convert the input text to tensor
        inputs = self.generate_inputs(text)
            
        neutral_concept = self.assemble_concept('neutral', 0, concepts_path="neutral.csv")
        neutral_concept2 = self.assemble_concept('neutral2', 1, concepts_path="neutral2.csv")
        neutral_concept3 = self.assemble_concept('neutral3', 2, concepts_path="neutral3.csv")
        neutral_concept4 = self.assemble_concept('neutral4', 3, concepts_path="neutral4.csv")
        neutral_concept5 = self.assemble_concept('neutral5', 4, concepts_path="neutral5.csv")

        positive_concept = self.assemble_concept('positive-adjectives', 5, concepts_path="positive-adjectives.csv")

        experimental_sets=[[positive_concept, neutral_concept],
                        [positive_concept, neutral_concept2],
                        [positive_concept, neutral_concept3],
                        [positive_concept, neutral_concept4],
                        [positive_concept, neutral_concept5]]

        # Create an instance of TCAV using the Concept object
        tcav = TCAV(model=self.__pipeline.model, 
                    input_tensors=inputs, 
                    internal_batch_size=16,  # Adjust the batch size as needed
                    attr_methods=['integrated_grad'], 
                    layers=['classifier'])  # Specify the layers
            
        # Calculate sensitivity scores
        sensitivity_scores = tcav.interpret(inputs, experimental_sets=experimental_sets)
        
        # Print or store the sensitivity scores as needed
        print(f"Sensitivity Scores for Concept {concepts}: {sensitivity_scores}")
        self.visualize(experimental_sets, sensitivity_scores, outfile_path + f"{self.__name}.png")

    def generate_inputs(self, text: str) -> tensor:
        """
        Convenience method for the generation of input ids as a list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device=self.__device).unsqueeze(0)
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
        Convenience method for the generation of a baseline vector as a list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device=self.__device).unsqueeze(0)

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

    idx = 0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], concepts, os.path.join(args.output_dir, f'example_{idx}'))
            output_file_path = os.path.join(args.output_dir, f'example_{idx}')
            print(f"Saving output to: {output_file_path}")
            idx += 1
            print(f"Example {idx} done")

if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_dir', default='/home/u1413911/local_exp/hw4/out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='incorrect_predictions.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()

    concepts = [
        ["good", "excellent", "wonderful", "adore", "great", "worth", "magnificent", "highlight", "fun", "quite well done"],
        ["pretty shallow","bad", "shockingly bad","terrible", "awful", "hate", "none had their best performances", "absolutely hate", "fall short","not very original", "sorry"]
    ]

    main(args)
