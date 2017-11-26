import nltk
import argparse

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--prediction_file", required=True,
                    help="Location to load the predictions.")
parser.add_argument("--reference_file", required=True,
                    help="Location to load the reference file.")
options = parser.parse_known_args()[0]
predictions = open(options.prediction_file, 'r').readlines()
references = open(options.reference_file, 'r').readlines()

print len(predictions), len(references)

print nltk.translate.bleu_score.corpus_bleu(references, predictions)
