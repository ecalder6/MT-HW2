Welcome to DREAMT
=================
**Decoding, Reranking, Evaluation, and Alignment for Machine Translation**

DREAMT consists of baseline components for the open-ended assignments
posted at [the JHU MT classs website](http://mt-class.org/jhu). 

1) Word alignment.
2) Decoding.

python align.py -n 1000 > data/1000_alignment.ef
python align.py -n 1000 -e f -f e > data/1000_alignment.fe
python grow_diag_final.py --data data/1000_alignment --final_and | python score-alignments
