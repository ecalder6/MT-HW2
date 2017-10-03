Achintya Gopal
Eric Calder
Philip Piantone

IBM Model I 
```bash
python align.py -n 100000
```

Grow Diag Final (And)

You create the input to the grow diagonal code by running these two commands:

```bash
python align.py -n 1000 > data/1000_alignment.ef
python align.py -n 1000 -e f -f e > data/1000_alignment.fe
```
Then you can the create and score the grow diagonal alignments with this command:
```bash
python grow_diag_final.py --data data/1000_alignment | python score-alignments
```
You can also run the code with the And option:
```bash
python grow_diag_final.py --data data/1000_alignment --final_and | python score-alignments
```




Dirichlet Prior

```bash
python dirichlet.py -n 100000 
```
