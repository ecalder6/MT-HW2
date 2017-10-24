Achintya Gopal
Eric Calder
Philip Piantone

Baseline
```bash
python decode.py -k 10 -s 100
```

Greedy algorithm with m best neighbors
```bash
python decode2.py -k 10 -s 100
```
Greedy with m best neighbors:
This command takes a while to run and produces our result on the leaderboard
```bash
python decode3.py -k 10 -s 100 -m 50
```

The difference between the second and third program is that the second will get
the top m hypotheses and allows for duplicates and the third will remove
duplicates allowing for a larger search space in the third.
