Achintya Gopal
Eric Calder
Philip Piantone

Baseline
```bash
python decode.py -k 10 -s 100
```

Greedy with m best neighbors:
This command takes a while to run and produces our result on the leaderboard
```bash
python decode3.py -k 10 -s 100 -m 50
```

We also have a program called decode2.py that is our iteration of the greedy
algorithm with m best neighbors but we improved it and got decode3.py.
