import optparse
import sys

def make_set(data, s, e_vocab, f_vocab, aligned, reverse):
    for pair in data.split():
        cur = pair.split('-')
        if reverse:
            e_vocab.add(int(cur[1]))
            f_vocab.add(int(cur[0]))
            aligned.add(int(cur[0]))
            s.add((int(cur[1]), int(cur[0])))
        else:
            e_vocab.add(int(cur[0]))
            f_vocab.add(int(cur[1]))
            aligned.add(int(cur[0]))
            s.add((int(cur[0]), int(cur[1])))

def grow_diag_final_and(e2f_data, f2e_data):
    directions = [(-1,0),(0,-1),(1,0),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for (i, (e2f, f2e)) in enumerate(zip(open(e2f_data), open(f2e_data))):
        e2f_set, f2e_set, e_vocab, f_vocab, e_aligned, f_aligned = set(), set(), set(), set(), set(), set()
        make_set(e2f, e2f_set, e_vocab, f_vocab, e_aligned, False)
        make_set(f2e, f2e_set, e_vocab, f_vocab, f_aligned, True)
        alignment = e2f_set & f2e_set
        union_alignment = e2f_set | f2e_set
        grow_diag(e_vocab, f_vocab, e_aligned, f_aligned, alignment, union_alignment, directions)
        final(e_vocab, f_vocab, e_aligned, f_aligned, alignment, union_alignment, True)

        for e, f in alignment:
            sys.stdout.write("%i-%i " % (e,f))
        sys.stdout.write("\n")

def grow_diag(e_vocab, f_vocab, e_alignment, f_alignment, alignment, union_alignment, directions):
    prev_len = 0
    while prev_len != len(alignment):
        prev_len = len(alignment)
        for e in e_vocab:
            for f in f_vocab:
                if (e, f) in alignment:
                    for d in directions:
                        en, fn = e + d[0], f + d[1]
                        if (en not in e_alignment or fn not in f_alignment) and (en, fn) in union_alignment:
                            alignment.add((en, fn))
                            e_alignment.add(en)
                            f_alignment.add(fn)

def final(e_vocab, f_vocab, e_alignment, f_alignment, alignment, union_alignment, final_and):
    for e in e_vocab:
        for f in f_vocab:
            c = False
            if final_and:
                c = e not in e_alignment and f not in f_alignment
            else:
                c = e not in e_alignment or f not in f_alignment
            if c and (e, f) in union_alignment:
                alignment.add((e, f))
                e_alignment.add(e)
                f_alignment.add(f)

def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/alignment", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--e2f", dest="e2f", default="ef", help="Suffix of English to French filename (default=ef)")
    optparser.add_option("-f", "--f2e", dest="f2e", default="fe", help="Suffix of French to English filename (default=fe)")
    optparser.add_option("-a", "--final_and", dest="final_and", action="store_true", help="Whether to use Final-And version of the algorithm")
    (opts, args) = optparser.parse_args()
    e2f_data = "%s.%s" % (opts.train, opts.e2f)
    f2e_data = "%s.%s" % (opts.train, opts.f2e)
    grow_diag_final_and(e2f_data, f2e_data)

if __name__ == "__main__":
    main()
