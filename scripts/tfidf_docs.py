import operator

# Create query-doc relevance dict
robust04_dict = {}
with open('data/robust04/robust04_test.csv', mode='r') as robust04_file:
    lines = robust04_file.readlines()
    for line in lines:
        label, _, a, b, _, docid, _, sentid = line.rstrip().split('\t')
        robust04_dict[sentid] = [label, a, b, docid.split('_')[0]]

# Create query-doc TF-IDF dict
tfidf_dict = {}
with open('result/result.csv', mode='r') as scores_file, \
    open('data/tfidf_sents/tfidf_sents.csv', mode='w') as out_sents:
    scores = scores_file.readlines()
    for line in scores:
        qid, _, sentid, _, score, _, _ = line.split()
        if qid not in tfidf_dict.keys():
            tfidf_dict[qid] = {}
        docid = robust04_dict[sentid][-1]
        if docid not in tfidf_dict[qid].keys():
            tfidf_dict[qid][docid] = {}
        tfidf_dict[qid][docid][sentid] = float(score)
    for qid in tfidf_dict.keys():
        for docid in tfidf_dict[qid].keys():
            tfidf_dict[qid][docid] = sorted(tfidf_dict[qid][docid].items(),
                                     key=operator.itemgetter(1), reverse=True)
            # Retrieve relevant docs
            rel_sents = list(filter(lambda x: robust04_dict[x[0]][0] == '1',
                                    tfidf_dict[qid][docid]))
            if len(rel_sents) > 0:
                rel_sentid = rel_sents[0][0]
                robust04_entry = robust04_dict[rel_sentid]
                out_sents.write("{}\t{}\t{}\t{}\tQ0\t{}\t0\t0.0\tlucene4lm\turl\n".format(
                                int(robust04_entry[0]),
                                robust04_entry[1],
                                robust04_entry[2],
                                qid, rel_sentid))
            # Retrieve non-relevant docs
            non_rel_sents = list(filter(lambda x: robust04_dict[x[0]][0] == '0',
                                    tfidf_dict[qid][docid]))
            for i in range(0, min(len(non_rel_sents), 5)):
                non_rel_sentid = non_rel_sents[i][0]
                robust04_entry = robust04_dict[non_rel_sentid]
                out_sents.write("{}\t{}\t{}\t{}\tQ0\t{}\t0\t0.0\tlucene4lm\turl\n".format(
                                int(robust04_entry[0]),
                                robust04_entry[1],
                                robust04_entry[2],
                                qid, non_rel_sentid))