import sys
from typing import MutableMapping

from attr import attrs
import plotnine
import pandas as pd
from immutablecollections import ImmutableDict, ImmutableSet
from immutablecollections.immutablemultidict import ImmutableSetMultiDict
from plotnine import ggplot

from bilm.temp_prob import WordPair


@attrs(auto_attribs=True, hash=True, cmp=True)
class SentenceProb:
    word1: str
    word2: str
    template: str
    prob: float


sentence_probs = []
with open(sys.argv[1]) as data_in:
    for line in data_in:
        line = line.strip()
        if line:
            fields = line.strip().split("\t")
            if len(fields) != 5:
                raise RuntimeError(f"Bad line: {fields}")
            sentence_probs.append(SentenceProb(word1=fields[3],
                                               word2=fields[4],
                                               template=fields[2],
                                               prob=float(fields[0])))

raw_word_pairs = []
with open(sys.argv[2]) as extreme_pairs_file:
    for line in extreme_pairs_file:
        fields = line.split("\t")
        if len(fields) != 3:
            raise RuntimeError(f"Bad line: {fields}")
        raw_word_pairs.append(((fields[0], fields[1]), float(fields[2])))

word_pairs = ImmutableDict.of(raw_word_pairs)


def normalize_template(sent: str) -> str:
    return sent.replace('after', 'before')


sentence_probs_by_template_pair = ImmutableSetMultiDict.of(
    (normalize_template(sentence_prob.template), sentence_prob)
    for sentence_prob in sentence_probs
)

margins = {}

for (template, sentence_probs_for_template) in sentence_probs_by_template_pair.as_dict().items():
    before_probs = ImmutableDict.of(((x.word1, x.word2), x.prob)
                                    for x in sentence_probs_for_template
                                    if 'before' in x.template)
    after_probs = ImmutableDict.of(((x.word1, x.word2), x.prob)
                                   for x in sentence_probs_for_template
                                   if 'after' in x.template)
    common_keys = ImmutableSet.of(before_probs.keys()).intersection(
        ImmutableSet.of(after_probs.keys()))

    for word_pair in common_keys:
        margin = before_probs[word_pair] - after_probs[word_pair]
        margins[word_pair] = max(margins.get(word_pair, -100), margin)

    points = []
    for (word1, word2) in before_probs.keys():
        if (word1, word2) in word_pairs:
            points.append((word_pairs[(word1, word2)],
                           before_probs[(word1, word2)] - after_probs[(word1, word2)]))
    print(points)
    # data = pd.DataFrame.from_records(points, ["before_temp_prob", "before_lm_margin"])
    data = pd.DataFrame(points, columns=["before_temp_prob", "before_lm_margin"])
    plot = (ggplot(data, plotnine.aes(x='before_temp_prob', y="before_lm_margin")) +
            plotnine.geom_point() +
            plotnine.theme_minimal())
    filename = template.replace(" ", "_").replace(".", "") + ".png"
    print(f"Writing {filename}")
    plot.save(filename)

oracle_data = pd.DataFrame([(word_pairs[word_pair], margins[word_pair], str(word_pair))
               for word_pair in margins.keys()
               if word_pair in word_pairs],
columns=["before_temp_prob", "oracle_before_lm_margin", "word_pair"])


oracle_plot = (ggplot(oracle_data, plotnine.aes(x='before_temp_prob', y="oracle_before_lm_margin")) +
        plotnine.geom_point() +
        plotnine.geom_text(plotnine.aes(label="word_pair"), color="white", size="7") +
        plotnine.theme_minimal())
print(f"Writing oracle.png")
oracle_plot.save("oracle.png", width=6, height=6, dpi=400)

