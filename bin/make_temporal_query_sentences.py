

#word_pairs = (
#    ('determined', 'accepted'),
#    ('asked', 'helped'),
#    ('attended', 'scheduled'),
#    ('accepted', 'proposed'),
#    ('died', 'exploded'),
#    ('chopped', 'tasted'),
#    ('concerned', 'protected'),
#    ('conspired', 'killed'),
#    ('debated', 'voted'),
#    ('dedicated', 'promoted'),
#    ('fought', 'overthrew'),
#    ('achieved', 'desired'),
#    ('admired', 'respected'),
#    ('cleaned', 'contaminated'),
#    ('defended', 'accused'),
#    ('died', 'crashed'),
#    ('overthrew', 'elected'),
#    ('involved', 'investigated'),
#    ('killed', 'investigated'),
#    ('suspected', 'investigated'),
#    ('stole', 'investigated'),
#    ('investigated', 'reported'),
#    ('investigated', 'prosecuted'),
#    ('investigated', 'paid'),
#    ('investigated', 'punished')
#)

#sentence_templates = {
#    'before-passive' : "It was {first} before it was {second} ,",
#    'before-passive-reversed' : "It was {second} before it was {first}",
#    'after-passive' : "It was {first} after it was {second}",
#    'after-passive-reversed': "It was {second} after it was {first}",
#    'before-active-he' : "He {first} before he {second}",
#    'before-active-he-reversed': "He {second} before he {first}",
#    'after-active-he' : "He {first} after he {second}",
#    'after-active-he-reversed': "He {second} after he {first}",
#    'before-active-double-transitive-he-it' : "He {first} it before he {second} it"
#}

#for (_, sentence_template) in sentence_templates.items():
#    for (first, second) in word_pairs:
#        print(sentence_template.format(first=first, second=second))
from sys import argv

word_pairs = []
with open(argv[1]) as extreme_pairs_file:
    for line in extreme_pairs_file:
        fields = line.split("\t")
        if len(fields) != 3:
            raise RuntimeError(f"Bad line: {fields}")
        word_pairs.append(((fields[0], fields[1]), float(fields[2])))

for subject in ('he', 'it'):
    for verb in ('{first}', 'was {first}'):
        for object in ([], ['him'], ['it']):
            for temporal_preposition in ('before', 'after'):
                for second_subject in ('he', 'it'):
                    for second_verb in ('{second}', 'was {second}'):
                        for second_object in ([], ['him'], ['it']):
                            tokens = [subject, verb]
                            tokens.extend(object)
                            tokens.extend([temporal_preposition, second_subject, second_verb])
                            tokens.extend(second_object)
                            template = " ".join(tokens)
                            for ((first, second), score) in word_pairs:
                                print("\t".join([template.format(first=first, second=second),
                                                template, first, second]))
                                print("\t".join([template.format(first=second, second=first),
                                                template, second, first]))

