import sys

from bilm.temp_prob import TempProb

with open(sys.argv[1]) as temp_prob_file:
    temp_prob = TempProb.from_file(temp_prob_file, smoothing_pseudo_count=5)

pairs_with_before_and_after = temp_prob.seen_as_before.intersection(temp_prob.seen_as_after)

for (verb1, verb2) in sorted(pairs_with_before_and_after):
    percent_before = temp_prob.smoothed_fraction_before(verb1, verb2)
    if percent_before <= 0.05 or percent_before >= 0.95:
        print("\t".join((verb1, verb2, str(percent_before))))
