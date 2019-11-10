import sys
import numpy as np
import math
f = open(sys.argv[1])

hi_count = 0.0
en_count = 0.0

language_span = []

total_count = 0.0

for line in f:
    tokens  = line.split()
    span_hi = 0
    span_en = 0
    prev = -1
    for token in tokens:
        total_count +=1
        t = 1
        try:
            token = token.encode("utf-8")
            t = 1
        except:
            t = 2
        if t == 2:
            hi_count +=1
            if prev == 0:
                span_hi += 1
            else:
                language_span.append(span_en)
                span_en = 0

            prev = 0
        else:
            en_count +=1
            if prev == 1:
                span_en += 1
            else:
                language_span.append(span_hi)
                span_hi = 0

            prev = 1


p_hi = hi_count / total_count

p_en = en_count / total_count

M_index = (1.0 - p_hi * p_hi - p_en * p_en) / (p_hi * p_hi + p_en * p_en)

language_span = np.array(language_span)

sd = np.std(language_span)
mean = np.mean(language_span)

B = (sd - mean) / (sd + mean)

unique, counts = np.unique(language_span, return_counts=True)

hist = list(np.asarray((unique, counts)).T)
print hist
total = len(language_span)
LE = 0.0
for (l, c) in hist:
    p_l = c*1.0 / total
    log_p = math.log(p_l)
    LE += p_l * log_p
LE *= -1

print "M_index", M_index, "B", B, "LE", LE
