import pandas as pd
import csv
# tsv = pd.read_csv('data/news-commentary-v17.en-kk.tsv', delimiter='\t')
#
# list_en = tsv.values[:, 0].tolist()
# list_kk = tsv.values[:, 1].tolist()
#
# en = open('data/test.en', 'w', encoding='utf8', newline='')
# kk = open('data/test.kk', 'w', encoding='utf8', newline='')
#
# for i in range(len(list_en)):
#     line_en = str(list_en[i])
#     line_kk = str(list_kk[i])
#     line_en.strip().replace('\n\r', '')
#     line_kk.strip().replace('\n\r', '')
#     if (line_en and line_kk) and (line_en != 'nan' and line_kk != 'nan'):
#         en.write(line_en + '\n')
#         kk.write(line_kk + '\n')
#
# # en.write(str(list_en[-1]))
# # kk.write(str(list_kk[-1]))
# en.close()
# kk.close()

en1 = open('data/test.en.1', 'w', encoding='utf8', newline='')
kk1 = open('data/test.kk.1', 'w', encoding='utf8', newline='')
with open('data/news-commentary-v17.en-kk.tsv', 'r', encoding='utf8') as orig:
    lines = orig.readlines()
    for line in lines:
        line = line.strip().replace('\n', '')
        line = line.split('\t')
        if len(line) == 2:
            en1.write(line[0] + '\n')
            kk1.write(line[1] + '\n')
en1.close()
kk1.close()
