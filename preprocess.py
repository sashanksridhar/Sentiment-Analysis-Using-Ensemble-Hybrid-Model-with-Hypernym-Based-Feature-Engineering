from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from os import walk
import nltk
import csv
stop_words = set(stopwords.words('english'))


def check_for_hypernim(token):
    hypernims = []
    for i in range(15):
        try:
            hypernims1 = []
            for i, j in enumerate(wn.synsets(token)):
                # hypernims1=list(chain(*[l.lemma_names() for l in j.hypernyms()]))
                for l in j.hypernyms():
                    hypernims1.append(l.lemma_names()[0])
                # print token
                # print(hypernims1)
                # print(hypernims1[0])
            token = hypernims1[0]
            hypernims.append(hypernims1)
        except IndexError:
            hypernims.append([token])

    return hypernims
for j in range(1,11):
    f0 = []
    for (dirpath, dirnames, filenames) in walk('E:\\sentimentanalysis\\test1\\'+str(j)):
        f0.extend(filenames)
        break
    count=1
    for f in f0:
        print(count)
        count+=1

        filename = f
        file1 = open('E:\\sentimentanalysis\\test1\\'+str(j)+'\\'+filename,encoding='latin1')
        line = file1.read()
        words = line.split()
        with open("E:\\sentimentanalysis\\test_csv\\"+str(j)+"\\"+filename.split('.')[0]+".csv", 'w',encoding='latin1') as csv_file:
            filewriter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            filtered_sentence = []

            x = []
            for r in words:

                if not r in stop_words:
                    # print(r)

                    filtered_sentence.append(r)
            tagged = nltk.pos_tag(filtered_sentence)
            for i in tagged:
                if len(i[0]) != 0 or len(i[0]) != 1:


                    if i[1] == 'IN' or i[1] == 'DT' or i[1] == 'CD' or i[1] == 'CC' or i[1] == 'EX' or i[1] == 'MD' or   i[1] == 'WDT' or i[1] == 'WP' or i[1] == 'UH' or i[1] == 'TO' or i[1] == 'RP' or i[1] == 'PDT' or i[1] == 'PRP' or i[1] == 'PRP$' or i[0] == 'co':
                        # print(i[0])
                        continue
                    else:

                        x.append(i[0].rstrip(".,?!"))
            print(x)
            for i in x:
                # print(i)
                l = []
                l.append(j)
                l.append(i)
                hype = check_for_hypernim(i)
                # print("hype")
                # print(i)
                # print(hype)
                if len(hype) == 0:
                    print("hi")
                    hype.append(i)  # 1
                    hype.append(i)  # 2
                    hype.append(i)  # 3
                    hype.append(i)  # 4
                    hype.append(i)  # 5
                    hype.append(i)  # 6
                    hype.append(i)  # 7
                    hype.append(i)  # 8
                    hype.append(i)  # 9
                    hype.append(i)  # 10
                    hype.append(i)  # 11
                    hype.append(i)  # 12
                    hype.append(i)  # 13
                    hype.append(i)  # 14
                    hype.append(i)  # 15
                for hyper in hype:
                    l.append(hyper[0])
                # print(l)

                filewriter.writerow(l)
            csv_file.close()



