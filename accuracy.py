# Thomas Horak (thorames)
# accuracy.py
import os
import re
import csv
import sys
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
from nltk.tokenize import RegexpTokenizer

def load_word_ranks():
    word_ranks = {}
    path = os.getcwd()

    with open(path + "/englishwords.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        count = 1
        for row in reader:
            if row[0].lower() not in stopwords:
                word_ranks[row[0].lower()] = count
                count += 1

    return word_ranks

def json_to_plain_txt(text):
    start = text.find("[{\"transcript\":\"") + 16
    end = text.find("\"}],\"items\":")
    return text[start:end]

def read_transcripts(inputDirectory):
    transcripts = {}

    path = os.getcwd()
    inputFiles = [files for (path, dir, files) in os.walk(inputDirectory)]

    for file in inputFiles[0]:
        document = open(path + '/' + file)
        text = document.read()
        if (re.search("{\"jobName", text)):
            text = json_to_plain_txt(text)
        transcripts[file] = text
    return transcripts

def score_gold_standard(transcripts, word_ranks):
    gold_standard_scores = {}

    for key, value in transcripts.iteritems():
        value = re.sub('\r', ' ', value)
        value = re.sub(r'[^a-zA-Z\d\s]', ' ', value)
        value = value.lower()
        tokens = value.split()
        tokens = [t for t in tokens if t not in stopwords]

        transcript_score = 0
        for token in tokens:
            if token in word_ranks:
                transcript_score += word_ranks[token]

        gold_standard_scores[key] = transcript_score

    return gold_standard_scores

def main():
    inputDirectory = sys.argv[1]

    word_ranks = load_word_ranks()
    transcripts = read_transcripts(inputDirectory)
    gold_standard_scores = score_gold_standard(transcripts, word_ranks)

    print(gold_standard_scores)

main()