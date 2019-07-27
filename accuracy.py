# Thomas Horak (thorames)
# accuracy.py
import os
import re
import csv
import sys
from num2words import num2words
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def load_word_ranks():
    word_ranks = {}
    path = os.getcwd()

    with open(path + "/englishwords.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        count = 1
        for row in reader:
            #if row[0].lower() not in stopwords:
            word_ranks[row[0].lower()] = count
            count += 1

    return word_ranks

def load_contractions():
    contractions = {}
    path = os.getcwd()

    with open(path + "/contractions.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        for row in reader:
            contractions[row[0].lower()] = row[1].lower()

    return contractions

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
        numbers = re.sub(r'[^\d]', ' ', file)
        numbers = numbers.split()
        for number in numbers:
            if len(number) == 6:
                transcripts[number] = text
                break
    return transcripts

def score_transcripts(transcripts, word_ranks, contractions):
    transcript_scores = {}

    for key, value in transcripts.iteritems():
        value = re.sub('\r', ' ', value)
        value = re.sub(r'[^a-zA-Z\d\s\']', ' ', value)
        value = value.lower()
        tokens = value.split()

        new_tokens = []
        for token in tokens:
            temp = re.search(r'\d', token)
            if temp != None:
                new_token = re.sub(r'[^\d]', '', token)
                new_token = num2words(new_token)
                new_token = re.sub(r'[^a-z]', ' ', new_token)
                new_token = new_token.split()
                new_tokens += new_token
            else:
                new_tokens.append(token)
        tokens = new_tokens

        new_tokens = []
        for token in tokens:
            if token in contractions:
                new_token = contractions[token]
                new_token = new_token.split()
                new_tokens += new_token
            else:
                token = re.sub(r'[^a-zA-Z\d\s]', ' ', token)
                token = token.split()
                new_tokens += token
        tokens = new_tokens

        #tokens = [t for t in tokens if t not in stopwords]

        transcript_score = 0
        for token in tokens:
            if token in word_ranks:
                transcript_score += word_ranks[token]

        transcript_scores[key] = transcript_score

    return transcript_scores

def accuracy(gold_standard_scores, silver_standard_scores):
    count = 0
    average_accuracy = 0

    for key, value in silver_standard_scores.iteritems():
        if key in gold_standard_scores:
            difference = abs(gold_standard_scores[key] - silver_standard_scores[key])
            average_accuracy += float(float(gold_standard_scores[key] - difference) / float(gold_standard_scores[key]))
            count += 1

    average_accuracy = float(float(average_accuracy) / float(count))
    print("Average Transcription Accuracy : " + str(average_accuracy))

def main():
    goldDirectory = sys.argv[1]
    silverDirectory = sys.argv[2]

    word_ranks = load_word_ranks()
    contractions = load_contractions()
    goldTranscripts = read_transcripts(goldDirectory)
    gold_standard_scores = score_transcripts(goldTranscripts, word_ranks, contractions)

    silverTranscripts = read_transcripts(silverDirectory)
    silver_standard_scores = score_transcripts(silverTranscripts, word_ranks, contractions)

    accuracy(gold_standard_scores, silver_standard_scores)

main()