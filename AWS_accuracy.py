# Thomas Horak (thorames)
# AWS_accuracy.py
import os
import re
import csv
import sys
import json
import tqdm as tqdm
from nltk.metrics import *
from num2words import num2words
from weighted_levenshtein import lev
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def load_filler_words():
    filler_words = []
    path = os.getcwd()

    with open(path + "/fillerwords.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        for row in reader:
            filler_words.append(row[0])

    return filler_words

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

def read_transcripts(input_directory):
    transcripts = {}

    input_files = [files for (path, dir, files) in os.walk(input_directory)]

    for file in input_files[0]:
        transcript_name = ""
        numbers = re.sub(r'[^\d]', ' ', file)
        numbers = numbers.split()
        for number in numbers:
            if len(number) == 6:
                transcript_name = number
                break

        if not transcript_name:
            continue

        file_path = (input_directory + '/' + file)
        if ".json" in file_path:
            with open(file_path) as json_file:
                document = json.load(json_file, strict=False)
                if not len(document["results"]["transcripts"][0]["transcript"]):
                    continue
                transcripts[transcript_name] = document["results"]["transcripts"][0]["transcript"]
        else:
            with open(file_path, encoding="ISO-8859-1") as txt_file:
                text = txt_file.read()
                if (re.search("{\"jobName", text)):
                    if not len(json_to_plain_txt(text)):
                        continue
                    transcripts[transcript_name] = json_to_plain_txt(text)
                else:
                    if not len(text):
                        continue
                    transcripts[transcript_name] = text

    return transcripts

def clean_transcriptions(transcripts, filler_words, contractions):
    clean_transcripts = {}

    for key, value in transcripts.items():
        value = re.sub(r'\s', ' ', value)
        value = re.sub(r'\s\'', ' ', value)
        value = re.sub(r'[^a-zA-Z\d\s\']', ' ', value)
        value = value.lower()
        tokens = value.split()

        new_tokens = []
        for token in tokens:
            if token in contractions:
                new_token = contractions[token]
                new_token = new_token.split()
                new_tokens += new_token
            elif "'s" == token[-2:]:
                new_token = token[:-2]
                new_token = re.sub(r'[^a-zA-Z\d\s]', '', new_token)
                new_tokens.append(new_token)
                new_tokens.append("is")
            elif "n't" == token[-3:]:
                new_token = token[:-3]
                new_token = re.sub(r'[^a-zA-Z\d\s]', '', new_token)
                new_tokens.append(new_token)
                new_tokens.append("not")
            elif "'ve" == token[-3:]:
                new_token = token[:-3]
                new_token = re.sub(r'[^a-zA-Z\d\s]', '', new_token)
                new_tokens.append(new_token)
                new_tokens.append("have")
            elif "'d" == token[-2:]:
                new_token = token[:-2]
                new_token = re.sub(r'[^a-zA-Z\d\s]', '', new_token)
                new_tokens.append(new_token)
                new_tokens.append("would")
            elif "'ll" == token[-3:]:
                new_token = token[:-3]
                new_token = re.sub(r'[^a-zA-Z\d\s]', '', new_token)
                new_tokens.append(new_token)
                new_tokens.append("will")
            elif "'re" == token[-3:]:
                new_token = token[:-3]
                new_token = re.sub(r'[^a-zA-Z\d\s]', '', new_token)
                new_tokens.append(new_token)
                new_tokens.append("are")
            else:
                token = re.sub(r'[^a-zA-Z\d\s]', ' ', token)
                token = token.split()
                new_tokens += token
        tokens = new_tokens

        new_tokens = []
        for token in tokens:
            temp = re.search(r'[^\d]', token)
            if temp == None:
                new_token = num2words(token)
                new_token = re.sub(r'[^a-z]', ' ', new_token)
                new_token = new_token.split()
                new_tokens += new_token
            else:
                new_tokens.append(token)
        tokens = new_tokens

        #new_tokens = []
        #for token in tokens:
            #if token not in filler_words:
                #new_tokens.append(token)
        #tokens = new_tokens

        clean_transcripts[key] = tokens

    return clean_transcripts

def calculate_levenshtein(gold_transcripts, silver_transcripts):
    average_accuracy = 0

    for key, value in tqdm.tqdm(gold_transcripts.items()):
        average_word_length = 0
        for token in value:
            average_word_length += len(token)
        average_word_length /= len(value)
        if key in silver_transcripts:
            distance = lev(" ".join(value), " ".join(silver_transcripts[key]))
            distance /= average_word_length
            error_rate = distance / len(value)
            accuracy = 1 - error_rate
            average_accuracy += accuracy
            print(key + " : " + str(accuracy))
    print("\n" + "Average Transcription Accuracy : " + str(average_accuracy / len(gold_transcripts)))

def calculate_WER(gold_transcripts, silver_transcripts):
    average_accuracy = 0

    for key, value in tqdm.tqdm(gold_transcripts.items()):
        if key in silver_transcripts:
            distance = edit_distance(value, silver_transcripts[key])
            error_rate = distance / len(value)
            accuracy = 1 - error_rate
            average_accuracy += accuracy
            print(key + " : " + str(accuracy))
    print("\n" + "Average Transcription Accuracy : " + str(average_accuracy / len(gold_transcripts)))

def main():
    gold_directory = sys.argv[1]
    silver_directory = sys.argv[2]
    filler_words = load_filler_words()
    contractions = load_contractions()

    gold_transcripts = read_transcripts(gold_directory)
    gold_transcripts = clean_transcriptions(gold_transcripts, filler_words, contractions)

    silver_transcripts = read_transcripts(silver_directory)
    silver_transcripts = clean_transcriptions(silver_transcripts, filler_words, contractions)

    #calculate_levenshtein(gold_transcripts, silver_transcripts)
    calculate_WER(gold_transcripts, silver_transcripts)

main()