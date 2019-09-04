# Thomas Horak (thorames)
# unmatched.py
import os
import re
import csv
import sys
import operator
from num2words import num2words

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

    inputFiles = [files for (path, dir, files) in os.walk(inputDirectory)]

    for file in inputFiles[0]:
        document = open(inputDirectory + '/' + file, encoding="ISO-8859-1")
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

def word_counts(transcripts, contractions):
    word_counts = {}

    for key, value in transcripts.items():
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

        word_counts[key] = {}
        for token in tokens:
            if token in word_counts[key]:
                word_counts[key][token] += 1
            else:
                word_counts[key][token] = 1

    return word_counts

def unmatched(gold_word_counts, silver_word_counts, silver_directory):
    positive_word_differences = {}
    negative_word_differences = {}

    for transcription, word_counts in gold_word_counts.items():
        for token, count in word_counts.items():
            if token in silver_word_counts[transcription]:
                difference = count - silver_word_counts[transcription][token]
                if difference > 0:
                    if token in positive_word_differences:
                        positive_word_differences[token] += count - silver_word_counts[transcription][token]
                    else:
                        positive_word_differences[token] = count - silver_word_counts[transcription][token]
                elif difference < 0:
                    if token in negative_word_differences:
                        negative_word_differences[token] += count - silver_word_counts[transcription][token]
                    else:
                        negative_word_differences[token] = count - silver_word_counts[transcription][token]

    sorted_positive_word_differences = sorted(positive_word_differences.items(), key=operator.itemgetter(1))
    sorted_positive_word_differences = reversed(sorted_positive_word_differences)
    sorted_positive_word_differences = list(sorted_positive_word_differences)

    sorted_negative_word_differences = sorted(negative_word_differences.items(), key=operator.itemgetter(1))
    sorted_negative_word_differences = reversed(sorted_negative_word_differences)
    sorted_negative_word_differences = list(sorted_negative_word_differences)

    underused_count = len(sorted_positive_word_differences)
    overused_count = len(sorted_negative_word_differences)

    underused_magnitude = 0
    for pair in sorted_positive_word_differences:
        underused_magnitude += pair[1]

    overused_magnitude = 0
    for pair in sorted_negative_word_differences:
        overused_magnitude += abs(pair[1])

    directory_path = silver_directory.split("/")
    filename = (directory_path[-1] + " Unmatched Words.txt")

    output_file = open(filename, "w+")
    output_file.write("Number of Unique Unmatched Words : " + str(underused_count) + "\n")
    output_file.write("Number of Unique Overused Words : " + str(overused_count) + "\n")
    output_file.write("Total Unmatched Words : " + str(underused_magnitude) + "\n")
    output_file.write("Total Overused Words : " + str(overused_magnitude) + "\n")

    for pair in sorted_positive_word_differences:
        line = (pair[0] + " : " + str(pair[1]) + "\n")
        output_file.write(line)

    for pair in sorted_negative_word_differences:
        line = (pair[0] + " : " + str(pair[1]) + "\n")
        output_file.write(line)

def main():
    gold_directory = sys.argv[1]
    silver_directory = sys.argv[2]

    contractions = load_contractions()

    gold_transcripts = read_transcripts(gold_directory)
    gold_word_counts = word_counts(gold_transcripts, contractions)

    silver_transcripts = read_transcripts(silver_directory)
    silver_word_counts = word_counts(silver_transcripts, contractions)

    unmatched(gold_word_counts, silver_word_counts, silver_directory)

main()