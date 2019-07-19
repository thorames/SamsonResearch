# Thomas Horak (thorames)
# sentence_summarize.py
import os
import re
import sys
import math
import operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def normalize_sentence(sentence):
    tokenizer = RegexpTokenizer(r'\w+')

    sentence = sentence.lower()
    tokens = tokenizer.tokenize(sentence)
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

def split_sentences(transcripts):
    split_transcripts = {}
    for key, value in transcripts.iteritems():
        value = re.sub('\r', ' ', value)
        sentences = value.split(". ")
        split_transcripts[key] = []
        for sentence in sentences:
            if len(sentence) > 35:
                if (sentence[0] == ' '):
                    split_transcripts[key].append(sentence[1:])
                else:
                    split_transcripts[key].append(sentence)
    return split_transcripts

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

def sentence_similarity(sentence_one, sentence_two):
    sentence_one_tokens = set(normalize_sentence(sentence_one))
    sentence_two_tokens = set(normalize_sentence(sentence_two))
    if len(sentence_one_tokens) > 0 and len(sentence_two_tokens) > 0:
        return (len(sentence_one_tokens & sentence_two_tokens) / (math.log10(len(sentence_one_tokens)) + math.log10(len(sentence_two_tokens)) + 1))
    else:
        return 0

def text_rank(file_name, sentences):
    damping_factor = 0.85
    convergence_threshold = 0.0001
    initial_value = 0.25
    in_graph = {}
    out_graph = {}

    for sentence_one in sentences:
        for sentence_two in sentences:
            if sentence_one != sentence_two:
                score = sentence_similarity(sentence_one, sentence_two)
                if score > 1.5:
                    if sentence_one not in in_graph:
                        in_graph[sentence_one] = {}
                        in_graph[sentence_one][sentence_two] = score
                    else:
                        in_graph[sentence_one][sentence_two] = score
                    if sentence_one not in out_graph:
                        out_graph[sentence_one] = [score]
                    else:
                        out_graph[sentence_one].append(score)

    ranks = {}
    for vertex in in_graph.keys():
        ranks[vertex] = initial_value

    delta = 1.0
    n_iterations = 0
    while delta > convergence_threshold:
        updated_ranks = {}
        for vertex in ranks.keys():
            out_sum = sum(out_graph[vertex])
            in_sum_list = []
            for key, value in in_graph[vertex].iteritems():
                in_weight = in_graph[vertex][key]
                in_sum_list.append(((in_weight / out_sum) * ranks[key]))
            updated_ranks[vertex] = 1 - damping_factor + damping_factor * sum(in_sum_list)
        delta = sum(abs(updated_ranks[vertex] - ranks[vertex]) for vertex in updated_ranks.keys())
        ranks = updated_ranks
        n_iterations += 1

    sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1))
    sorted_ranks = reversed(sorted_ranks)
    sorted_ranks = list(sorted_ranks)

    sorted_sentences = {}
    for sentence in sorted_ranks[:10]:
        for j in range(len(sentences)):
            if sentence[0] == sentences[j]:
                sorted_sentences[sentence[0]] = j

    sorted_sentences = sorted(sorted_sentences.items(), key=operator.itemgetter(1))
    sorted_sentences = [(sentence[0] + ".") for sentence in sorted_sentences]
    summary = " ".join(sorted_sentences)

    output_file = open("SUMMARY_" + file_name, "w+")
    output_file.write(summary)

    print("Summary of " + file_name + " Completed!")

def main():
    inputDirectory = sys.argv[1]

    transcripts = read_transcripts(inputDirectory)
    transcripts = split_sentences(transcripts)

    for key, value in transcripts.iteritems():
        text_rank(key, value)

main()
