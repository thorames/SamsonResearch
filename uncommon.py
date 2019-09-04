# Thomas Horak (thorames)
# uncommon.py
import os
import re
import csv
import sys
import operator
from nltk import pos_tag
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stopwords = set(stopwords.words('english'))

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
        transcripts[file] = text

    return transcripts

def load_contractions():
    contractions = {}
    path = os.getcwd()

    with open(path + "/contractions.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        for row in reader:
            contractions[row[0].lower()] = row[1].lower()

    return contractions

def load_word_ranks():
    word_ranks = {}
    path = os.getcwd()

    with open(path + "/englishwords.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        count = 1
        for row in reader:
            word_ranks[row[0].lower()] = count
            count += 1

    return word_ranks

def load_filler_words():
    filler_words = []
    path = os.getcwd()

    with open(path + "/fillerwords.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        for row in reader:
            filler_words.append(row[0])

    return filler_words

def load_number_words():
    number_words = []
    path = os.getcwd()

    with open(path + "/numberwords.txt") as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')

        for row in reader:
            number_words.append(row[0])

    return number_words

def split_phrases(transcripts, contractions, good_POS, bad_filler, number_words, word_ranks):
    transcript_phrases = []
    transcript_tokens = []
    stemmer = PorterStemmer()

    for key, value in transcripts.items():
        value = re.sub('\r', ' ', value)
        value = re.sub(r'\n', ' @ ', value)
        value = re.sub(r'[^a-zA-Z\d\s\'\-]', ' @ ', value)
        value = value.lower()
        tokens = value.split()

        new_tokens = []
        for token in tokens:
            temp = re.search(r'\d', token)
            if temp != None:
                new_token = re.sub(r'[^\d]', '', token)
                new_token = num2words(new_token)
                new_token = re.sub(r'[^a-z]', ' ', new_token)
                if " " in new_token:
                    new_token = new_token.split()
                    new_tokens += new_token
                else:
                    new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        tokens = new_tokens

        new_tokens = []
        for token in tokens:
            if "--" in token:
                new_token = re.sub(r'[^a-zA-Z\d\']', '', token)
                if len(new_token) > 3 and new_token in word_ranks:
                    new_tokens.append(new_token)
            elif "-" in token:
                new_token = re.sub(r'[^a-zA-Z\d\s\']', ' ', token)
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
                token = re.sub(r'[^a-zA-Z\d\s]', ' @ ', token)
                token = token.split()
                new_tokens += token
        tokens = new_tokens

        tokens = pos_tag(tokens)

        phrase_list = []
        temp_list = []
        for token, pos in tokens:
            if pos in good_POS and token not in bad_filler and token not in number_words and token not in stopwords:
                temp_list.append(token)
            else:
                if len(temp_list) > 1:
                    temp_set = set(temp_list)
                    if len(temp_set) == len(temp_list):
                        bool = 1
                        for i in range(len(temp_list)):
                            for j in range(len(temp_list)):
                                if i != j:
                                    if temp_list[i] in temp_list[j]:
                                        bool = 0
                        if bool:
                            phrase_list.append("-".join(temp_list))
                            temp_list = []
                    else:
                        temp_list = []
                else:
                    temp_list = []

        transcript_phrases += phrase_list

        token_list = []
        for token, pos in tokens:
            if pos in good_POS and token not in bad_filler and token not in number_words and token not in stopwords:
                token_list.append(token)
            else:
                token_list.append("NULL")

        token_list.append("NULL")
        transcript_tokens += token_list

    tokens = transcript_tokens[:]

    transcript_phrases = set(transcript_phrases)
    transcript_phrases = list(transcript_phrases)
    transcript_phrases = sorted(transcript_phrases)

    temp_phrases = []
    count = 0
    while count < (len(transcript_phrases) - 1):
        if transcript_phrases[count] in transcript_phrases[count + 1]:
            jump = 1
            while transcript_phrases[count] in transcript_phrases[count + jump]:
                if (len(transcript_phrases[count]) + 1) == len(transcript_phrases[count + jump]):
                    jump += 1
                elif (len(transcript_phrases[count]) + 2) == len(transcript_phrases[count + jump]):
                    jump += 1
                else:
                    phrase_one = transcript_phrases[count].split("-")
                    phrase_two = transcript_phrases[count + jump].split("-")
                    if (len(phrase_one) + 1) < len(phrase_two):
                        jump += 1
                    else:
                        if pos_tag([phrase_two[-1]])[0][1] not in good_POS:
                            jump += 1
                        else:
                            temp_phrases.append(transcript_phrases[count + jump])
                            jump += 1
            temp_phrases.append(transcript_phrases[count])
            count += jump
        else:
            split_phrase = transcript_phrases[count].split("-")
            pos_tags = []
            for token in split_phrase:
                pair = pos_tag([token])
                pos_tags.append(pair[0])
            bool = 1
            for token, pos in pos_tags:
                if pos not in good_POS:
                    bool = 0
            if bool:
                temp_phrases.append(transcript_phrases[count])
            count += 1

    transcript_phrases = temp_phrases

    transcript_tokens = set(transcript_tokens)
    transcript_tokens = list(transcript_tokens)
    transcript_tokens = sorted(transcript_tokens)

    temp_tokens = []
    count = 0
    while count < (len(transcript_tokens) - 1):
        if transcript_tokens[count] in transcript_tokens[count + 1]:
            jump = 1
            while transcript_tokens[count] in transcript_tokens[count + jump]:
                if (len(transcript_tokens[count]) + 1) == len(transcript_tokens[count + jump]):
                    jump += 1
                else:
                    if pos_tag([transcript_tokens[count + jump]])[0][1] in good_POS:
                        temp_tokens.append(transcript_tokens[count + jump])
                        jump += 1
                    else:
                        jump += 1
            if pos_tag([transcript_tokens[count]])[0][1] in good_POS:
                temp_tokens.append(transcript_tokens[count])
                count += jump
            else:
                count += jump
        else:
            if pos_tag([transcript_tokens[count]])[0][1] in good_POS:
                temp_tokens.append(transcript_tokens[count])
                count += 1
            else:
                count += 1

    transcript_tokens = temp_tokens

    return transcript_phrases, transcript_tokens, tokens

def new_node(inlink_map, outlink_counts, node):
    if node not in inlink_map:
        inlink_map[node] = set()
    if node not in outlink_counts:
        outlink_counts[node] = 0

def build_graph(transcript_tokens):
    graph = []
    temp_graph = []

    for i in range(len(transcript_tokens) - 2):
        if transcript_tokens[i] != "NULL" and transcript_tokens[i + 1] != "NULL" and transcript_tokens[i + 2] != "NULL":
            temp_graph.append((transcript_tokens[i], transcript_tokens[i + 1]))
            temp_graph.append((transcript_tokens[i], transcript_tokens[i + 2]))
    graph = temp_graph

    inlink_map = {}
    outlink_counts = {}

    for tail_node, head_node in graph:
        new_node(inlink_map, outlink_counts, tail_node)
        new_node(inlink_map, outlink_counts, head_node)

        if tail_node not in inlink_map[head_node]:
            inlink_map[head_node].add(tail_node)
            outlink_counts[tail_node] += 1

    all_nodes = set(inlink_map.keys())
    for node, outlink_count in outlink_counts.items():
        if outlink_count == 0:
            outlink_counts[node] = len(all_nodes)
            for l_node in all_nodes:
                inlink_map[l_node].add(node)

    return inlink_map, outlink_counts, all_nodes

def text_rank(inlink_map, outlink_counts, all_nodes):
    initial_value = 0.25
    ranks = {}
    for node in inlink_map.keys():
        ranks[node] = initial_value

    new_ranks = {}
    delta = 1.0
    damping = 0.85
    n_iterations = 0
    convergence_threshold = 0.0001
    while delta > convergence_threshold:
        new_ranks = {}
        for node, inlinks in inlink_map.items():
            new_ranks[node] = ((1 - damping) / len(all_nodes)) + \
                              (damping * sum(ranks[inlink] / outlink_counts[inlink] for inlink in inlinks))
        delta = sum(abs(new_ranks[node] - ranks[node]) for node in new_ranks.keys())
        ranks = new_ranks
        new_ranks = ranks
        n_iterations += 1

    return ranks

def normalize_phrase_ranks(ranks, word_ranks, transcript_phrases, inputDirectory):
    transcript_phrases = [phrase.split("-") for phrase in transcript_phrases]

    phrase_ranks = {}
    for phrase in transcript_phrases:
        temp_rank = 0
        for token in phrase:
            if token in ranks and token in word_ranks:
                temp_rank += (ranks[token] * (float(float(word_ranks[token] / float(10000)))))
            elif token in ranks:
                temp_rank += (ranks[token] * 15)
                temp_rank += ranks[token]
            else:
                temp_rank = 0
                break
        phrase_ranks["-".join(phrase)] = float(float(temp_rank) / float(len(phrase)))

    sorted_phrase_ranks = sorted(phrase_ranks.items(), key=operator.itemgetter(1))
    sorted_phrase_ranks = reversed(sorted_phrase_ranks)
    sorted_phrase_ranks = list(sorted_phrase_ranks)
    sorted_phrase_ranks = sorted_phrase_ranks[:500]

    phrases = [phrase for phrase, rank in sorted_phrase_ranks]
    phrases = sorted(phrases)

    directory_path = inputDirectory.split("/")
    filename = (directory_path[-1] + " Transcript Uncommon Vocabulary (Phrases).txt")

    output_file = open(filename, "w+")
    for phrase in phrases:
        line = (phrase + "\n")
        output_file.write(line)

def main():
    inputDirectory = sys.argv[1]
    contractions = load_contractions()
    word_ranks = load_word_ranks()
    filler_words = load_filler_words()
    number_words = load_number_words()
    good_POS = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'POS']

    transcripts = read_transcripts(inputDirectory)
    transcript_phrases, transcript_tokens, tokens = split_phrases(transcripts, contractions, good_POS, filler_words, number_words, word_ranks)
    inlink_map, outlink_counts, all_nodes = build_graph(tokens)

    ranks = text_rank(inlink_map, outlink_counts, all_nodes)
    normalize_phrase_ranks(ranks, word_ranks, transcript_phrases, inputDirectory)
    #normalize_phrase_ranks(ranks, word_ranks, transcript_tokens, inputDirectory)

main()