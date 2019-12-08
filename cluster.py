# Thomas Horak (thorames)
# cluster.py
import os
import re
import csv
import sys
import math
import operator
from nltk import pos_tag
from num2words import num2words
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def read_backchannel_questions(contents):
    questions = {}
    answers = {}

    with open(contents) as csvin:
        reader = csv.reader(csvin, delimiter=',')

        for row in reader:
            questions[row[0]] = row[1].lower()
            answers[row[0]] = row[10].lower()

    return questions, answers

def tokenize_text(text):
    text = text.lower()
    text = re.sub(r'\s\.\s', ' ', text)
    text = re.sub(r'[^a-zA-Z\d\s\'\-\.\,]', ' ', text)
    text = re.sub(r'\,\s', ' ', text)
    return word_tokenize(text)

def stem_words(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = []

    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens

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

def split_phrases(questions, contractions, good_POS, bad_filler, number_words, word_ranks):
    question_phrases = []
    question_tokens = []

    for key, value in questions.items():
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

        question_phrases += phrase_list

        token_list = []
        for token, pos in tokens:
            if pos in good_POS and token not in bad_filler and token not in number_words and token not in stopwords:
                token_list.append(token)
            else:
                token_list.append("NULL")

        token_list.append("NULL")
        question_tokens += token_list

    tokens = question_tokens[:]

    question_phrases = set(question_phrases)
    question_phrases = list(question_phrases)
    question_phrases = sorted(question_phrases)

    temp_phrases = []
    count = 0
    while count < (len(question_phrases) - 1):
        if question_phrases[count] in question_phrases[count + 1]:
            jump = 1
            while question_phrases[count] in question_phrases[count + jump]:
                if (len(question_phrases[count]) + 1) == len(question_phrases[count + jump]):
                    jump += 1
                elif (len(question_phrases[count]) + 2) == len(question_phrases[count + jump]):
                    jump += 1
                else:
                    phrase_one = question_phrases[count].split("-")
                    phrase_two = question_phrases[count + jump].split("-")
                    if (len(phrase_one) + 1) < len(phrase_two):
                        jump += 1
                    else:
                        if pos_tag([phrase_two[-1]])[0][1] not in good_POS:
                            jump += 1
                        else:
                            temp_phrases.append(question_phrases[count + jump])
                            jump += 1
            temp_phrases.append(question_phrases[count])
            count += jump
        else:
            split_phrase = question_phrases[count].split("-")
            pos_tags = []
            for token in split_phrase:
                pair = pos_tag([token])
                pos_tags.append(pair[0])
            bool = 1
            for token, pos in pos_tags:
                if pos not in good_POS:
                    bool = 0
            if bool:
                temp_phrases.append(question_phrases[count])
            count += 1

    question_phrases = temp_phrases

    question_tokens = set(question_tokens)
    question_tokens = list(question_tokens)
    question_tokens = sorted(question_tokens)

    temp_tokens = []
    count = 0
    while count < (len(question_tokens) - 1):
        if question_tokens[count] in question_tokens[count + 1]:
            jump = 1
            while question_tokens[count] in question_tokens[count + jump]:
                if (len(question_tokens[count]) + 1) == len(question_tokens[count + jump]):
                    jump += 1
                else:
                    if pos_tag([question_tokens[count + jump]])[0][1] in good_POS:
                        temp_tokens.append(question_tokens[count + jump])
                        jump += 1
                    else:
                        jump += 1
            if pos_tag([question_tokens[count]])[0][1] in good_POS:
                temp_tokens.append(question_tokens[count])
                count += jump
            else:
                count += jump
        else:
            if pos_tag([question_tokens[count]])[0][1] in good_POS:
                temp_tokens.append(question_tokens[count])
                count += 1
            else:
                count += 1

        question_tokens = temp_tokens

    return question_phrases, question_tokens, tokens

def new_node(inlink_map, outlink_counts, node):
    if node not in inlink_map:
        inlink_map[node] = set()
    if node not in outlink_counts:
        outlink_counts[node] = 0

def build_graph(question_tokens):
    graph = []
    temp_graph = []

    for i in range(len(question_tokens) - 2):
        if question_tokens[i] != "NULL" and question_tokens[i + 1] != "NULL" and question_tokens[i + 2] != "NULL":
            temp_graph.append((question_tokens[i], question_tokens[i + 1]))
            temp_graph.append((question_tokens[i], question_tokens[i + 2]))
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

def normalize_phrase_ranks(ranks, word_ranks, question_phrases):
    question_phrases = [phrase.split("-") for phrase in question_phrases]

    phrase_ranks = {}
    for phrase in question_phrases:
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

    phrases = [phrase for phrase, rank in sorted_phrase_ranks]
    phrases = sorted(phrases)

    phrase_counts = {}
    for phrase in phrases:
        tokens = phrase.split("-")

        token_split = ""
        for token in tokens:
            if token_split == "":
                token_split = token
            else:
                token_split += ("-" + token)
            if token_split in phrase and "-" in token_split:
                if token_split in phrase_counts:
                    phrase_counts[token_split] += 1
                else:
                    phrase_counts[token_split] = 1

    sorted_phrase_counts = sorted(phrase_counts.items(), key=operator.itemgetter(1))
    sorted_phrase_counts = reversed(sorted_phrase_counts)
    sorted_phrase_counts = list(sorted_phrase_counts)
    sorted_phrase_counts = [topic for topic, count in sorted_phrase_counts if count > 1]

    topics = {}
    for i in range(len(sorted_phrase_counts)):
        topics[i] = " ".join(sorted_phrase_counts[i].split("-"))

    return topics

def index_topic(content, inverted_index):
    content_tokens = tokenize_text(content)
    topic_ID = content_tokens[0]
    content_tokens = content_tokens[1:]
    content_tokens = [token for token in content_tokens if token not in stopwords]
    content_tokens = stem_words(content_tokens)

    for token in content_tokens:
        if token not in inverted_index:
            inverted_index[token] = [1, {topic_ID: 1}]
        else:
            if topic_ID in inverted_index[token][1]:
                inverted_index[token][1][topic_ID] += 1
            else:
                inverted_index[token][0] += 1
                inverted_index[token][1][topic_ID] = 1

def topic_length(inverted_index, num_topics):
    topic_lengths = {}

    for token in inverted_index:
        for k, v in inverted_index[token][1].items():
            if k in topic_lengths:
                topic_lengths[k] += math.pow((inverted_index[token][1][k] * math.log10(float(float(num_topics) / float(inverted_index[token][0])))), 2)
            else:
                topic_lengths[k] = math.pow((inverted_index[token][1][k] * math.log10(float(float(num_topics) / float(inverted_index[token][0])))), 2)

    for k, v in topic_lengths.items():
        topic_lengths[k] = math.sqrt(v)

    return topic_lengths

def tfidf_topics(question_tokens, viable_topics, inverted_index, num_topics):
    weighted_topics = {}

    for topic in viable_topics:
        weighted_topics[topic] = []
        for token in question_tokens:
            if token in inverted_index:
                if topic in inverted_index[token][1]:
                    weighted_topics[topic].append(
                        inverted_index[token][1][topic] * math.log10(float(float(num_topics) / float(inverted_index[token][0]))))
                else:
                    weighted_topics[topic].append(0)

    return weighted_topics

def tfidf_question(question_tokens, inverted_index, num_topics):
    question_dictionary = {}
    question_vector = []

    for token in question_tokens:
        if token in question_dictionary:
            question_dictionary[token] += 1
        else:
            question_dictionary[token] = 1

    for token in question_tokens:
        if token in inverted_index:
            question_vector.append(question_dictionary[token] * math.log10(float(float(num_topics) / float(inverted_index[token][0] + 1))))

    return question_vector

def retrieve_topics(question, inverted_index, num_topics):
    weighted_topics = {}
    weighted_question = []

    question_tokens = tokenize_text(question)
    question_tokens = [token for token in question_tokens if token not in stopwords]
    question_tokens = stem_words(question_tokens)

    viable_topics = []
    for token in question_tokens:
        if token in inverted_index:
            for k, v in inverted_index[token][1].items():
                viable_topics.append(k)

    weighted_topics = tfidf_topics(question_tokens, viable_topics, inverted_index, num_topics)
    weighted_question = tfidf_question(question_tokens, inverted_index, num_topics)

    topic_lengths = topic_length(inverted_index, num_topics)

    inner_products = {}
    for topic in viable_topics:
        product = 0
        for i in range(len(weighted_question)):
            product += (weighted_topics[topic][i] * weighted_question[i])
        inner_products[topic] = product

    cosine_similarity = {}
    for k, v in inner_products.items():
        question_length = 0
        for weight in weighted_question:
            question_length += (weight * weight)
        sqrt_question_weights = math.sqrt(question_length)

        if (sqrt_question_weights * topic_lengths[k]) > 0:
            cosine_similarity[k] = float((float(v) / float((sqrt_question_weights * topic_lengths[k]))))

    sorted_cosine_similarities = sorted(cosine_similarity.items(), key=operator.itemgetter(1))
    sorted_cosine_similarities = reversed(sorted_cosine_similarities)
    sorted_cosine_similarities = list(sorted_cosine_similarities)

    return sorted_cosine_similarities

def cluster_questions(questions, inverted_index, num_topics):
    clusters = {}
    for question_id, question in questions.items():
        sorted_cosine_similarities = retrieve_topics(question, inverted_index, num_topics)

        if len(sorted_cosine_similarities) > 0:
            max_score = sorted_cosine_similarities[0][1]
            for topic_id, score in sorted_cosine_similarities:
                if score == max_score:
                    if topic_id in clusters:
                        clusters[topic_id].append((question_id, score))
                    else:
                        clusters[topic_id] = [(question_id, score)]
        else:
            if str(num_topics) in clusters:
                clusters[str(num_topics)].append((question_id, 0))
            else:
                clusters[str(num_topics)] = [(question_id, 0)]

    return clusters

def output_clusters(clusters, questions, topics):
    if not os.path.exists("Topic_Clusters"):
        os.mkdir("Topic_Clusters")
    else:
        filelist = [file for file in os.listdir("Topic_Clusters") if file.endswith(".txt")]
        for file in filelist:
            os.remove(os.path.join("Topic_Clusters", file))

    for topic_id, question_ids in clusters.items():
        sorted_question_ids = sorted(question_ids, key=operator.itemgetter(1))
        sorted_question_ids = reversed(sorted_question_ids)
        sorted_question_ids = list(sorted_question_ids)

        with open(("Topic_Clusters/" + "_".join(topics[int(topic_id)].split()) + ".txt"), 'w') as outfile:
            for question_id, score in sorted_question_ids:
                outfile.write(question_id + "," + questions[question_id] + "\n")

def main():
    backchannel_file = sys.argv[1]
    contractions = load_contractions()
    word_ranks = load_word_ranks()
    filler_words = load_filler_words()
    number_words = load_number_words()
    good_POS = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'POS']
    inverted_index = {}

    questions, answers = read_backchannel_questions(backchannel_file)
    question_phrases, question_tokens, tokens = split_phrases(questions, contractions, good_POS, filler_words, number_words, word_ranks)
    inlink_map, outlink_counts, all_nodes = build_graph(tokens)

    ranks = text_rank(inlink_map, outlink_counts, all_nodes)
    topics = normalize_phrase_ranks(ranks, word_ranks, question_phrases)

    num_topics = len(topics)
    for key, value in topics.items():
        content = (str(key) + " " + value)
        index_topic(content, inverted_index)

    topics[len(topics)] = "miscellaneous"
    clusters = cluster_questions(questions, inverted_index, num_topics)

    output_clusters(clusters, questions, topics)

main()