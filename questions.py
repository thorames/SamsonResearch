# Thomas Horak (thorames)
# questions.py
import os
import re
import csv
import sys
import math
import operator
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def prompt_student():
    print(chr(27) + "[2J")
    print("********************************************************************************")
    print("                            EXAM QUESTION GENERATOR                             ")
    print("********************************************************************************")
    print("Thanks for using the Exam Question Generator. This program allows you to enter  ")
    print("a specific phrase, keyword, or sentence from which potential exam questions will")
    print("be generated. These questions are selected from the provided database of exam   ")
    print("questions and will be written to two text files -- one file containing the exam ")
    print("questions (in order of relevance) and the other consisting of the corresponding ")
    print("answers to the questions. If no questions exist pertaining to the provided      ")
    print("topic, the user will be notified and prompted to enter a new phrase/keyword.    ")
    print("********************************************************************************")
    return input("\nPlease enter a Phrase/Word/Sentence : ")

def read_questions(questions_file):
    questions = {}
    choices = {}
    answers = {}
    path = os.getcwd()

    with open(path + "/" + questions_file) as csvin:
        reader = csv.reader(csvin, delimiter=',')

        for row in reader:
            questions[row[0]] = row[1]
            choices[row[0]] = row[2]
            answers[row[0]] = row[3]

    return questions, choices, answers

def clean_choices(choices):
    clean_choices = {}

    for key, value in choices.items():
        split_choices = re.split(r'\d\.\s', value)

        split_choices = [choice for choice in split_choices if choice]
        choice_list = [(chr(ord('A') + i) + ") " + split_choices[i]) for i in range(len(split_choices))]
        clean_choices[key] = choice_list

    return clean_choices

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

def index_question(content, inverted_index):
    content_tokens = tokenize_text(content)
    question_ID = content_tokens[0]
    content_tokens = content_tokens[1:]
    content_tokens = [token for token in content_tokens if token not in stopwords]
    content_tokens = stem_words(content_tokens)

    for token in content_tokens:
        if token not in inverted_index:
            inverted_index[token] = [1, {question_ID: 1}]
        else:
            if question_ID in inverted_index[token][1]:
                inverted_index[token][1][question_ID] += 1
            else:
                inverted_index[token][0] += 1
                inverted_index[token][1][question_ID] = 1

def question_length(inverted_index, num_questions):
    question_lengths = {}

    for token in inverted_index:
        for k, v in inverted_index[token][1].items():
            if k in question_lengths:
                question_lengths[k] += math.pow((inverted_index[token][1][k] * math.log10(float(float(num_questions) / float(inverted_index[token][0])))), 2)
            else:
                question_lengths[k] = math.pow((inverted_index[token][1][k] * math.log10(float(float(num_questions) / float(inverted_index[token][0])))), 2)

    for k, v in question_lengths.items():
        question_lengths[k] = math.sqrt(v)

    return question_lengths

def tfidf_questions(query_tokens, viable_questions, inverted_index, num_questions):
    weighted_questions = {}

    for question in viable_questions:
        weighted_questions[question] = []
        for token in query_tokens:
            if token in inverted_index:
                if question in inverted_index[token][1]:
                    weighted_questions[question].append(
                        inverted_index[token][1][question] * math.log10(float(float(num_questions) / float(inverted_index[token][0]))))
                else:
                    weighted_questions[question].append(0)

    return weighted_questions

def tfidf_query(query_tokens, inverted_index, num_questions):
    query_dictionary = {}
    query_vector = []

    for token in query_tokens:
        if token in query_dictionary:
            query_dictionary[token] += 1
        else:
            query_dictionary[token] = 1

    for token in query_tokens:
        if token in inverted_index:
            query_vector.append(query_dictionary[token] * math.log10(float(float(num_questions) / float(inverted_index[token][0] + 1))))

    return query_vector

def retrieve_questions(query, inverted_index, num_questions):
    weighted_questions = {}
    weighted_query = []

    query_tokens = tokenize_text(query)
    query_tokens = [token for token in query_tokens if token not in stopwords]
    query_tokens = stem_words(query_tokens)

    viable_questions = []
    for token in query_tokens:
        if token in inverted_index:
            for k, v in inverted_index[token][1].items():
                viable_questions.append(k)

    weighted_questions = tfidf_questions(query_tokens, viable_questions, inverted_index, num_questions)
    weighted_query = tfidf_query(query_tokens, inverted_index, num_questions)

    question_lengths = question_length(inverted_index, num_questions)

    inner_products = {}
    for question in viable_questions:
        product = 0
        for i in range(len(weighted_query)):
            product += (weighted_questions[question][i] * weighted_query[i])
        inner_products[question] = product

    cosine_similarity = {}
    for k, v in inner_products.items():
        query_length = 0
        for weight in weighted_query:
            query_length += (weight * weight)
        sqrt_query_weights = math.sqrt(query_length)

        if (sqrt_query_weights * question_lengths[k]) > 0:
            cosine_similarity[k] = float((float(v) / float((sqrt_query_weights * question_lengths[k]))))

    return cosine_similarity

def output_questions(sorted_cosine_similarity, questions, choices, answers):
    path = os.getcwd()

    question_file = open("questions.txt", "w+")
    answer_file = open("answers.txt", "w+")

    for i in range(len(sorted_cosine_similarity)):
        question_file.write(str(i + 1) + ") " + questions[sorted_cosine_similarity[i][0]] + "\n")

        if sorted_cosine_similarity[i][0] in choices:
            question_file.write("\n")
            for choice in choices[sorted_cosine_similarity[i][0]]:
                question_file.write(choice + "\n")
            question_file.write("\n")

        if re.search(r'\d\.', answers[sorted_cosine_similarity[i][0]][:2]):
            answer_file.write(str(i + 1) + ") " + answers[sorted_cosine_similarity[i][0]][3:] + "\n")
        else:
            answer_file.write(str(i + 1) + ") " + answers[sorted_cosine_similarity[i][0]] + "\n")
        answer_file.write("\n")

    print("\nYou can find these questions and their corresponding answers in \"questions.txt\" and \"answers.txt\".\n")

def main():
    inverted_index = {}
    question_ids = []
    questions_file = sys.argv[1]

    student_input = prompt_student()
    student_input = student_input.lower()

    questions, choices, answers = read_questions(questions_file)
    choices = clean_choices(choices)

    num_questions = len(questions)
    for key, value in questions.items():
        question_ids.append(key)
        content = (key + " " + value + " " + answers[key])
        index_question(content, inverted_index)

    cosine_similarity = retrieve_questions(student_input, inverted_index, num_questions)
    sorted_cosine_similarity = sorted(cosine_similarity.items(), key=operator.itemgetter(1))
    sorted_cosine_similarity = list(reversed(sorted_cosine_similarity))

    if len(sorted_cosine_similarity):
        print(chr(27) + "[2J")
        print("\nThe Exam Question Generator found " + str(len(sorted_cosine_similarity)) + " questions that match your query!")
    else:
        sorted_cosine_similarity = []
        while not len(sorted_cosine_similarity):
            print(chr(27) + "[2J")
            print("\nThere are no questions which match your query...")
            student_input = input("\nPlease enter a different Phrase/Word/Sentence : ")

            cosine_similarity = retrieve_questions(student_input, inverted_index, num_questions)
            sorted_cosine_similarity = sorted(cosine_similarity.items(), key=operator.itemgetter(1))
            sorted_cosine_similarity = list(reversed(sorted_cosine_similarity))

        print(chr(27) + "[2J")
        print("\nThe Exam Question Generator found " + str(len(sorted_cosine_similarity)) + " questions that match your query!")

    output_questions(sorted_cosine_similarity, questions, choices, answers)

main()
