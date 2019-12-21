# Thomas Horak (thorames)
# lecture_questions.py
import os
import re
import csv
import sys
import json
import math
import tqdm
import random
import operator
from nltk import pos_tag
from datetime import datetime
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def json_to_plain_txt(text):
    start = text.find("[{\"transcript\":\"") + 16
    end = text.find("\"}],\"items\":")
    return text[start:end]

def read_transcripts(input_directory):
    transcripts = {}

    input_files = [files for (path, dir, files) in os.walk(input_directory)]

    for file in input_files[0]:
        if not file.startswith('.'):
            transcript_name = file.split(".")[0]

            file_path = (input_directory + '/' + file)
            if ".json" in file_path:
                with open(file_path) as json_file:
                    document = json.load(json_file, strict=False)
                    if not len(document["results"]["transcripts"][0]["transcript"]):
                        continue
                    transcripts[transcript_name] = document["results"]["transcripts"][0]["transcript"].lower()
            else:
                with open(file_path, encoding="ISO-8859-1") as txt_file:
                    text = txt_file.read()
                    if (re.search("{\"jobName", text)):
                        if not len(json_to_plain_txt(text)):
                            continue
                        transcripts[transcript_name] = json_to_plain_txt(text).lower()
                    else:
                        if not len(text):
                            continue
                        transcripts[transcript_name] = text.lower()

    return transcripts

def read_custom_vocabularies(input_directory):
    custom_vocabularies = {}

    input_files = [files for (path, dir, files) in os.walk(input_directory)]

    for file in input_files[0]:
        if not file.startswith('.'):
            transcript_name = file.split(".")[0]

            with open(input_directory + "/" + file) as tsvin:
                reader = csv.reader(tsvin, delimiter='\t')
                next(reader)

                custom_vocab = []
                for row in reader:
                    custom_vocab.append(row[0].lower())

                custom_vocabularies[transcript_name] = " ".join(custom_vocab)

    return custom_vocabularies

def condense_transcripts(transcripts):
    condensed_transcripts = {}

    for key, value in transcripts.items():
        word_counts = {}

        tokens = tokenize_text(value)

        new_tokens = []
        for token in tokens:
            if token.isalnum():
                new_tokens.append(token)
            else:
                new_tokens.append("NONE")
        tokens = new_tokens

        new_tokens = []
        for token in tokens:
            if token not in stopwords:
                new_tokens.append(token)
            else:
                new_tokens.append("NONE")
        tokens = new_tokens

        ngrams = []
        ngram = []
        for token in tokens:
            if token != "NONE":
                ngram.append(token)
            else:
                if len(ngram) > 1:
                    ngrams.append("_".join(ngram))
                    ngram = []
                else:
                    ngram = []
        tokens = ngrams

        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

        sorted_word_counts = sorted(word_counts.items(), key=operator.itemgetter(1))
        sorted_word_counts = list(reversed(sorted_word_counts))
        sorted_word_counts = sorted_word_counts[:50]

        keyword_list = []
        for pair in sorted_word_counts:
            for token in pair[0].split("_"):
                keyword_list.append(token)

        keyword_counts = {}
        for keyword in keyword_list:
            if keyword in keyword_counts:
                keyword_counts[keyword] += 1
            else:
                keyword_counts[keyword] = 1

        sorted_keyword_counts = sorted(keyword_counts.items(), key=operator.itemgetter(1))
        sorted_keyword_counts = list(reversed(sorted_keyword_counts))
        sorted_keyword_counts = sorted_keyword_counts[:50]

        keyword_list = []
        for pair in sorted_keyword_counts:
            keyword_list.append(pair[0])

        condensed_transcripts[key] = " ".join(keyword_list)

    return condensed_transcripts

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
            answers[row[0]] = row[3].rstrip()

    return questions, choices, answers

def clean_choices(choices):
    clean_choices = {}

    for key, value in choices.items():
        split_choices = re.split(r'\d\.\s', value)

        split_choices = [choice for choice in split_choices if choice]
        choice_list = [str(chr(ord('A') + i) + ") " + split_choices[i]) for i in range(len(split_choices))]
        choice_list = [choice.rstrip() for choice in choice_list]
        clean_choices[key] = choice_list

    return clean_choices

def clean_answers(choices, answers):
    clean_answers = {}

    for key, value in answers.items():
        if len(choices[key]):
            option_numbers = re.findall(r'\d\.\s', value)

            num_to_char = {}
            if len(option_numbers) > 1:
                for option_number in option_numbers:
                    num_to_char[option_number] = str(chr(ord('A') + (int(option_number[0]) - 1)) + ") ")

            elif len(option_numbers):
                num_to_char[option_numbers[0]] = str(chr(ord('A') + (int(option_numbers[0][0]) - 1)) + ") ")

            for num, char in num_to_char.items():
                value = value.replace(num, char, 1)

        clean_answers[key] = value

    return clean_answers

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

def tfidf_questions(transcript_tokens, viable_questions, inverted_index, num_questions):
    weighted_questions = {}

    for question in tqdm.tqdm(viable_questions):
        weighted_questions[question] = []
        for token in transcript_tokens:
            if token in inverted_index:
                if question in inverted_index[token][1]:
                    weighted_questions[question].append(
                        inverted_index[token][1][question] * math.log10(float(float(num_questions) / float(inverted_index[token][0]))))
                else:
                    weighted_questions[question].append(0)

    return weighted_questions

def tfidf_transcript(transcript_tokens, inverted_index, num_questions):
    transcript_dictionary = {}
    transcript_vector = []

    for token in transcript_tokens:
        if token in transcript_dictionary:
            transcript_dictionary[token] += 1
        else:
            transcript_dictionary[token] = 1

    for token in transcript_tokens:
        if token in inverted_index:
            transcript_vector.append(transcript_dictionary[token] * math.log10(float(float(num_questions) / float(inverted_index[token][0] + 1))))

    return transcript_vector

def retrieve_questions(transcripts, class_name, inverted_index, num_questions, questions, choices, answers):
    question_counts = {}

    if not os.path.exists("Lecture_Questions"):
        os.mkdir("Lecture_Questions")
    else:
        filelist = [file for file in os.listdir("Lecture_Questions") if file.endswith(".json")]
        for file in filelist:
            os.remove(os.path.join("Lecture_Questions", file))

    for key, value in transcripts.items():
        weighted_questions = {}
        weighted_transcript = []

        print("\n" + str(key))

        transcript_tokens = tokenize_text(value)
        transcript_tokens = [token for token in transcript_tokens if token not in stopwords]
        transcript_tokens = stem_words(transcript_tokens)

        viable_questions = []
        for token in transcript_tokens:
            if token in inverted_index:
                for k, v in inverted_index[token][1].items():
                    viable_questions.append(k)

        weighted_questions = tfidf_questions(transcript_tokens, viable_questions, inverted_index, num_questions)
        weighted_transcript = tfidf_transcript(transcript_tokens, inverted_index, num_questions)

        question_lengths = question_length(inverted_index, num_questions)

        inner_products = {}
        for question in tqdm.tqdm(viable_questions):
            product = 0
            for i in range(len(weighted_transcript)):
                product += (weighted_questions[question][i] * weighted_transcript[i])
            inner_products[question] = product

        cosine_similarity = {}
        for k, v in inner_products.items():
            transcript_length = 0
            for weight in weighted_transcript:
                transcript_length += (weight * weight)
            sqrt_transcript_weights = math.sqrt(transcript_length)

            if (sqrt_transcript_weights * question_lengths[k]) > 0:
                cosine_similarity[k] = float((float(v) / float((sqrt_transcript_weights * question_lengths[k]))))

        sorted_cosine_similarity = sorted(cosine_similarity.items(), key=operator.itemgetter(1))
        sorted_cosine_similarity = list(reversed(sorted_cosine_similarity))
        sorted_cosine_similarity = sorted_cosine_similarity[:10]

        for pair in sorted_cosine_similarity:
            if pair[0] in question_counts:
                question_counts[pair[0]] += 1
            else:
                question_counts[pair[0]] = 1

        counts_file = open("QuestionCounts.txt", "w+")

        sorted_question_counts = sorted(question_counts.items(), key=operator.itemgetter(1))
        sorted_question_counts = list(reversed(sorted_question_counts))

        total_question_count = 0
        unique_question_count = 0
        for pair in sorted_question_counts:
            total_question_count += pair[1]
            unique_question_count += 1
        counts_file.write("Average Question Frequency : " + str(total_question_count / unique_question_count) + "\n")

        for pair in sorted_question_counts:
            counts_file.write(str(pair[0]) + " : " + str(pair[1]) + "\n")

        counts_file.close()

        new_output_questions(class_name, key, sorted_cosine_similarity, questions, choices, answers)
        #old_output_questions(key, sorted_cosine_similarity, questions, choices, answers)

def new_output_questions(class_name, key, sorted_cosine_similarity, questions, choices, answers):
    json_output = {}

    json_output["TableName"] = "VirtualGSI_QA"
    json_output["Item"] = {}

    json_output["Item"]["CourseID"] = {}
    json_output["Item"]["CourseID"]["S"] = class_name

    now = datetime.now()
    json_output["Item"]["Date"] = {}
    json_output["Item"]["Date"]["S"] = now.strftime("%Y-%m-%d")

    json_output["Item"]["listOfQuestions"] = {}
    json_output["Item"]["listOfQuestions"]["L"] = []

    for i in range(len(sorted_cosine_similarity)):
        question = {}
        question["M"] = {}

        question["M"]["QuestionID"] = {}
        question["M"]["QuestionID"]["S"] = sorted_cosine_similarity[i][0]

        question["M"]["Question"] = {}
        question["M"]["Question"]["S"] = questions[sorted_cosine_similarity[i][0]]

        question["M"]["Answer"] = {}
        if len(answers[sorted_cosine_similarity[i][0]]):
            question["M"]["Answer"]["S"] = answers[sorted_cosine_similarity[i][0]]
        else:
            question["M"]["Answer"]["S"] = "(NO ANSWER PROVIDED)"

        question["M"]["Choices"] = {}
        question["M"]["Choices"]["L"] = []
        if len(choices[sorted_cosine_similarity[i][0]]):
            for j in range(len(choices[sorted_cosine_similarity[i][0]])):
                choice = {}
                choice["S"] = choices[sorted_cosine_similarity[i][0]][j]
                question["M"]["Choices"]["L"].append(choice)

        json_output["Item"]["listOfQuestions"]["L"].append(question)

    with open("Lecture_Questions/" + str(key) + "_questions.json", 'w') as outfile:
        json.dump(json_output, outfile)

def old_output_questions(key, sorted_cosine_similarity, questions, choices, answers):
    json_output = {}
    json_output["assignmentTitle"] = key

    now = datetime.now()
    json_output["openDate"] = now.strftime("%m%d%Y %H:%M")

    json_output["closeDate"] = ''

    json_output["questions"] = []

    for i in range(len(sorted_cosine_similarity)):
        question = {}
        question["title"] = ("Question " + str(i + 1) + " for Class " + key)
        question["main"] = questions[sorted_cosine_similarity[i][0]]

        if len(answers[sorted_cosine_similarity[i][0]]):
            question["answer"] = answers[sorted_cosine_similarity[i][0]]
        else:
            question["answer"] = "(NO ANSWER PROVIDED)"

        question["mcq"] = []

        if len(choices[sorted_cosine_similarity[i][0]]):
            for j in range(len(choices[sorted_cosine_similarity[i][0]])):
                choice = {}

                choice["option"] = choices[sorted_cosine_similarity[i][0]][j]
                if choices[sorted_cosine_similarity[i][0]][j] in answers[sorted_cosine_similarity[i][0]]:
                    choice["isAnswer"] = "true"
                else:
                    choice["isAnswer"] = "false"

                choice["order"] = (j + 1)

                question["mcq"].append(choice)

        if (len(re.findall(r'\d\.\s', answers[sorted_cosine_similarity[i][0]])) > 1) and (choices[sorted_cosine_similarity[i][0]]):
            question["multiselectMCQ"] = "true"
        else:
            question["multiselectMCQ"] = "false"

        question["points"] = 1.0

        json_output["questions"].append(question)

    with open("Lecture_Questions/" + str(key) + "_questions.json", 'w') as outfile:
        json.dump(json_output, outfile)

def main():
    inverted_index = {}
    question_ids = []
    questions_file = sys.argv[1]
    input_directory = sys.argv[2]
    class_name = questions_file.split("_")[0]

    transcripts = read_transcripts(input_directory)
    transcripts = condense_transcripts(transcripts)
    #custom_vocabularies = read_custom_vocabularies(input_directory)

    questions, choices, answers = read_questions(questions_file)
    choices = clean_choices(choices)
    answers = clean_answers(choices, answers)

    num_questions = len(questions)
    for key, value in questions.items():
        question_ids.append(key)
        content = (key + " " + value + " " + answers[key])
        index_question(content, inverted_index)

    print("\nGenerating Questions For Lecture...")

    retrieve_questions(transcripts, class_name, inverted_index, num_questions, questions, choices, answers)
    #retrieve_questions(custom_vocabularies, class_name, inverted_index, num_questions, questions, choices, answers)

main()