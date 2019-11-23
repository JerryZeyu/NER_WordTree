# -*- coding: utf-8 -*-
import codecs
import glob
import json
import os
import spacy
import argparse
import pandas as pd
from collections import OrderedDict

numWarnings = 0
numProcessed = 0

def replace_unicode_whitespaces_with_ascii_whitespace(string):
    return ' '.join(string.split())

def get_start_and_end_offset_of_token_from_spacy(temp, token):
    start = temp + token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
    global numWarnings

    temp = 0
    warningOccured = False
    line_sentences = []
    for idx, line in enumerate(text.split('\n')[:-1]):
        sentences = []
        document = spacy_nlp(line)
        for span in document.sents:
            sentence = [document[i] for i in range(span.start, span.end)]
            sentence_tokens = []
            for token in sentence:
                token_dict = {}
                token_dict['pos'] = token.pos_
                token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(temp, token)
                token_dict['text'] = text[token_dict['start']:token_dict['end']]
                if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                    continue
                # Make sure that the token text does not contain any space
                if len(token_dict['text'].split(' ')) != 1:
                    print(
                        "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                            token_dict['text'],
                            token_dict['text'].replace(' ', '-')))
                    token_dict['text'] = token_dict['text'].replace(' ', '-')
                    warningOccured = True

                sentence_tokens.append(token_dict)
                if token == sentence[-1]:
                    temp = token_dict['end'] + 1
            temp_flag = temp
            temp = 0
            sentences.append(sentence_tokens)
        temp = temp_flag
        line_sentences.append(sentences)

    # If a warning occured in this file, then increment the warning count by 1
    if (warningOccured == True):
        numWarnings += 1
        # return empty sentences
        return []

    return line_sentences


def get_entities_from_brat(text_filepath, annotation_filepath):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text = f.read()
    # parse annotation file
    entities = []
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        for line in f.read().splitlines():
            anno = line.split()
            id_anno = anno[0]
            # parse entity
            if id_anno[0] == 'T':
                entity = OrderedDict()
                entity['id'] = id_anno
                entity['type'] = anno[1]
                entity['start'] = int(anno[2])
                entity['end'] = int(anno[3])
                entity['text'] = ' '.join(anno[4:])
                # Check compatibility between brat text and anootation
                if replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
                        replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
                    print("Warning: brat text and annotation do not match.")
                    print("\ttext: {0}".format(text[entity['start']:entity['end']]))
                    print("\tanno: {0}".format(entity['text']))
                entities.append(entity)
    return text, entities


def brat_to_conll(input_folder, output_filepath, questionid_list,  tokenizer, language):
    global numWarnings
    global numProcessed
    numSkippedQuestions = 0
    idSkippedQuestions = []
    if tokenizer == 'spacy':
        spacy_nlp = spacy.load(language, disable=["parser", "ner", "entity_linker", "textcat", "entity_ruler"])
        spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

    else:
        raise ValueError("tokenizer should be 'spacy'.")
    dataset_type = os.path.basename(input_folder)
    print("Formatting {0} set from BRAT to CONLL... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(input_folder, '*.txt')))

    numFiles = 0
    conll_ner_dict = OrderedDict()
    for text_filepath in text_filepaths:
        numProcessed += 1
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # create annotation file if it does not exist
        if not os.path.exists(annotation_filepath):
            codecs.open(annotation_filepath, 'w', 'UTF-8').close()
        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
        entities = sorted(entities, key=lambda entity: entity["start"])

        if tokenizer == 'spacy':
            #print(base_filename)
            line_sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)

            # If the number of sentences returned was zero, there was an error loading the annotation for this question.
            # In this case, we skip over this question.
            if (len(line_sentences) == 0):
                print ("Error loading annotation -- skipping this question. ")
                numSkippedQuestions += 1
                idSkippedQuestions.append(base_filename)
                continue
        single_conll_ner = OrderedDict()
        for q_sentences in line_sentences[0:1]:
            for q_sentence in q_sentences:
                for token in q_sentence:
                    token['label'] = []
                    for entity in entities:
                        if entity['start'] == token['start'] and token['end'] <= entity['end']:
                            token['label'].append('B-{0}'.format(entity['type'].replace('-', '_')))
                        elif entity['start'] < token['start'] and token['end'] <= entity['end']:
                            token['label'].append('I-{0}'.format(entity['type'].replace('-', '_')))
                        elif token['end'] < entity['start']:
                            break
                    if token['label'] == []:
                        token['label'] = ['O']
                    del token['start']
                    del token['end']
            single_conll_ner['question'] = q_sentences
        answer_sentences = line_sentences[1:]
        for sentences in answer_sentences:
            for sentence in sentences:
                for token in sentence:
                    token['label'] = []
                    for entity in entities:
                        if entity['start'] == token['start'] and token['end']<= entity['end']:
                            token['label'].append('B-{0}'.format(entity['type'].replace('-','_')))
                        elif entity['start'] < token['start'] and token['end']<= entity['end']:
                            token['label'].append('I-{0}'.format(entity['type'].replace('-','_')))
                        elif token['end'] < entity['start']:
                            break
                    if token['label'] == []:
                        token['label'] = ['O']
                    del token['start']
                    del token['end']
        single_conll_ner['answer'] = answer_sentences
        conll_ner_dict[base_filename] = single_conll_ner

        # Warning output
        numFiles += 1
        # if (numFiles % 10 == 0):
        #     print("numProcessed:" + str(numProcessed) + "\t Warnings: " + str(numWarnings))
    final_data = OrderedDict()
    for qid in questionid_list:
        final_data[qid] = conll_ner_dict[qid]
    print('json length: ', len(final_data))
    with open(output_filepath, 'w') as file_:
        json.dump(final_data, file_)
    print('Done.')
    print ("Number of skipped questions: " + str(numSkippedQuestions))
    print(idSkippedQuestions)
    if tokenizer == 'spacy':
        del spacy_nlp

def read_originalfile(original_data_dir, dataset_type):
    print(os.path.join(original_data_dir, 'questions.tsv.{}.tsv'.format(dataset_type.split('_')[0])))
    df_q = pd.read_csv(os.path.join(original_data_dir, 'questions.tsv.{}.tsv'.format(dataset_type.split('_')[0])),sep='\t')
    question_list = df_q['QuestionID'].tolist()
    return question_list

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--brat_ner_data_dir", default='./brat_data', type=str,
                        help="The brat format ner data dir which have been chunked and divided already")
    parser.add_argument("--original_data_dir", default='/home/zeyuzhang/expl-tablestore-export-2019-11-22-173702', type=str,
                        help="The brat format ner data dir which have been chunked and divided already")
    parser.add_argument("--conll_ner_data_output_dir", default='output', type=str,
                        help="The brat format ner data dir which have been chunked and divided already")
    args = parser.parse_args()

    print("Initializing...")
    print("brat_ner_data_dir: " + args.brat_ner_data_dir)

    if not os.path.exists(args.conll_ner_data_output_dir):
        os.makedirs(args.conll_ner_data_output_dir)
    tokenizer = 'spacy'
    for dataset_type in ['train_brat', 'dev_brat']:
        questionid_list = read_originalfile(args.original_data_dir, dataset_type)
        brat_to_conll(os.path.join(args.brat_ner_data_dir, dataset_type),
                      os.path.join(args.conll_ner_data_output_dir, 'questions_{}_ner.json'.format(dataset_type.split('_')[0])), questionid_list, tokenizer, 'en_core_web_sm')

    print("Warnings: " + str(numWarnings))

if __name__ == '__main__':
    main()