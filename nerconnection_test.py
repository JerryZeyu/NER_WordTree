import os
import json
import pandas as pd
from collections import OrderedDict

def count_overlapped_number(qa_entities, sentence_entities):
    overlapped_number = len([item for item in sentence_entities if item in qa_entities])
    return overlapped_number

def read_top_sentences(filepath, top_number):
    df_data = pd.read_csv(filepath, sep='\t')
    sentences_id_list = []
    for _, row in list(df_data.iterrows())[0:top_number]:
        sentences_id_list.append(row['Row ID'])
    return sentences_id_list

def main():
    with open('/home/zeyuzhang/PycharmProjects/NER_WordTree/output/questions_dev_conll_ner.json', 'r') as file:
        dic_q_ner = json.load(file)
    with open('/home/zeyuzhang/PycharmProjects/NER_WordTree/output/table_data_conll_ner.json', 'r') as file_:
        dic_t_ner = json.load(file_)
    with open('/home/zeyuzhang/PycharmProjects/NER_WordTree/output/table_data_plain.json', 'r') as file__:
        dic_t_plain = json.load(file__)
    filepath = '/home/zeyuzhang/PycharmProjects/NER_WordTree/expl-tablestore-export-2019-09-10-165215/questions.tsv.dev.tsv'
    df_q = pd.read_csv(filepath, sep='\t')
    df_q_filter = df_q[df_q['flags'].str.contains('SUCCESS', na=False)]
    for idx_flag in [5,7,8,12,13,16,17,18,19]:
        for _, row in list(df_q_filter.iterrows())[idx_flag-1:idx_flag]:
            # if 'SUCCESS' not in str(row['flags']).split(' '):
            #     continue
            #print(row['QuestionID'])
            explanations_id_list = []
            for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
                explanations_id_list.append(single_row_id.split('|')[0])
            # print(len(explanations_id_list))
            question_tokens = []
            question_ner = []
            for sentence in dic_q_ner[row['QuestionID']]['question']:
                question_tokens.extend([token['text'] for token in sentence])
                question_ner.extend([token['entity'] for token in sentence])
            answer_tokens = []
            answer_ner = []
            for sentence in dic_q_ner[row['QuestionID']]['answer']:
                answer_tokens.extend([token['text'] for token in sentence])
                answer_ner.extend([token['entity'] for token in sentence])
            qa_ner = question_ner+answer_ner
            qa_ner_list = []
            for item in qa_ner:
                if item != ['O']:
                    qa_ner_list.extend(item)

            output_filepath = '/home/zeyuzhang/PycharmProjects/ER_BERT_without_partialknowledge/debug_output_3_epoches/question_{}.csv'.format(str(idx_flag))
            if idx_flag == 19:
                sentences_id_list = read_top_sentences(output_filepath, 1400)
            else:
                sentences_id_list = read_top_sentences(output_filepath, 100)
            print('{}\t{}\t{}\t{}\t{}'.format('Row ID', 'Sentence','Entities Overlapped Number', 'Original Ranking', 'Gold'))

            for idx, sentence_id in enumerate(sentences_id_list):
                sentence_tokens = [token['text'] for answer_sentence in dic_t_ner[sentence_id] for token in answer_sentence]
                sentence_ner = [item for answer_sentence in dic_t_ner[sentence_id] for token in
                                      answer_sentence for item in token['entity']]
                if sentence_id in explanations_id_list:
                    print('{}\t{}\t{}\t{}\t{}'.format(sentence_id, ' '.join(sentence_tokens), str(count_overlapped_number(qa_ner_list, sentence_ner)), str(idx+1), str(1)))
                else:
                    print('{}\t{}\t{}\t{}\t{}'.format(sentence_id, ' '.join(sentence_tokens), str(count_overlapped_number(qa_ner_list, sentence_ner)), str(idx+1), str(0)))

        print('\n')
        print('\n')
if __name__ == '__main__':
    main()