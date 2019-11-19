import os
import json
import warnings
import pandas as pd
from collections import OrderedDict

PATH_data= '/Users/zhangzeyu/PycharmProjects/NER_WordTree/expl-tablestore-export-2019-09-10-165215'
PATH_OUTPUT = '/Users/zhangzeyu/PycharmProjects/NER_WordTree/output'

def divide_questionAndanswer(question_answer):
    qa_dict = {}
    if '(3)' in question_answer and '(4)' not in question_answer:
        qa_dict['question'] = question_answer.split('(1)')[0].strip()
        qa_dict['1'] = question_answer.split('(1)')[1].split('(2)')[0].strip()
        qa_dict['2'] = question_answer.split('(2)')[1].split('(3)')[0].strip()
        qa_dict['3'] = question_answer.split('(3)')[1].strip()
        qa_dict['4'] = None
        qa_dict['5'] = None
    elif '(1)' and '(2)' and '(3)' in question_answer:
        qa_dict['question'] = question_answer.split('(1)')[0].strip()
        qa_dict['1'] = question_answer.split('(1)')[1].split('(2)')[0].strip()
        qa_dict['2'] = question_answer.split('(2)')[1].split('(3)')[0].strip()
        qa_dict['3'] = question_answer.split('(3)')[1].split('(4)')[0].strip()
        qa_dict['4'] = question_answer.split('(4)')[1].strip()
        qa_dict['5'] = None
    elif '(D)' not in question_answer:
        qa_dict['question'] = question_answer.split('(A)')[0].strip()
        qa_dict['A'] = question_answer.split('(A)')[1].split('(B)')[0].strip()
        qa_dict['B'] = question_answer.split('(B)')[1].split('(C)')[0].strip()
        qa_dict['C'] = question_answer.split('(C)')[1].strip()
        qa_dict['D'] = None
        qa_dict['E'] = None
    elif '(E)' not in question_answer:
        qa_dict['question'] = question_answer.split('(A)')[0].strip()
        qa_dict['A'] = question_answer.split('(A)')[1].split('(B)')[0].strip()
        qa_dict['B'] = question_answer.split('(B)')[1].split('(C)')[0].strip()
        qa_dict['C'] = question_answer.split('(C)')[1].split('(D)')[0].strip()
        qa_dict['D'] = question_answer.split('(D)')[1].strip()
        qa_dict['E'] = None
    else:
        qa_dict['question'] = question_answer.split('(A)')[0].strip()
        qa_dict['A'] = question_answer.split('(A)')[1].split('(B)')[0].strip()
        qa_dict['B'] = question_answer.split('(B)')[1].split('(C)')[0].strip()
        qa_dict['C'] = question_answer.split('(C)')[1].split('(D)')[0].strip()
        qa_dict['D'] = question_answer.split('(D)')[1].split('(E)')[0].strip()
        qa_dict['E'] = question_answer.split('(E)')[1].strip()
    return qa_dict

def save_question_file(output_path, data_type, question_data):
    with open(os.path.join(output_path, 'questions_{}.json'.format(data_type)),'w+') as file:
        json.dump(question_data, file)
        print('{} set questions has been completed !'.format(data_type))

def save_table_file(output_path, table_data):
    with open(os.path.join(output_path, 'table_data.json'),'w+') as file:
        json.dump(table_data, file)
        print('table data  has been completed !')

def question_process(data_path, data_type):

    question_data_map = {}
    question_file = os.path.join(data_path, 'questions.tsv.{}.tsv'.format(data_type))
    df_q = pd.read_csv(question_file, sep='\t')
    #df_q_fileter = df_q[df_q['flags'].str.contains('SUCCESS', na=False)].copy()
    df_q['explanation_lenth'] = None
    df_q['explanation_lenth'] = df_q['explanation'].map(
        lambda y: len(list(OrderedDict.fromkeys(str(y).split(' ')).keys())))
    for _, row in df_q.iterrows():
        if 'SUCCESS' not in str(row['flags']).split(' '):
            continue
        qa_dict = divide_questionAndanswer(row['question'])
        question_ac = [qa_dict['question'].replace("''", '" ').replace("``", '" '), qa_dict[row['AnswerKey']].replace("''", '" ').replace("``", '" ')]
        question_data_map[row['QuestionID']] = question_ac
    print('{} set totally has {} questions'.format(data_type, len(question_data_map)))
    return question_data_map

def read_tsv(input_file):
    header = []
    uid = None
    df = pd.read_csv(input_file, sep='\t')
    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)
    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: '+input_file)
        return []
    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isnull(s))), 1).tolist()

def read_tsv_DEP(input_file):
    header = []
    uid = None
    df = pd.read_csv(input_file, sep='\t')
    if '[SKIP] DEP' in df.columns:
        df_filter = df[df['[SKIP] DEP'].notna()].copy()
        for name in df_filter.columns:
            if name.startswith('[SKIP]'):
                if 'UID' in name and not uid:
                    uid = name
            else:
                header.append(name)
        if not uid or len(df_filter) == 0:
            warnings.warn('Possibly misformatted file: ' + input_file)
            return []
        return df_filter.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isnull(s))), 1).tolist()
    else:
        warnings.warn('Possibly not contain DEP: ' + input_file)
        return []

def read_tsv_normal_words(input_file):
    header = []
    uid = None
    df = pd.read_csv(input_file, sep='\t')
    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        elif name.startswith('[FILL]'):
            continue
        else:
            header.append(name)
    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: '+input_file)
        return []
    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isnull(s))), 1).tolist()

def explanations_filtering(dict_explanations, dict_explanations_DEP, data_dir):
    explanations_id_list = []
    df_q_train = pd.read_csv(os.path.join(data_dir, 'questions.tsv.train.tsv'), sep='\t')
    df_q_dev = pd.read_csv(os.path.join(data_dir, 'questions.tsv.dev.tsv'), sep='\t')
    for _, row in df_q_train.iterrows():
        if 'SUCCESS' not in str(row['flags']).split(' '):
            continue
        for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
            explanations_id_list.append(single_row_id.split('|')[0])
    for _, row in df_q_dev.iterrows():
        if 'SUCCESS' not in str(row['flags']).split(' '):
            continue
        for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
            explanations_id_list.append(single_row_id.split('|')[0])
    removed_list = [item for item in dict_explanations_DEP if item not in explanations_id_list]
    for id in removed_list:
        del dict_explanations[id]
    #print('filtered dict explanations lenth: ', len(dict_explanations.keys()))
    return dict_explanations

def tablestore_process(data_dir):
    explanations = []
    explanations_DEP = []
    for path, _, files in os.walk(os.path.join(data_dir, 'tables')):
        for file in files:
            explanations += read_tsv(os.path.join(path, file))
            explanations_DEP += read_tsv_DEP(os.path.join(path, file))
    if not explanations:
        warnings.warn('Empty explanations')
    dict_explanations = {}
    dict_explanations_DEP = {}
    for item in explanations:
        dict_explanations[item[0]] = item[1]
    for item__ in explanations_DEP:
        dict_explanations_DEP[item__[0]] = item__[1]
    filtered_dict_explanations = explanations_filtering(dict_explanations, dict_explanations_DEP, data_dir)
    print('tablestore totally has {} sentences'.format(len(filtered_dict_explanations)))
    return filtered_dict_explanations

def main():
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)
    for data_type in ['train', 'dev', 'test']:
        question_data = question_process(PATH_data, data_type)
        save_question_file(PATH_OUTPUT, data_type, question_data)
    table_data = tablestore_process(PATH_data)
    save_table_file(PATH_OUTPUT, table_data)

if __name__ == '__main__':
    main()