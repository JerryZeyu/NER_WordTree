import os
import json
def create_questions(filepath, y_pred):
    with open(filepath, 'r') as file:
        f = json.load(file)
    data = []
    temp = 0
    for item in f.keys():
        for idx, question_sentence in enumerate(f[item]['question']):
            sentence = []
            label = []
            pred_idx = idx+temp
            for tokenid, token in enumerate(question_sentence):
                token['entity'] = y_pred[pred_idx][tokenid]
                sentence.append(token['text'])
                label.append(['O'])
            data.append((sentence,label))
        temp = pred_idx+1
        for idx_, answer_sentence in enumerate(f[item]['answer']):
            sentence = []
            label = []
            pred_idx_ = idx_ + temp
            for tokenid, token in enumerate(answer_sentence):
                token['entity'] = y_pred[pred_idx_][tokenid]
                sentence.append(token['text'])
                label.append(['O'])
            data.append((sentence, label))
        temp = pred_idx_+1
    assert temp == len(y_pred)
    assert len(data) == len(y_pred)
    return f

def create_tablestore(filepath, y_pred):
    with open(filepath, 'r') as file:
        f = json.load(file)
    data = []
    temp = 0
    for item in f.keys():
        for idx, table_sentence in enumerate(f[item]):
            sentence = []
            label = []
            pred_idx = idx + temp
            for tokenid, token in enumerate(table_sentence):
                token['entity'] = y_pred[pred_idx][tokenid]
                sentence.append(token['text'])
                label.append(['O'])
            data.append((sentence, label))
        temp = pred_idx + 1
    assert temp == len(y_pred)
    assert len(data) == len(y_pred)
    return f

def token_predition_write(conll_file, y_pred, set_type):
    if set_type == 'questions':
        final_data = create_questions(conll_file, y_pred)
    else:
        final_data = create_tablestore(conll_file, y_pred)
    with open(conll_file.split('.')[0]+'_ner.json', 'w') as file:
        json.dump(final_data, file)

    print('load {} prediction results into json format files completely'.format(os.path.splitext(os.path.basename(conll_file))[0]))