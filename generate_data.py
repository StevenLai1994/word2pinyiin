import os

from utils import get_dict 
from utils.changeTone import change_dict_tone
from utils.change_style import change_dict_style
import json

zi_pinyin_path = './generate_datas/py_dict.json'
phrase_path = './generate_datas/phrase_dict.json'

zi_path = './generate_datas/zi.json'
pinyin_path = './generate_datas/pinyin.json'

crf_data_path = './generate_datas/crf_train_data.data'

def load_json(path):
    with open(path, 'r', encoding='UTF-8') as fin:
        mdata = json.load(fin)
    return mdata

def write_json(path, data):
    with open(path, 'w', encoding='UTF-8') as fp:
        json.dump(data, fp, ensure_ascii=False)

def generate_dicts():
    pinyindic = get_dict.get_pinyins()
    phrasedic = get_dict.get_phrases()
    pinyindic = change_dict_style(pinyindic)
    phrasedic = change_dict_style(change_dict_tone(phrasedic))
    write_json(zi_pinyin_path, pinyindic)
    write_json(phrase_path, phrasedic)

def generate_zi_pinyin_set():
    zi = []
    pinyin = []
    phrase_dict = load_json(phrase_path)
    for key, val in phrase_dict.items():
        keys = list(key)
        for key in keys:
            if key not in zi:
                zi.append(key)
        val = val.replace(',', ' ')
        for py in val.split():
            if py not in pinyin:
                pinyin.append(py)
    write_json(zi_path, zi)
    write_json(pinyin_path, pinyin)

def generate_crf_train_data():
    phrase_dict = load_json(phrase_path)
    pinyin_table = load_json(pinyin_path)
    count = 0
    with open(crf_data_path, 'w', encoding='UTF-8') as fo:
        for key, val in phrase_dict.items():
            pys = val.split(',')[0].split()
            zis = list(key)
            for zi, py in zip(zis, pys):
                pyid = pinyin_table.index(py)
                new_line = '{}\t{}\n'.format(zi, pyid)
                fo.write(new_line)
            fo.write('\n')
            count += 1
            if count > 10000:
                break
    print("success")

if __name__ == '__main__':
    generate_crf_train_data()
    pass
