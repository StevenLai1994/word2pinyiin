import os, sys
sys.path.append(os.getcwd())

out_phrases_path = 'origin_datas/phrases'
phrases_in_fs = [
                'overwrite.txt',
                'cc_cedict.txt',
                'pinyin.txt',
                'zdic_cybs.txt',
                'zdic_cibs.txt'
                ]

out_pinyins_path = 'origin_datas/pinyins'
pinyins_in_fs = [
                'overwrite.txt',
                'kHanyuPinyin.txt',
                'kMandarin.txt',
                'pinyin.txt'
                ]

def get_phrases():

    def remove_dup_items(lst):
        new_lst = []
        for item in lst:
            if item not in new_lst:
                new_lst.append(item)
        return new_lst

    phrases_dict = {}
    for fname in phrases_in_fs:
        path = os.path.join(out_phrases_path, fname)
        with open(path, 'r', encoding='UTF-8') as fp:
            for line in fp.readlines():
                line = line.strip()
                if line.startswith('#') or not line:
                    continue

                # 中国: zhōng guó
                data = line.split('#')[0]
                hanzi, pinyin = data.strip().split(':')
                hanzi = hanzi.strip()
                # [[zhōng], [guó]]
                pinyin_list = [[s] for s in pinyin.split()]

                if hanzi not in phrases_dict:
                    phrases_dict[hanzi] = pinyin_list
                else:
                    for index, value in enumerate(phrases_dict[hanzi]):
                        value.extend(pinyin_list[index])
                        phrases_dict[hanzi][index] = remove_dup_items(value)

    return phrases_dict

def get_pinyins():

    def remove_dup_items(pinyin, new_pinyin):
        pinyin = pinyin.split(',')
        new_pinyin = new_pinyin.split(',')
        for py in new_pinyin:
            if py not in pinyin:
                pinyin.append(py)
        return ','.join(pinyin)

    pinyin_dict = {}
    for fname in pinyins_in_fs:
        path = os.path.join(out_pinyins_path, fname)
        with open(path, 'r', encoding='UTF-8') as fp:
            for line in fp.readlines():
                line = line.strip()
                if line.startswith('#') or not line:
                    continue

                # U+3429: xíng  # 㐩
                data = line.split('#')[0]
                ucode, pinyin = data.split(':')
                ucode = ucode.strip().replace('U+', '0x')
                ucode = int(ucode, 16)
                pinyin = pinyin.strip()

                if ucode not in pinyin_dict:
                    pinyin_dict[ucode] = pinyin
                else:
                    pinyin_dict[ucode] = remove_dup_items(pinyin_dict[ucode], pinyin)
    
    return  pinyin_dict


if __name__ == '__main__':
    pinyins = get_pinyins()
    phrases = get_phrases()
    pass