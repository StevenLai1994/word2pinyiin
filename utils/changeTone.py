from ._constants import RE_PHONETIC_SYMBOL, PHONETIC_SYMBOL_DICT, SYMBOL_PHONETIC_DICT

def get_tone_number(pinyin):
    """找到单个拼音中拼音的音调"""
    value = RE_PHONETIC_SYMBOL.findall(pinyin)
    if len(value) == 0:
        return '0'
    symbol = value[0]
    return PHONETIC_SYMBOL_DICT[symbol][-1]

def change_one_tone(pinyin, new_tone):
    """改变单个拼音的音调"""
    # 替换拼音中的带声调字符
    value = RE_PHONETIC_SYMBOL.findall(pinyin)
    if len(value) == 0:
        return 0
    symbol = value[0]
    index = pinyin.find(symbol)
    new_symbol = PHONETIC_SYMBOL_DICT[symbol]
    new_symbol = new_symbol[:-1] + str(new_tone)
    pinyin = pinyin[:index] + SYMBOL_PHONETIC_DICT[new_symbol] + pinyin[index+1:]
    return pinyin

def change_dict_tone(phrases_dict):
    new_dict = {}
    prepost = ['第', '号', '月', '零','一', '二', '三', '四', '五', '六', '七', '八', '九', '十']

    for key, value in sorted(phrases_dict.items(), key=lambda x:x[1]):
        if '不' in key:
            ''' 不 变调'''
            index = key.find('不')
            if index + 1 < len(value):
                tone = get_tone_number(value[index+1][0])
                if tone == '4':
                    value[index][0] = 'bú'

        if '一' in key:
            ''' 一 变调'''
            index = key.find('一')
            if index + 1 < len(value):
                if key[index+1] not in prepost:
                    if index - 1 < 0 or (index - 1 >= 0 and key[index-1] not in prepost):
                        tone = get_tone_number(value[index+1][0])
                        if tone == '4':
                            value[index][0] = 'yí'
                        else:
                            value[index][0] = 'yì'
                    else:
                        value[index][0] = 'yī'
                else:
                    value[index][0] = 'yī'
        
        #三声连读变调
        tones = [get_tone_number(py[0]) for py in value]
        tones = ''.join(tones)
        if '33' in tones:
            index = tones.find('3')
            value[index][0] = change_one_tone(value[index][0], 2)
            pass

        new_dict[key] = value

    return new_dict