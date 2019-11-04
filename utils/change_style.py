import os
from ._constants import RE_PHONETIC_SYMBOL, PHONETIC_SYMBOL_DICT

from . import get_dict


def move_number(str, five=True):
    for i, c in enumerate(str):
        if ord('1') <= ord(c) <= ord('4'):
            return str[:i] + str[i+1:] + str[i]
    if five:
        str += '5'
    return str

def replace_symbol_to_number(pinyin):
    """把声调替换为数字"""
    def _replace(match):
        symbol = match.group(0)  # 带声调的字符
        # 返回使用数字标识声调的字符
        return PHONETIC_SYMBOL_DICT[symbol]

    # 替换拼音中的带声调字符
    value = RE_PHONETIC_SYMBOL.sub(_replace, pinyin)
    return value

def change_dict_style(pydict, five=True):
    new_dict = {}
    for key, val in pydict.items():
        if isinstance(val, list):
            pylist = [pys[0] for pys in val]
            py = ' '.join(pylist)
            val = replace_symbol_to_number(py)
        elif isinstance(val, str):
            val = replace_symbol_to_number(val)
        
        res_val = []
        for pys in val.split(','):
            res_py = []
            for py in pys.split():
                res_py.append(move_number(py, five))
            res_py = ' '.join(res_py)
            res_val.append(res_py)

        new_dict[key] = ','.join(res_val)

    return new_dict

if __name__ == '__main__':
    pass