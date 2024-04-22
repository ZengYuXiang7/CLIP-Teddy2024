# coding : utf-8
# Author : yuxiang Zeng
import re
import jionlp as jio


def jionlp_keys(text):
    key_phrases = jio.keyphrase.extract_keyphrase(top_k=3,func_word_num=2,strict_pos=True,text=text)
    return key_phrases


def clear_txt(sent):
    if len(sent) < 5:
        return sent
    clear_words = ["<人名>", "《", "》", "【", "】", "→", ":", "—"]
    pattern = re.compile(r'【(.*?)】|《(.*?)》|“(.*?)”', re.DOTALL)
    # 查找所有匹配项
    matches = pattern.findall(sent)
    # 整理匹配结果
    resultskeys = []
    for match in matches:
        for group in match:
            if group:
                resultskeys.append(group)  ##提取了上面符号中的关键词

    for cw in clear_words:
        if cw in sent:
            sent = sent.replace(cw, " ")  ##将上面的特殊符号替换成空格
    sent2 = sent
    # print("sent2", sent2)
    for key in resultskeys:
        sent = sent.replace(key, "")  ## 不仅删除了上面的特殊符号也删除了关键词
    sent_dropkey = sent
    # print("sent_dropkey", sent_dropkey)
    key_list = []
    if len(resultskeys) > 0:
        for key in resultskeys:
            key_list.append(key)
    for v1 in jionlp_keys(sent2):
        if v1 not in key_list:
            key_list.append(v1)

    for v2 in jionlp_keys(sent_dropkey):
        if v2 not in key_list:
            key_list.append(v2)

    new_sent = ""
    for kw in key_list:
        new_sent = new_sent + kw + ' '
    return new_sent


if __name__ == "__main__":
    origin_text = "《绿色北京》摄影大赛胡子<人名>作品"
    change_text = clear_txt(origin_text)
    print(change_text)
