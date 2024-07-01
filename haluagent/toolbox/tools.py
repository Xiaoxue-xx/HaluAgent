from html import unescape
import re
from googletrans import Translator
import openai
import time
from Calculator import *
import jieba
from googleapiclient.discovery import build
from interpreter_api import safe_execute 
from sympy import sympify
import datetime

cx = ""
keys = [""]
openai.api_key = ""
openai.api_base = ""


def split_text(text):
    """
    结合正则表达式和jieba分词来分割中文文本为句子列表。
    
    参数:
    - text: 字符串，待分割的中文文本。
    
    返回:
    - sentences: 句子列表。
    """
    # 使用正则表达式来匹配中文句子结束的标点符号
    text = text.replace('\n', '').replace('\r', '').replace('\t', '')
    sentence_delimiters = re.compile(r'([。？！；…])')
    
    # 使用正则表达式分割文本，并保留结束符号
    parts = sentence_delimiters.split(text)
    temp_sentences = []
    for i in range(0, len(parts) - 1, 2):
        temp_sentences.append(parts[i] + (parts[i+1] if i+1 < len(parts) else ''))
    
    if not temp_sentences:
        temp_sentences = [text]

    sentences = []
    for temp_sentence in temp_sentences:
        temp_sentence = temp_sentence.strip()
        if not temp_sentence:
            continue
        
        # 使用jieba分词来进一步确认句子边界
        words = list(jieba.cut(temp_sentence))
        sentence = ''
        for word in words:
            sentence += word
            # 如果分词结果中的词包含句子结束符号，则视为句子结束
            if any(delimiter in word for delimiter in '。？！；…'):
                sentences.append(sentence)
                sentence = ''
        
        # 添加最后一个句子（如果有）
        if sentence:
            sentences.append(sentence)
    
    return sentences

def web_search(sentence, api_keys=keys, lr='lang_zh-CN', num=10):
    for api_key in api_keys:
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            res = (
                service.cse()
                .list(
                    q=sentence,
                    cx=cx,
                    lr=lr,
                    num=num
                )
                .execute()
            )
            snippets = []
            if 'items' in res:
                for result in res['items']:
                    if "snippet" in result:
                        snippets.append(result["snippet"])
                fact = "\n".join(snippets)
                return fact  # 如果请求成功，返回结果并退出函数
            else:
                return "No results found."  # 如果没有找到结果，返回相应信息
        except Exception as e:
            # 如果有其他类型的异常发生，打印错误信息并尝试下一个API密钥
            print(f"An error occurred:\n{e}")

def match(sentence, fact):
    template = "给定一段文本和这段文本对应上下文，可能是文本对应的问题或与这段文本相关的事实。基于上下文，如果你认为给定文本存在错误或无法核实的内容，请回答“是”。如果不存在，回答“否”。\n\n#给定文本#: {sentence}\n#上下文#: {context}\n#你的判断#:"
    input = template.format(sentence=sentence, fact=fact)
    message = [
        {"role": "user", "content": input},
    ]
    while True:
        try:
            res = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-4",
                messages=message,
                max_tokens=512
            )
            break

        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(20)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    if "yes" in res['choices'][0]['message']['content'].lower():
        label = 1
    elif "no" in res['choices'][0]['message']['content'].lower():
        label = 0
    
    return label

def calculator(sentence, formula):
    try:
        result = sympify(formula)
        if isinstance(result, float):
            # 将result转换为字符串，以检查小数点后的位数
            result_str = str(result)
            # 检查是否存在小数点且小数点后的位数大于4
            if '.' in result_str and len(result_str.split('.')[1]) > 4:
                # 如果满足条件，保留4位小数
                result = round(result, 4)
        fact = f"{formula} = {result}"
        label = match(sentence, fact)
        return fact, label
    except SyntaxError as e:
        return f"Error: {e}"

def get_answer(sentences, labels, facts):
    sen = []
    for i in range(len(labels)):
        if labels[i] == 1:
            sen.append({"sentence":sentences[i], "fact":facts[i]})
    if len(sen) == 0:
        return "否。"
    else:
        return "是。" + str(sen)
    
def word_count(length, text):
    # 计算答案的字数
    answer_length = len(text)

    # 比较答案字数和要求字数，添加标签
    if answer_length == length: 
        label = 0 
    else:
        label = 1

    return answer_length, label

def code_interpreter(code):
    print(code)
    an, report = safe_execute(code)
    print(an, report)
    label = 1
    if report == "Done":
        label = 0
    return label, report

def translate(text, target_language='en'):
    translator = Translator()
    if "en" in target_language:
        target_language = 'en'
    result = translator.translate(text, dest=target_language)
    return result.text

def date(date1, date2):
    date_format = "%Y-%m-%d"
    d1 = datetime.datetime.strptime(date1, date_format)
    d2 = datetime.datetime.strptime(date2, date_format)
    delta = d2 - d1
    return delta.days