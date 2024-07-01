from googletrans import Translator
import openai
import time
import spacy
from Calculator import *
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
    使用 spaCy 进行句子分割。

    参数:
    text (str): 要分割的文本。

    返回:
    list: 分割后的句子列表。
    """
    # 加载英文模型（确保您已经安装并下载了该模型）
    nlp = spacy.load("en_core_web_sm")
    # 使用 spaCy 处理文本
    doc = nlp(text)

    # 提取句子
    sentences = [sentence.text.strip() for sentence in doc.sents]

    return sentences


def web_search(sentence, api_keys=keys, lr='lang_en', num=5):
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
            print(f"An error occurred:\n{e}")

def match(sentence, context):
    template = "Given a piece of text and its corresponding context, please determine whether the text contains incorrect content. If you think there is an error, answer \"Yes\". Otherwise, answer \"No\".\n\n#Given Text#: {sentence}\n#Context#: {context}\n#Your Judgment#:\n\n#Given Text#: {sentence}\n#Context#: {context}\n#Your Judgment#:"
    input = template.format(sentence=sentence, context=context)
    print(input)
    message = [
        {"role": "user", "content": input},
    ]
    while True:
        try:
            res = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-4-turbo",
                messages=message,
                temperature=0,
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
    print("gpt4:", res['choices'][0]['message']['content'])
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
    label = 0 if answer_length == length else 1

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
    result = translator.translate(text, dest=target_language)
    return result.text
    
def calculate_days_between_dates(date1, date2):
    date_format = "%Y-%m-%d"
    d1 = datetime.datetime.strptime(date1, date_format)
    d2 = datetime.datetime.strptime(date2, date_format)
    delta = d2 - d1
    return delta.days
    