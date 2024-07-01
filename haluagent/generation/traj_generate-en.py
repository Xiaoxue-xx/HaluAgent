import openai
import time
import json
from toolbox.tools_en import *


openai.api_key = ""
openai.api_base = ""     

with open("prompt/trajectory_generation_en.jsonl", "r") as f:
    global prompt
    prompt = []
    for line in f:
        prompt.append(json.loads(line))
    print(len(prompt))

def get_res_batch(input):
    global prompt 
    messages = prompt+[input]
    prompt = messages
    print("######", messages[-1])
    while True:
        try:
            res = openai.ChatCompletion.create(
                model = "gpt-4",
                messages=messages,
                temperature=0
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
    
    return res['choices'][0]['message']['content']

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')

def generate_p(file, output):
    global prompt 
    with open(file, "r", encoding="utf-8") as f:
        data = []
        data = json.load(f)
        print(len(data))
    cnt = 0
    res = []
    fail_num = []
    try:
        for d in data:
            cnt += 1
            prompt = prompt[:14]
            # print(len(prompt))
            query = {"role": "user", "content": "QUERY: " + d["question"] + " RESPONSE: " + d["answer"]}
            ans = get_res_batch(query)
            if ("ACTION:") in ans:
                action = ans.split("ACTION:")[-1].strip(".").strip()
                if "\n" in action:
                    action = action.split("\n")[0]
                ans = ans.split("ACTION:")[0] + "ACTION:" + action
            prompt.append({"role": "assistant", "content": ans})
            print(ans)
            label_list = []
            fact_list = []
            sen_list = [d["answer"]]
            # Tool Check
            try:
                while True:
                    response = "OBSERVATION:"
                    action = ans.split("ACTION:")[-1].strip(".").strip(":").strip()
                    print(action)
                    if "split" in action:
                        action = action.replace("：","")
                        sentences = eval(action)
                        response = {"role": "user", "content": response+str(sentences)}
                        sen_list = sentences
                    elif "calculator" in action:
                        result, label = eval(action)
                        response = {"role": "user", "content": response + result+", label = " + str(label)}
                        fact_list.append(result)
                        label_list.append(label)
                    elif "web_search" in action:
                        fact, label = eval(action)
                        response = {"role": "user", "content": response+fact+", label = " + str(label)}
                        fact_list.append(fact)
                        label_list.append(label)
                    elif "code_interpreter" in action:
                        label, report = eval(action)
                        # label, report = code_interpreter(code=d["answer"])
                        text = "label = " + str(label)
                        response = response+text
                        sen_list.append(d["answer"])
                        fact = report
                        fact_list.append(fact)
                        label_list.append(label)
                    elif "match" in action:
                        label = eval(action)
                        text = "label = " + str(label)
                        response = {"role": "user", "content": response+text}
                        fact = action.split("context=")[-1].strip("\"")
                        fact_list.append(fact)
                        label_list.append(label)
                    elif "word_count" in action:
                        count, label = eval(action)
                        response = {"role": "user", "content": response+"The number of words in this text is "+str(count)}
                        label_list.append(label)
                        fact_list.append("The number of words in this text is "+str(count))
                    elif "get_answer" in action:
                        if sen_list == []:
                            sen_list.append(d["answer"])
                        final_answer = get_answer(sen_list,label_list,fact_list)
                        response = {"role": "user", "content": response+final_answer}
                        prompt.append(response)
                        break
                    else:
                        print("No such tool, repeat!")

                    assert(isinstance(response, dict))
                    # print(response)
                    ans = get_res_batch(response)
                    if ("ACTION:") in ans:
                        action = ans.split("ACTION:")[-1].strip(".").strip()
                        if "\n" in action:
                            action = action.split("\n")[0]
                        ans = ans.split("ACTION:")[0] + "ACTION:" + action
                    prompt.append({"role": "assistant", "content": ans})
                    print(ans)
            except Exception as e:
                print(e)
                fail_num.append(cnt)
                d["final_answer"] = "Error"
                d["trajectory"] = prompt[14:]
                res.append(d)
                dump_jsonl(d,f"{output}l", append=True)
                continue
            final_answer = final_answer.split("OBSERVATION:")[-1].strip()
            d["final_answer"] = final_answer
            d["trajectory"] = prompt[14:]
            res.append(d)
            dump_jsonl(d, f"{output}l", append=True)

        with open(output,"w") as f:
            json.dump(res,f,ensure_ascii=False,indent=4)

    except Exception as e:
        print(f"发生错误: {e}")
        # 在发生错误时保存当前结果
        with open(f"{output}_backup.json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
    print(f"共有{len(fail_num)}个问题处理失败")
    print(fail_num)
        
if __name__ == '__main__':
    file = ""
    output_file = ""
    generate_p(file, output_file)