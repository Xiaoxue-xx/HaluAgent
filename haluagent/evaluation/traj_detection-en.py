import json
import torch
from toolbox.tools_en import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import argparse
temp = "You are an agent tasked with detecting hallucinations in reply texts using a specific framework. Below is a detailed explanation of the detection framework:\nFirstly, you need to determine whether to split the input reply text into a list of sentences using a sentence segmentation tool. If required, you should check each sentence individually; otherwise, the entire text should be checked as a whole. You can choose an appropriate fact-checking tool to obtain relevant information and knowledge for verification and then use the matching tool to output the judgment results or directly output the judgment results. If you do not use the match tool and directly output the judgment results, you need to output the label in your thought. There is an error output \"label = 1\"; there is no error output \"label = 0\". After the verification is completed, you need to reflect on all detection results and output the label in your thought, then call get\_answer() to produce the final detection result. \n\nSentence Splitting Tool: \nsplit_text(text: str) → sentence_list\nThis function splits the text into a list of sentences.\n\nFact-Checking Tools: \nweb_search(sentence: str) → fact\nThis function uses a search engine to find information related to the sentence. After using web_search, you must use the match tool to determine if the reply matches the retrieved information.\n\ncalculator(sentence: str, formula: str) → result, label\nThis function uses a calculator to obtain the result of a formula and checks if the result matches the sentence. If they match, the label is 0; otherwise, it is 1. Valid operators include +, -, *, /, and parentheses. For instance, a valid input could be “(1 + 2) * 3”. If the input is an equation, it needs to be converted to a formula without unknowns.\n\nword_count(length: int, text: str) → count, label\nThis function calculates the word count of a text and outputs the count. If the word count does not meet the specified length, the label is 1; otherwise, it is 0.\n\ncode_interpreter() → label\nThis function checks whether the code can execute correctly. If it executes correctly, the output label is 0; otherwise, it is 1.\n\nMatching Tool:\nmatch(sentence: str, context:str) → label\nThis function checks a sentence against its context, which might include content from questions and replies around the detected sentence. It looks for irrelevant or contradictory answers. If any are found, the label is 1; otherwise, it is 0. If you think the output of match is wrong, you can correct the label in thought. For example, if you think the \"label = 0\" output by match is wrong, you can correct the answer and output \"label = 1\" in thought.\n\nEach time it’s your turn to respond, you must strictly follow this format to present your thoughts and actions: \"Thought: Your thought process.\nACTION: Tool call, e.g., match(sentence=\"...\", context=\"...\")\". After each tool use, I will provide the output as follows: \"Observation: Tool's output result\"."

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')

def init_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

def get_response(messages):
    model, tokenizer = init_model()
    response = model.chat(tokenizer, messages)
    return response
    
def generate_p(args):
    model, tokenizer = init_model(args.model_path)
    with open(args.input, "r", encoding="utf-8") as f:
        data = []
        data = json.load(f)
        print(len(data))
    cnt = 0
    res = []
    fail_num = []
    try:
        for d in data:
            cnt += 1
            fnum = 0
            messages = []
            query = {"role": "user", "content": temp + "QUERY: " + d["question"] + " RESPONSE:" + d["answer"]}
            messages.append(query)
            #  baichuan2
            ans = model.chat(tokenizer, messages)
            if ("ACTION:") in ans:
                action = ans.split("ACTION:")[-1].strip(".").strip()
                    
                if "\n" in action:
                    action = action.split("\n")[0]
                ans = ans.split("ACTION:")[0] + "ACTION:" + action
            messages.append({"role": "assistant", "content": ans})
            # print(ans)
            label_list = []
            fact_list = []
            sen_list = []
            web_fact = ""
            # Tool Check
            try:
                while True:
                    response = ""
                    response0 = "OBSERVATION:"
                    action = ans.split("ACTION:")[-1].strip(".").strip(":").strip()
                    print(action)
                    if "split" in action:
                        action = action.replace("：","")
                        sentences = eval(action)
                        response = {"role": "user", "content": response0+str(sentences)}
                    elif "calculator" in action:
                        result, label = eval(action)
                        response = {"role": "user", "content": response0 + result+", label = " + str(label)}
                        sentence = action.split("sentence=")[-1].split("\"")[1]
                        if d["answer"] in result:
                            label_list.append(0)
                        else:
                            sen_list.append(sentence)
                            fact_list.append(result)
                            label_list.append(label)
                    elif "web_search" in action:
                        fact= eval(action)
                        response = {"role": "user", "content": response0+fact}
                        web_fact = fact
                    elif "match" in action:
                        label = eval(action)
                        text = "label = " + str(label)
                        response = {"role": "user", "content": response0+text}
                        sentence = action.split("sentence=")[-1].split("\"")[1]
                        sen_list.append(sentence)
                        fact = action.split("context=")[-1].strip("\")")
                        fact_list.append(fact)
                        label_list.append(label)
                    elif "code_interpreter" in action:
                        label, report = eval(action)
                        # label, report = code_interpreter(code=d["answer"])
                        text = "label = " + str(label)
                        response = {"role": "user", "content": response0+text}
                        sen_list.append(d["answer"])
                        fact = report
                        fact_list.append(fact)
                        label_list.append(label)
                    elif "word_count" in action:
                        text = d["answer"]
                        action = action.split("text=")[0]+"text=text)"
                        print(action)
                        count, label = eval(action)
                        response = {"role": "user", "content": response0+"The number of words in this text is "+str(count)+"，label="+str(label)}
                        sen_list.append(d["answer"])
                        label_list.append(label)
                        fact_list.append("The number of words in this text is "+str(count))
                    elif "get_answer" in action:
                        fnum += 1
                        if fnum > 5:
                            break
                        if sen_list == []:
                            sen_list.append(d["answer"])
                            if web_fact:
                                fact_list.append(web_fact)
                        if label_list == []:
                            response = {"role": "user", "content": "OBSERVATION: The label is not detected, please output the label in THOUGHT like label = 1 or label = 0 and call get_answer() again."}
                            print(response)
                            messages.append(response)
                            ans = model.chat(tokenizer, messages)
                            if ("ACTION:") in ans:
                                action = ans.split("ACTION:")[-1].strip(".").strip()
                                thought = ans.split("ACTION:")[0].strip(".").strip()
                                if "label=1" in thought.lower() or "label = 1" in thought.lower() or "LABEL = 1" in thought or "label should be 1" in thought:
                                    if len(label_list) > 0:
                                        label_list[-1] = 1
                                    else:
                                        label_list.append(1)
                                elif "label=0" in thought.lower() or "label = 0" in thought.lower() or "LABEL = 0" in thought or "label should be 0" in thought:
                                    if len(label_list) > 0:
                                        label_list[-1] = 0
                                    else:
                                        label_list.append(0)
                                if "\n" in action:
                                    action = action.split("\n")[0]
                                ans = ans.split("ACTION:")[0] + "ACTION:" + action
                            messages.append({"role": "assistant", "content": ans})
                            print(ans)
                            continue
                        final_answer = get_answer(sen_list,label_list,fact_list)
                        messages.append({"role":"user", "content": "OBSERVATION:"+final_answer})
                        break
                    else:
                        print("No such tool, failed!")
                        assert(1==0)
                        break
                    assert(isinstance(response, dict))
                    print(response)
                    messages.append(response)
                    # Baichuan2
                    ans = model.chat(tokenizer, messages)
                    if ("ACTION:") in ans:
                        action = ans.split("ACTION:")[-1].strip(".").strip()
                        thought = ans.split("ACTION:")[0].strip(".").strip()
                        if "label=1" in thought.lower() or "label = 1" in thought.lower() or "LABEL = 1" in thought or "label should be 1" in thought:
                            if len(label_list) > 0:
                                label_list[-1] = 1
                            else:
                                label_list.append(1)
                        elif "label=0" in thought.lower() or "label = 0" in thought.lower() or "LABEL = 0" in thought or "label should be 0" in thought:
                            if len(label_list) > 0:
                                label_list[-1] = 0
                            else:
                                label_list.append(0)
                        if "\n" in action:
                            action = action.split("\n")[0]
                        ans = ans.split("ACTION:")[0] + "ACTION:" + action
                    messages.append({"role": "assistant", "content": ans})
                    print(ans)
            except Exception as e:
                print(e)
                fail_num.append(cnt)
                # i -= 1
                d["final_answer"] = "Error"
                d["trajectory"] = messages
                res.append(d)
                dump_jsonl(d,f"{args.output}l", append=True)
                continue
            d["final_answer"] = final_answer
            d["trajectory"] = messages
            res.append(d)
            dump_jsonl(d, f"{args.output}l", append=True)

        with open(args.output,"w") as f:
            json.dump(res,f,ensure_ascii=False,indent=4)

    except Exception as e:
        print(f"发生错误: {e}")
        # 在发生错误时保存当前结果
        with open(f"{args.output}_backup.json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
    print(f"共有{len(fail_num)}个问题处理失败")
    print(fail_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=""
    )
    parser.add_argument(
        "--output",
        type=str,
        default=""
    )
    args = parser.parse_args()
    generate_p(args)
