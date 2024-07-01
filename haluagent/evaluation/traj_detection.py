import json
import torch
from toolbox.tools import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import argparse
temp = "你是一个通过特定的框架检测回复文本中的幻象的智能体。下面是检测框架的详细说明。\n首先，你需要判断是否要将输入中的回复文本拆分为句子列表。 你可以使用拆分句子的工具。如果需要拆分，需要对每个句子逐一进行核查；否则就对整个回复文本进行核查。你可以选择适当的事实核查工具来获取用于核查的相关信息和知识然后使用匹配工具输出判断结果或者直接输出判断结果。如果不使用match工具而直接输出判断结果，则需要在思考中输出label。存在错误输出\"label = 1\"；不存在错误输出\"label = 0\"。核查完毕后，你需要在思考中反思所有检测结果并输出label，在行为中调用get\_answer()输出最终的检测结果，如果存在幻象一并输出幻象内容和证据。\n\n分句工具：\nsplit_text(text: str) → sentence_list\n输入是文本，该函数将文本分割成句子列表。\n\n事实核查工具：\nweb_search(sentence: str) → fact\n输入是一个句子，该函数使用搜索引擎来搜索相关信息。调用web_search后必须接着调用match工具来判断言判断的回复与检索到的信息是否匹配。\n\ncalculator(sentence: str, formula: str) → result, label\n输入是需要检查的公式，此函数使用计算器来获取计算结果并判断得到的结果是否与句子匹配。如果匹配label为0，否则为1。有效的运算符有 +、-、*、/ 和 (, )。例如，合法的输入可以是“(1 + 2) * 3”。如果输入为方程，需要将其转换为不含未知数的算式。\n\nword_count(length: int, text: str) → count, label\n输入文本的指定字数和一段文本。该函数计算这段文本的字数并输出为count。如果字数不符合要求，输出label为1，否则为0。\n\ncode_interpreter() → label\n该函数检查代码中是否能够正确执行。如果能正确执行，输出标签为0，否则为1。\n\n匹配工具：\nmatch(sentence: str, context:str) → label\n输入是一个句子以及相应的上下文。上下文可以是问题和回复中的检测句子之前的内容。该函数检查句子中是否存在答非所问或自相矛盾的情况。如果有，则输出标签为1，否则为0。如果你认为match的输出是错误的，可以在思考中修正label，例如如果你认为match输出的\"label = 0\"是错误的，可以在思考中输出\"label = 1\"。\n\n每次轮到你回复时，你必须严格遵循以下格式给出你的思考和行为：“思考：你的思考过程。\n行为：工具调用。如match(sentence=\"...\", context=\"...\")”，其中思考部分是你的规划内容，行为部分必须为一个工具调用指令。每次你调用工具后，我会以这种格式为你提供结果：“观察：工具的输出结果”。"

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
            query = {"role": "user", "content": temp + "问题：" + d["question"] + " 回复：" + d["answer"]}
            messages.append(query)
            #  baichuan2
            ans = model.chat(tokenizer, messages)
            if ("行为：") in ans:
                action = ans.split("行为：")[-1].strip(".").strip()
                    
                if "\n" in action:
                    action = action.split("\n")[0]
                ans = ans.split("行为：")[0] + "行为：" + action
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
                    response0 = "观察："
                    action = ans.split("行为：")[-1].strip(".").strip(":").strip()
                    print(action)
                    if "split" in action:
                        action = action.replace("：","")
                        sentences = eval(action)
                        response = {"role": "user", "content": response0+str(sentences)}
                    elif "calculator" in action:
                        result, label = eval(action)
                        response = {"role": "user", "content": response0 + result+", label = " + str(label)}
                        sentence = action.split("sentence=")[-1].split("\"")[1]
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
                        response = {"role": "user", "content": response0+"这段文本的字数是"+str(count)+"，label="+str(label)}
                        sen_list.append(d["answer"])
                        label_list.append(label)
                        fact_list.append("这段文本的字数是"+str(count))
                    elif "get_answer" in action:
                        fnum += 1
                        if fnum > 5:
                            break
                        if sen_list == []:
                            sen_list.append(d["answer"])
                            if web_fact:
                                fact_list.append(web_fact)
                        if label_list == []:
                            response = {"role": "user", "content": "观察：未检测到label，请输出label并重新调用get_answer()。"}
                            print(response)
                            messages.append(response)
                            ans = model.chat(tokenizer, messages)
                            if ("行为：") in ans:
                                action = ans.split("行为：")[-1].strip(".").strip()
                                thought = ans.split("行为：")[0].strip(".").strip()
                                if "label=1" in thought or "label = 1" in thought or "LABEL = 1" in thought:
                                    if len(label_list) > 0:
                                        label_list[-1] = 1
                                    else:
                                        label_list.append(1)
                                elif "label=0" in thought or "label = 0" in thought or "LABEL = 0" in thought:
                                    if len(label_list) > 0:
                                        label_list[-1] = 0
                                    else:
                                        label_list.append(0)
                                if "\n" in action:
                                    action = action.split("\n")[0]
                                ans = ans.split("行为：")[0] + "行为：" + action
                            messages.append({"role": "assistant", "content": ans})
                            print(ans)
                            continue
                        final_answer = get_answer(sen_list,label_list,fact_list)
                        messages.append({"role":"user", "content": "观察："+final_answer})
                        break
                    else:
                        print("No such tool, failed!")
                        assert(1==0)
                        break
                    assert(isinstance(response, dict))
                    assert(len(messages)<=20)
                        
                    print(response)
                    messages.append(response)
                    # Baichuan2
                    ans = model.chat(tokenizer, messages)
                    if ("行为：") in ans:
                        action = ans.split("行为：")[-1].strip(".").strip()
                        thought = ans.split("行为：")[0].strip(".").strip()
                        if "label=1" in thought or "label = 1" in thought or "LABEL = 1" in thought:
                            if len(label_list) > 0:
                                label_list[-1] = 1
                            else:
                                label_list.append(1)
                        elif "label=0" in thought or "label = 0" in thought or "LABEL = 0" in thought:
                            if len(label_list) > 0:
                                label_list[-1] = 0
                            else:
                                label_list.append(0)
                        if "\n" in action:
                            action = action.split("\n")[0]
                        ans = ans.split("行为：")[0] + "行为：" + action
                    messages.append({"role": "assistant", "content": ans})
                    print(ans)
            except Exception as e:
                print(e)
                fail_num.append(cnt)
                d["final_answer"] = "Error"
                d["trajectory"] = messages
                res.append(d)
                dump_jsonl(d,f"{args.output}l", append=True)
                continue
            final_answer = final_answer.split("观察：")[-1].strip()
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
