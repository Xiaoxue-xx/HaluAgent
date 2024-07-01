import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import argparse

sys_mes = {"role": "system", "content": "You are a hallucination detection agent. Given a question and its corresponding response, please determine whether there is any incorrect or unsatisfactory content within the response. If there is, output \"Yes\"; if there is not, output \"No\". "}


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
    # Baichuan2
    response = model.chat(tokenizer, messages, temperature=0.0)
    return response
    
def generate_p(args):
    model, tokenizer = init_model(args.model_path)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(len(data))

    cnt = 0
    res = []
    fail_num = []
    try:
        for d in data:
            cnt += 1
            messages = []
            messages.append(sys_mes)
            query = {"role": "user", "content": "QUESTION: " + d["question"] + " RESPONSE: " + d["answer"]}
            messages.append(query)
            # Baichuan2
            ans = model.chat(tokenizer, messages)
            d["final_answer"]=ans
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
