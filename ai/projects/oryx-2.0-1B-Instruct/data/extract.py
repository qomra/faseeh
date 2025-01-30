
"""
extract generated text from this json list and put it in a text file generation.rtl.txt
{
    "index": 0,
    "completion": [
        {
            "generated_text": "يبدو أن الجذر \"أَبأ\" يمتد من الاستخدامات المادية المحسوسة إلى الاستخدامات المجردة والمعنوية في النثر. يبدو أن المعنى المحوري له يتراوح بين \"الخلاط\" أو \"الجذور\" لشكل الحرف، ويتجاوز ذلك لوصف الجذر بشكل أكثر تعقيدية.\n\nتتماشى معنا هذه النقطة مع ظهور الاستخدامات المجردة والمعنوية في النص، حيث يتم تعريف الجذر \"أَبأ\" بأنه يمتد من \"الخلاط\" إلى \"الجذور\"، ويتجاوز ذلك لوصف الجذر بشكل أكثر تعقيدية."
        }
    ]
}
{
    "index": 1,
    "completion": [
        {
            "generated_text": "الجذر: الشهنز\n\nتحديد المعنى المحوري: \"أَبَا الدُقَيْشِ\" هو مصطلح محوري يرتبط بمعنى \"أب الدَقّ\" أو \"أب الدَقّ مِنَ الدَقّ\". هذه المعنى المحوري يُشير إلى الفرد الذي يُعتبر \"أب الدَقّ\"، وهو ما يعني \"أب الدَقّ المُستقل\" أو \"أب الدَقّ الذي لا يملك anything من العلاقات المتبادلة\". هذا المعنى المحوري يُشير إلى الخصال الفردي أو الخصال الجدير بالثقة، ويشير إلى الفرد الذي يملك \"دَقّاً\" مُستقلًا من العلاقات المتبادلة.\n\nترتيب المعنى المحوري:\n\n1. \"أب الدَقّ\" هو مصطل"
        }
    ]
}

"""
import json
import os
import sys
import argparse

def extract_text(json_file, output_file):
    # read first as text
    with open(json_file, "r") as f:
        data = f.read()
    # split by newline
    data = data.strip().split("\n")
    # loop 4 lines at a time
    data_str = []
    for i in range(0, len(data), 8):
        # join 8 lines
        text = "\n".join(data[i:i+8])
        # load json
        text = json.loads(text)

        # extract text
        text = text["completion"][0]["generated_text"]
        # replace \" with "
        text = text.replace("\\", "")
        data_str.append(text.strip())

    # write to file
    with open(output_file, "w") as f:
        f.write("\n#######################\n".join(data_str))

def extract_text_2(json_file, output_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    data_str = []
    for k,v in data.items():
        text = v
        text = text.replace("\\", "")
        data_str.append(text.strip())
    with open(output_file, "w") as f:
        f.write("\n#######################\n".join(data_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract generated text from json list')
    parser.add_argument('json_file', type=str, help='json file containing generated text')
    parser.add_argument('output_file', type=str, help='output file to write generated text')
    args = parser.parse_args()
    if args.json_file.endswith(".jsonl"):
        extract_text(args.json_file, args.output_file)
    else:
        extract_text_2(args.json_file, args.output_file)