---
annotations_creators:
- expert-generated
language_creators:
- expert-generated
- crowdsourced
language:
- ar
multilinguality:
- monolingual
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- multi-label-classification
---



# Arabic Quotes Dataset (arabic_Q)

The "Arabic Quotes" dataset contains a collection of Arabic quotes along with their corresponding authors and tags. The dataset is scraped from the website "arabic-quotes.com" and provides a diverse range of quotes from various authors.

## Dataset Details

- **Version**: 1.0.0
- **Total Quotes**: 3778
- **Languages**: Arabic
- **Source**: arabic-quotes.com

## Dataset Structure

The dataset is provided in the JSONL (JSON Lines) format, where each line represents a separate JSON object. The JSON objects have the following fields:

- `quote`: The Arabic quote text.
- `author`: The author of the quote.
- `tags`: A list of tags associated with the quote, providing additional context or themes.

## Dataset Examples

Here are a few examples of the quotes in the dataset:

```json
{
  "quote": "اذا لم يكن لديك هدف ، فاجعل هدفك الاول ايجاد واحد .",
  "author": "وليام شكسبير",
  "tags": ["تنمية الذات", "تحفيز"]
}

{
  "quote": "قيمة الحياة ليست في مدى طولها ، بل في مدى قيمتها",
  "author": "وليام شكسبير",
  "tags": ["الحياة", "القيمة"]
}

{
  "quote": "التحدث عن الامور العميقة ليس سهلاً كما يبدو",
  "author": "جبران خليل جبران",
  "tags": ["التواصل", "العمق"]
}
```


## Dataset Usage

The "Arabic Quotes" dataset can be used for various purposes, including:

- Natural Language Processing (NLP) tasks in Arabic text analysis.
- Text generation and language modeling.
- Quote recommendation systems.
- Inspirational content generation.
- text-classification

## Acknowledgements

We would like to thank the website "arabic-quotes.com" for providing the valuable collection of Arabic quotes used in this dataset.

## License

The dataset is provided under the [bigscience-bloom-rail-1.0 License](https://huggingface.co/spaces/bigscience/license), which permits non-commercial use and sharing under certain conditions.
