---
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 7334642
    num_examples: 4675
  download_size: 2883783
  dataset_size: 7334642
license: mit
language:
- ar
pretty_name: arywiki-articles-withoutbots
size_categories:
- 1K<n<10K
---
# Dataset Card for "Moroccan_Arabic_Wikipedia_20230101_nobots"

This dataset is created using the Moroccan Arabic Wikipedia articles (**after removing bot-generated articles**), downloaded on the 1st of January 2023, processed using `Gensim` Python library, and preprocessed using `tr` Linux/Unix utility and `CAMeLTools` Python toolkit for Arabic NLP. This dataset was used to train this Moroccan Arabic Wikipedia Masked Language Model: [SaiedAlshahrani/arywiki_20230101_roberta_mlm_nobots](https://huggingface.co/SaiedAlshahrani/arywiki_20230101_roberta_mlm_nobots).

For more details about the dataset, please **read** and **cite** our paper:

```bash
@inproceedings{alshahrani-etal-2023-performance,
    title = "{Performance Implications of Using Unrepresentative Corpora in {A}rabic Natural Language Processing}",
    author = "Alshahrani, Saied  and Alshahrani, Norah  and Dey, Soumyabrata  and Matthews, Jeanna",
    booktitle = "Proceedings of the The First Arabic Natural Language Processing Conference (ArabicNLP 2023)",
    month = December,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.19",
    doi = "10.18653/v1/2023.arabicnlp-1.19",
    pages = "218--231",
    abstract = "Wikipedia articles are a widely used source of training data for Natural Language Processing (NLP) research, particularly as corpora for low-resource languages like Arabic. However, it is essential to understand the extent to which these corpora reflect the representative contributions of native speakers, especially when many entries in a given language are directly translated from other languages or automatically generated through automated mechanisms. In this paper, we study the performance implications of using inorganic corpora that are not representative of native speakers and are generated through automated techniques such as bot generation or automated template-based translation. The case of the Arabic Wikipedia editions gives a unique case study of this since the Moroccan Arabic Wikipedia edition (ARY) is small but representative, the Egyptian Arabic Wikipedia edition (ARZ) is large but unrepresentative, and the Modern Standard Arabic Wikipedia edition (AR) is both large and more representative. We intrinsically evaluate the performance of two main NLP upstream tasks, namely word representation and language modeling, using word analogy evaluations and fill-mask evaluations using our two newly created datasets: Arab States Analogy Dataset (ASAD) and Masked Arab States Dataset (MASD). We demonstrate that for good NLP performance, we need both large and organic corpora; neither alone is sufficient. We show that producing large corpora through automated means can be a counter-productive, producing models that both perform worse and lack cultural richness and meaningful representation of the Arabic language and its native speakers.",
}
```