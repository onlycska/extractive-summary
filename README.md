# Extractive summarization

## Dataset description
Dataset for extractive summarization is gazeta from arXiv:2006.11063 [cs.CL]. It contains news articles and their summary.
Original dataset is jsonl file. Every article is json. Every article has 5 keys. Two of them is important for my preparings – ‘summary’ (with summary for news written by human) and ‘text’ (news written by human).
Dataset Split	Number of Instances in Split
Train	52,400
Validation	5,265
Test	5,770

## Dataset preparing
For two different tasks solving I used two types of datasets preparing:
### Dataset for first sentence classifying (`bert_dataset_preprocess_first_sentence.py`)
With this dataset I wanted to prepare model which can understand is the first sentence summary of next article or not. So, there are some requirements for dataset:
-	Dataset should be not larger than 512 tokens (for BERT)
-	Dataset should be with labels – 0 or 1.
First, I cut text to 410 BERT tokens. 
After that, I embedded first sentence in ‘summary’ by rubert-base-cased-sentence and found most similar sentence from cut text (by cosine similarity). And this sentence I placed as the first sentence in cut text. It was 50% of dataset with labels 1. Another 50% with label 0 – they were with sentence from cut text which was less like first sentence from ‘summary’.
Before first sentence I placed token [CLS] and after every sentence I placed [SEP] token

### Dataset for every sentence classifying (`lstm_dataset_preprocess_every_sentence.py`)
In this case my model should understand is sentence summary of the article or not only by its embedding. Therefore, in this dataset I took every sentence from ‘summary’ and found most similar (by cosine similarity) to them in ‘text’. Save their indexes. After that, I embedded every sentence from text by rubert-base-cased-sentence. And place label – 0 or 1 (summary or not).

## Approach description

### Classify first sentence (`bert_classifier_colab.ipynb`)
For this task I used BERT model. Before first sentence I placed [CLS] token and [SEP] token after every sentence in text. I used dataset from 4.1. [CLS] stands for classification. So, in my case [CLS] token stands for classification is the first sentence summary of the article or not
### Classify every sentence (`lstm_classifier.ipynb`)
In this case I worked with dataset from 4.2 and I used sentence BERT + 2-layer BiLSTM. Every sentence of dataset was embedded by BERT in matrix [1, 768]. These embeddings used in BiLSTM for classification – is it summary or not.

## Results
Results of two solved tasks:
Task	                          Metric	        Result
First sentence classifying	    F1-measure	    95%
Every sentence classifying	    Accuracy	      92.6%

Task | Metric | Result
--- | --- | ---
First sentence classifying | F1-measure | 95%
Every sentence classifying | Accuracy | 92.6%

## Conclusion
### First sentence classifying
It's interesting approach but is very difficult to make it fast. If we want to classify every sentence, we need to classify N sentences. Where N = number of sentences in text. And in every classifying, we need to use all the text, so BERT can understand either it’s summary for all text or not. With BERT this task is hard because of limitations of the attention mechanism (number of tokens).
### Every sentence classifying
This approach is limited to  because embeddings of every sentence loss context of all text around it. So trained model needs to understand is the sentence summary of the text without knowing text itself. I think it’s bad decision, but it’s very fast and easy to code.
