import transformers as T
articles = []
tokenizer = T.BertTokenizer.from_pretrained('hfl/chinese-pert-base')