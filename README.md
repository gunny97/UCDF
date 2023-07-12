# User-defined Content Detection Framework	

*This paper presents UCDF, a simple interactive framework for training a classifier that detects user-defined content. We define user-defined content as texts that a user wants to detect. UCDF receives examples of user-defined content as queries to make a Customized dataset automatically using a method similar to dense passage retrieval. It uses the Customized dataset to train a classifier, a user-defined content detector. We tested it using SimCSE (base) on a user-defined content detection task with a manually built dataset crawled from FactsNet that contains 291 different subcategories. Also, this simple method can be extended to a general text classifier by feeding queries for each category. We evaluated it on several classification benchmark datasets by setting randomly sampled texts from a training set as queries. It showed a competitive level of performance to GPT-3 175B. Moreover, we found that UCDF can act as text augmentation, showing similar performance compared to recent text augmentation techniques such as back translation.*

This is the official repository for our [User-defined Content Detection Framework], submitted on [NeurIPS 2022 Workshopt of InterNLP(Interactive Learning For Natural Langugage Processing)](https://internlp.github.io/2022/accepted_papers.html).
We introdue a novel research topic, 'user-defined content detection' and propose **UCDF** to solve the task.
UCDF is a set of framework to detect any kind of content that users have defined.
We conducted an exmperiment utilizing a source code, https://github.com/facebookresearch/DPR(*DPR repo*), from facebookresearch. We modified the source code to adjust the experiment in this paper. 
If you want to reproduce this work, you should install dependencies based on *DPR repo* and additional libraries follow requirements.txt. 
<p align="center">
  <img src="./ucdf_framework.PNG" width="80%" height="80%">
</p>   



## Content
While generating embeddings for Data Collection and training a dual-encoder, we used SimCSE as an encoder not BERT in order to better understand the contexts in sentences. We trained a daul-encoder with NaturalQuestion dataset on contrastive learning. We used source code from *DPR repo* for this process. 
```sh
# generate embeddings for Data Collection
# use train_dense_encoder.py
bash generated_DE.sh

# training a dual-encoder which conducts retrieving process
# use generate_dense_embeddings.py
bash train_encoder.sh
```


Next step, we need to conduct __Information Retrieval__ we should make Customized dataset using Semantic Search algorithm. 'build_dataset.py' builds 'AGNews, DBpedia, FactsNet, Custom data(Religion, South Korea)', 'build_sst2_dataset.py' builds 'SST-2', and 'build_trec_dataset' builds 'TREC'.
```sh
# generate Customized dataset
python build_dataset.py
```

After building Customized dataset, we should fine-tune a classifier on it. Its process is just simple that only uses cross-entropy objective function. In case of detection task, we used BCE loss but general CE in multi-class classification tasks. Experiment results can be seen in *experiment_logs.xlsx*.

- benchmark dataset experiment: path follows finetuning/Multi_class
- Data Augmentation: path follows finetuning/augmentation
- negative query: path follows finetuning/negative_query
- user-defined content detection: path follows finetuning/user-defined-content-detection(FactsNet)
