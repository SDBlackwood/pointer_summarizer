# MSc Data Science Masters 

## Improving Answer Summary Quality in Community Question Answering Systems using Metadata Represented via Structural Attention and Neural Abstractive Summarization

```
── data                                 -- Relates to the data generation script                                             
│   └── generator
│       ├── config.py
│       ├── document.py
│       ├── findImportantSentences.py
│       ├── main.py
│       ├── parser.py
│       ├── pointerDataset.py
├── data_util                           -- Config and utils for the network
│   ├── batcher.py
│   ├── config.py
│   ├── data.py
|   ├── determine_mean_answer_length.py
|   ├── total_answer_lengths.txt
│   └── utils.py
├── start_decode.sh                     -- Script to generate outputs and create ROUGE scores
├── start_train.sh                      -- Script to train the model 
├── tests
│   └── test_train.py
└── training_ptr_gen
    ├── decode.py                       
    ├── eval.py
    ├── explict_structured_attention.py -- Explict Structural Attenion Code
    ├── model.py                        -- Main file containing Pytorch model code
    ├── structured_attention.py         -- Latent structural attenion code
    ├── train.py                        -- Training code
    └── train_util.py
```
## PGN = Pointer Generator Network


ROUGE-1:
rouge_1_f_score: 0.1279 with confidence interval (0.0951, 0.1581)
rouge_1_recall: 0.0817 with confidence interval (0.0610, 0.1012)
rouge_1_precision: 0.3081 with confidence interval (0.2326, 0.3733)

ROUGE-2:
rouge_2_f_score: 0.0108 with confidence interval (0.0028, 0.0193)
rouge_2_recall: 0.0069 with confidence interval (0.0018, 0.0123)
rouge_2_precision: 0.0257 with confidence interval (0.0074, 0.0448)

ROUGE-l:
rouge_l_f_score: 0.1130 with confidence interval (0.0834, 0.1411)
rouge_l_recall: 0.0721 with confidence interval (0.0532, 0.0897)
rouge_l_precision: 0.2728 with confidence interval (0.2002, 0.3382)

## LSA - Latent Structura Attenion 

ROUGE-1:
rouge_1_f_score: 0.0959 with confidence interval (0.0803, 0.1086)
rouge_1_recall: 0.0681 with confidence interval (0.0559, 0.0773)
rouge_1_precision: 0.1705 with confidence interval (0.1497, 0.1928)

ROUGE-2:
rouge_2_f_score: 0.0082 with confidence interval (0.0029, 0.0138)
rouge_2_recall: 0.0058 with confidence interval (0.0021, 0.0097)
rouge_2_precision: 0.0141 with confidence interval (0.0046, 0.0238)

ROUGE-l:
rouge_l_f_score: 0.0808 with confidence interval (0.0659, 0.0972)
rouge_l_recall: 0.0571 with confidence interval (0.0461, 0.0678)
rouge_l_precision: 0.1452 with confidence interval (0.1166, 0.1737)



## ESA - Explict Structura Attenion 

ROUGE-1:
rouge_1_f_score: 0.1431 with confidence interval (0.1102, 0.1753)
rouge_1_recall: 0.0897 with confidence interval (0.0679, 0.1102)
rouge_1_precision: 0.3679 with confidence interval (0.2858, 0.4438)

ROUGE-2:
rouge_2_f_score: 0.0110 with confidence interval (0.0029, 0.0197)
rouge_2_recall: 0.0069 with confidence interval (0.0018, 0.0123)
rouge_2_precision: 0.0279 with confidence interval (0.0083, 0.0481)

ROUGE-l:
rouge_l_f_score: 0.1210 with confidence interval (0.0931, 0.1481)
rouge_l_recall: 0.0757 with confidence interval (0.0582, 0.0925)
rouge_l_precision: 0.3125 with confidence interval (0.2435, 0.3782)

