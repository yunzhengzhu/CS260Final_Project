# CS260Final_Project: Commonsense Explanation for Counter-factual Arguments

This project is from [SemEval 2020 Task 4 - Commonsense Validation and Explanation (ComVE)](https://competitions.codalab.org/competitions/21080) Subtask C.
The dataset for this project is from [1].

## Our Experiments:
Approaches: 
1. NMT
2. GPT-2
3. KaLM(Bart-Large)
4. GRF


## Project Structure
* `data_dir`: experiment results with GPT-2:
    *  `subtaskC_answers_CONFIG.csv` are the files of the generated dev results
    *  `subtaskC_answers_test_CONFIG.csv` are the files of the generated test results
* `GPT-2`: please refer to the readme under this directory
* `NMT`: please refer to the readme under this directory
* `KaLM`: please refer to the readme under this directory
* `graph_method`: GRF is implemented under this directory, please refer to the readme under this directory

## Evaluation Results:
System                  | Dev(BLEU)       | Test(BLEU)         | Human              | PT LM Params
:----------             |----------:      |----------:         |----------:         |----------:
Copy Source             | 16.53           | 17.23              | N.A.               | N.A.
NMT                     | around 1.0      | around 1.0         | 0.59               | N.A. 
GPT2-small              | 14.01           | 13.66              | 1.80               | 117 M
GPT2-medium             | 16.79           | 16.08              | 1.94               | 345 M
KaLM (BART-Large)       | 18.98           | **18.5**           | 2.08               | **406 M**
GRF (GPT2-small)        | 17.83           | 17.56              | 1.91               | 117 M
Muti-Task (GPT2-small)  | 17.11           | 17.82              | 1.97               | 117 M
GRF (GPT2-medium)       | 18.36           | 18.45              | **2.21**           | 345 M
Muti-Task (GPT2-medium) | **19.15**       | 18.31              | **2.26**           | 345 M 

- Copy Source - from the original data

[1]: https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation
