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



[1]: https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation
