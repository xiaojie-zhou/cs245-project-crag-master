# Documentation

## Methods

We implemented three distinct RAG systems. Each system is self-contained in a single file. Assuming you are currently located
in root directory of the repository, the systems can be located in:
* Document-Level Similarity Matching, written in `./course_code/file_levl_rag.py`
* Minimum Information Threshold, written in `./course_code/chunk_threshold.py`
* Enhanced Chunk Division and Selection, written in `./course_code/rag_enhanced.py`

## Generate Predictions

Before running the methods, make sure you've downloaded the `crag_task_1_dev_v4_release.jsonl.bz2` into `./data`. Then, use
the shell script `bench.sh` to invoke the RAG systems for generation. Note that `bench.sh` defines a series of environment variables:
* `CUDA_VISIBLE_DEVICES`, used to select the GPUs for generation.
* `THRESHOLD`, used to specify the hyperparameter `K` in the method `Minimum information Threshold`. This variable is only relevant for `Minimum information Threshold`, and will be ignored(therefore can use arbitrary values) in other methods.
* `METHOD`, used to specify which RAG system to use. Choose from `[vanilla_baseline, rag_baseline, file_levl_rag, chunk_threshold, rag_enhanced]`
* `LLM`, used to specify the generation LLM of the RAG system. See `./course_code/generate.py` for valid options.
* `DATA`, used to specify the location of the dataset. See `./course_code/generate.py` for valid options.
* `SPLIT`, used to specify which split of the dataset to work on. Consistent with the project spec notion page.

The generated responses will be located in `./output/data/` directory. The file structure of each trial will be

`./output/data/<METHOD>/<LLM>/...`

For each specification of the `bench.sh`.
Note that we manually organized the file structure for `chunk_threshold` so that its structure is like:

`./output/data/chunk_thresold/<LLM>/<K>/...`

## Evaluate the Predictions

Invoke `eval.sh`,  which takes the following parameters:
* `METHOD`, used to specify which RAG system to use. Choose from `[vanilla_baseline, rag_baseline, file_levl_rag, chunk_threshold, rag_enhanced]`
* `LLM`, used to specify the generation LLM of the RAG system. See `./course_code/evaluate.py` for valid options.
* `DATA`, used to specify the location of the dataset. See `./course_code/evaluate.py` for valid options.
* `RETRIES`, specify the maximum retry budgets for the evaluator LLM.

The evaluation results will share the same directory as the generated `predictions.json`. Again, we manually restructured that for `chunk_threshold`.