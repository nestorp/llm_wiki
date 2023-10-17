This guide sets out to create general guidelines for locally fine-tuning pre-trained large language models. It outlines the available tools, datasets and models that can be used and makes some recommendations. It also points to additional resources that can be useful for further understanding the fine-tuning process.

## Hardware
The rule of thumb is you need (Model Parameters in Billions) x 4 in GPU RAM GB for inference. Eg. for the Llama 2 7B version, you would need 7x4 = 28 GB of GPU RAM. For fine-tuning, the requirements increase up to double this amount, depending on the optimiser used. 

Low Rank Fine-tuning (LoRa) and Quantization are techniques aimed at optimising the use of resources by language models during both inference and training. They can allow for more efficient fine-tuning requiring fewer resources.
### Low Rank Fine-tuning (LoRa)
Low Rank Fine-tuning (LoRa) is a method designed to accelerate the training of large language models while conserving memory resources. LoRa incorporates pairs of rank-decomposition weight matrices, referred to as update matrices, to existing weights in the model, and only trains these newly added matrices.

- [Low-Rank Adaptation of Large Language Models (LoRA)](https://huggingface.co/docs/diffusers/main/en/training/lora)
- [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
### Quantization
Quantization is a method that reduces the computational and memory demands of model inference by representing weights and activations using low-precision data types, like 8-bit integers (int8), instead of the typical 32-bit floating point (float32). Despite the potential advantages, it's worth noting that aggressive quantization can sometimes degrade model performance

- [Quantization](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- [Quantize Transformers models](https://huggingface.co/docs/transformers/main/main_classes/quantization)

## Models

### Open Source

Open source models can be run on your own hardware and fine-tuned for specialist tasks.

- [Llama 2](https://ai.meta.com/llama/): Second version of the Llama model by Meta. Three variations of the model with different sizes have been released: 7B, 13B and 70B. For comparison, the largest Llama model was 65B. In most benchmarks, Llama 2 outperforms other open source LLMs such as Falcon.
-  [Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm): Developed by Databricks, Dolly 2.0, known as dolly-v2-12b, is an instruction-following model trained on around 15,000 instruction/response records. It was developed on the Databricks machine learning platform using the pythia-12b model. The models can be found [here](https://huggingface.co/databricks).
- [Falcon](https://falconllm.tii.ae/falcon.html): Developed by the Technology Innovation Institute in Abu Dhabi. Has been fine-tuned on a curated subset of the Common Crawl dataset, which the authors have released as the [falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) dataset. ⚠️ Note that the Huggingface website marks 6 files in this dataset as unsafe (containing trojans). Versions with 1.3B, 7B, 40B and 180B have been released.

### Closed models

Closed models can only be used through an (usually paid) API. Some may offer fine-tuning functionality through their API, but more commonly you'd use these models as-is. Currently, GPT4 offers the best performance at most tasks, which makes it useful for benchmarking.

- [ChatGPT](https://openai.com/blog/chatgpt) (GPT3.5 and GPT4): Access API [here](https://chat.openai.com/auth/login).
- [Claude 2](https://www.anthropic.com/index/claude-2): Developed by Anthropic. Access API [here](https://claude.ai/login).

## Datasets
### Instruction tuning
Instruction-tuning datasets are used for making language models better at responding to user prompts. 

- [awesome-text/visual-instruction-tuning-dataset](https://github.com/yaodongC/awesome-instruction-dataset#awesome-textvisual-instruction-tuning-dataset): A collection of open-source instruction tuning datasets to train (text and multi-modal) chat-based LLMs.
- [Super-Natural Instructions](https://instructions.apps.allenai.org/) A dataset of expert-created instructions for a wide variety of tasks, including classification, extraction, infilling, sequence tagging, text rewriting, and text composition. See [SUPER-NATURAL INSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705) for details.
- [unnatural-instructions-core](https://huggingface.co/datasets/mrm8488/unnatural-instructions-core) and [unnatural-instructions-full](https://huggingface.co/datasets/mrm8488/unnatural-instructions-full): Includes examples created by prompting a language model with three seed examples of instructions and eliciting a fourth. This set is then expanded by prompting the model to rephrase each instruction, creating a total of approximately 240,000 examples of instructions, inputs, and outputs. See [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689) for details.
- [Dolly 2.0 Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k): Dolly 2.0 is a language model fine-tuned on a human-generated instruction dataset. It was developed to exhibit ChatGPT-like human interactivity and instruction-following capabilities, with the dataset serving as a platform for fine-tuning to enhance the model's adherence to instructions​.
- [FLAN datasets](https://github.com/google-research/FLAN): compiles datasets from Flan 2021, P3, Super-Natural Instructions, along with dozens more datasets into one place, formats them into a mix of zero-shot, few-shot and chain-of-thought templates, then mixes these in proportions that are found to achieve strong results on held-out evaluation benchmarks, as reported for Flan-T5 and Flan-PaLM in the [Scaling Flan paper](https://arxiv.org/abs/2210.11416) and [Flan Collection paper](https://arxiv.org/abs/2301.13688).
## Fine-tuning tools
### 'Mainstream' tools:
- [Huggingface Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Huggingface Reward Modeling Trainer](https://huggingface.co/docs/trl/reward_trainer)

### Open source efficient fine-tuning tools:
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ): LLM quantization package based on GPTQ algorithm.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes): A lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and quantization functions.
- [exllama](https://github.com/turboderp/exllama): A standalone Python/C++/CUDA implementation of Llama for use with 4-bit GPTQ weights, designed to be fast and memory-efficient on modern GPUs.
- [llama.cpp](https://github.com/ggerganov/llama.cpp): Port of Facebook's LLaMA model in C/C++.
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, llama.cpp (GGUF), Llama models.

## Resources
https://huggingface.co/docs/transformers/perf_train_gpu_one
https://huggingface.co/blog/hf-bitsandbytes-integration
https://huggingface.co/blog/4bit-transformers-bitsandbytes
https://huggingface.co/blog/gptq-integration