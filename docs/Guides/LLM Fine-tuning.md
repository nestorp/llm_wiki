This guide sets out to create general guidelines for locally fine-tuning pre-trained large language models. It outlines the available tools, datasets and models that can be used and makes some recommendations. It also points to additional resources that can be useful for further understanding the fine-tuning process.

## Hardware
The rule of thumb is you need (Model Parameters in Billions) x 4 in GPU RAM GB for inference. Eg. for the Llama 2 7B version, you would need 7x4 = 28 GB of GPU RAM. 

For fine-tuning, the requirements increase up to double, depending on the optimizer used. 

Quantization can be useful to reduce the memory requirements. This technique involves reducing the precision of the model weights, which makes it lighter, but negatively impacts performance. 

Using 16 bit quantization cuts the required RAM in half, meaning the same Llama 2 7B could be fine-tuned on a 14 GB GPU. Further quantization to 8 bit or even 4 bit 

## Models

## Datasets

## Libraries
### 'Mainstream' tools:
- [Huggingface Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Huggingface Reward Modeling Trainer](https://huggingface.co/docs/trl/reward_trainer)

### Open source fine-tuning tools:
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [exllama](https://github.com/turboderp/exllama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)




## Resources
https://huggingface.co/docs/transformers/perf_train_gpu_one
https://huggingface.co/blog/hf-bitsandbytes-integration
https://huggingface.co/blog/4bit-transformers-bitsandbytes
https://huggingface.co/blog/gptq-integration