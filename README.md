<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp
</p>

# Minimalist-TinyLLaMA-to-Onnx

This is a minimalist deployment plan for the [Tiny-LLaMA-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model on edge devices. The project has the following contributions:

- The 1.1B model is exported to a 2GB FP16 ONNX file to prevent the creation of external onnx weights.
- All dynamic tensor shapes introduced by Large-Language Model (LLM) real-time inference are converted to static shapes and padded as needed.
- A mechanism is implemented to prevent value overflow in RMSNorm when using FP16 precision.

**If my open source projects have inspired you, giving me some sponsorship will be a great help to my subsequent open source work.** 
[Support my subsequent open source work❤️🙏](https://kaihuatang.github.io/donate.html) [(Previous Supporters)](https://kaihuatang.github.io/supporters.html)


## Contents
1. [Getting Started](#getting-started)
2. [Citation](#citation)


## Getting Started
Simply Run the following command to export Tiny LLaMA to onnx:

```
python onnx_export.py --model_path YOUR_LLAMA_PATH --output_path YOUR_ONNX_PATH --max_length 1024
```
Explaination of arguments used in the script
```
model_path: your Tiny LLaMA checkpoint and config path
output_path: save all the files for onnx inference
max_length: the maximum inference length for LLM
```

To inference using onnx, simply run:
```
python onnx_inference.py --model_path YOUR_LLAMA_PATH --output_path YOUR_ONNX_PATH --max_length 1024
```
Explaination of arguments used in the script
```
max_length: it has to be the same as used in export.
(optional)dump_index: dump kv cache for i-th token to diagnose 
```

## Citation
If you find this project helps your research, please kindly consider citing our project in your publications.

```
@misc{tang2024tinyllama2onnx,
    title = {Minimalist TinyLLaMA to Onnx},
    author = {Tang, Kaihua},
    year = {2024},
    note = {\url{https://github.com/KaihuaTang/Minimalist-TinyLLaMA-to-Onnx}},
}
```