<p align="left">
    中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
</p>

中文版说明由ChatGPT-4o翻译并由人工校验。

# Minimalist-TinyLLaMA-to-Onnx
这是一个针对端侧设备上[Tiny-LLaMA-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)模型的极简部署计划。该项目的主要贡献如下：
- 将 1.1B 模型导出为一个 2GB 的 FP16 ONNX 文件，以防止创建外部 ONNX 权重。
- 所有由大语言模型 (LLM) 实时推理引入的动态张量形状都被转换为静态形状，并根据需要进行填充。
- 实现了一种机制，以防止在使用 FP16 精度时 RMSNorm 中的值溢出。

**如果我的开源项目给您带来了启发，提供一些赞助将对我后续的开源工作有很大的帮助。** 
[支持我的后续开源工作❤️🙏](https://kaihuatang.github.io/donate.html) [(往期支持者)](https://kaihuatang.github.io/supporters.html)

## Contents
1. [使用指南](#使用指南)
2. [引用](#引用)

## 使用指南
只需运行以下命令即可将 Tiny LLaMA 导出为 ONNX：

```
python onnx_export.py --model_path YOUR_LLAMA_PATH --output_path YOUR_ONNX_PATH --max_length 1024
```
脚本中参数的解释：
```
model_path: 您的 Tiny LLaMA 检查点和配置文件路径
output_path: 保存用于 ONNX 推理的所有文件的路径
max_length: LLM 的最大推理长度
```

要使用 ONNX 进行推理，只需运行：
```
python onnx_inference.py --model_path YOUR_LLAMA_PATH --output_path YOUR_ONNX_PATH --max_length 1024
```
Explaination of arguments used in the script
```
max_length: 必须与导出时使用的相同。
dump_index（可选）: 转储第 i 个toekn的 kv cache 以进行诊断。
```

## 引用
如果您发现此项目对您的研究有所帮助，请考虑在您的出版物中引用我们的项目。

```
@misc{tang2024tinyllama2onnx,
    title = {Minimalist TinyLLaMA to Onnx},
    author = {Tang, Kaihua},
    year = {2024},
    note = {\url{https://github.com/KaihuaTang/Minimalist-TinyLLaMA-to-Onnx}},
}
```