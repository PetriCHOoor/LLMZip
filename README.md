
# LLMZip-大语言模型文本压缩与重建


## LLMZip
本程序使用**大语言模型（Large-Language-Model）**对文本进行压缩，使用大预言模型预测结合传统文本压缩的方法，可实现对文本的**高倍率**的**无损**压缩与重建

## 使用介绍

### **1.压缩界面**
在压缩界面，从上到下依次可以选择压缩所使用的模型，被压缩文件(.txt)的路径，压缩完成后文件的保存路径以及进行记忆(Memory)的设置。在所有这些都设置好后，点击开始压缩，本程序即可对所选文件进行无损压缩。

### **2.解压界面**
在解压界面，从上到下依次可以选择被压缩文件(.bin)的路径，解压完成后文件的保存路径。在这两项设置好后，点击开始解压，本程序即可对所选的被压缩文件进行自动的无损解压，复原为原先的文本文件。

### **3.设置界面**
在设置界面，可以对所提供的五种模型进行路径的选择，模型运行设备的选择，模型加载与释放操作。<br>

### **4.模型下载与使用**
#### 模型的下载地址如下：(注意必须下载pytorch版本的模型)
* LLaMA2-7B模型下载教程：https://www.bilibili.com/video/BV1H14y1X7kD (建议使用Gmail进行申请，不建议使用QQ邮箱，可能会申请不上) <br>
* GPT2-Small模型下载地址：https://huggingface.co/openai-community/gpt2
* GPT2-Medium模型下载地址：https://huggingface.co/openai-community/gpt2-medium
* GPT2-Large模型下载地址：https://huggingface.co/openai-community/gpt2-large
* GPT2-XL模型下载地址：https://huggingface.co/openai-community/gpt2-xl<br>
#### 模型文件夹的格式如下：
LLaMA模型按照如下方式放在一个文件夹中：
* checklist.chk
* consolidated.00.pth
* params.json
* tokenizer_checklist.chk
* tokenizer.model 

GPT2模型按照如下方式放在一个文件夹中：
* config.json
* merges.txt
* pytorch_model.bin
* tokenizer_config.json
* tokenizer.json
* vocab.json

_**注：**_**_不同版本GPT2模型的config和pytorch_model不同，其余文件通用_**
