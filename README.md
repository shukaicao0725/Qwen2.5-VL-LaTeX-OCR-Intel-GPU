# 📝 Intel Arc LaTeX OCR

这是一个专为 **Intel Core Ultra (Meteor Lake)** 处理器优化的本地离线 LaTeX 公式识别工具。

## ✨ 项目亮点

- **零配置粘贴**：直接从剪贴板读取截图，识别流程极简。
- **硬件级加速**：调用 Core Ultra 5 的 **Intel Arc GPU**，利用 OpenVINO™ 实现秒级推理。
- **纯净 LaTeX**：内置正则表达式提取逻辑，过滤模型废话，直接输出标准代码并支持预览。

## 🛠 1. 环境准备

建议使用 Python 3.11。

Bash

```
# 创建并激活环境
conda create -n qwen_latex python=3.11 -y
conda activate qwen_latex

# 安装依赖项
pip install -r requirements.txt
```

> **注意**：除常规库外，本项目还依赖 `streamlit-paste-button` 以实现截图粘贴功能。

## 🚀 2. 模型导出 (核心步骤)

由于代码中直接加载量化后的模型，你需要运行以下命令将 **Qwen2.5-VL-3B** 转换为 **OpenVINO INT4** 格式：

Bash

```
optimum-cli export openvino \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --task image-text-to-text \
    --weight-format int4 \
    ./qwen_ov_model
```

*请务必确保导出目录名为 `qwen_ov_model`，否则程序无法启动。*

## 🖥 3. 运行应用

Bash

```
streamlit run latex_app.py
```

## ⚙️ 适配说明

- **硬件检查**：启动时程序会自动检测 `./qwen_ov_model` 目录。
- **显存优化**：针对 Core Ultra 5 的 Arc GPU，采用了 INT4 量化，运行时约占用 2.2GB 共享显存。
- **首次推理**：首次执行识别时，OpenVINO 会进行 GPU Shader 编译，耗时约 10-20 秒，后续识别将恢复秒级响应。