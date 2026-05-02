import streamlit as st
import re
from PIL import Image
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from streamlit_paste_button import paste_image_button

# ==========================================
# 1. 页面配置与样式优化
# ==========================================
st.set_page_config(page_title="Intel Arc LaTeX OCR", page_icon="📝", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stCodeBlock { border: 1px solid #0078d4; }
    </style>
    """, unsafe_allow_html=True)

st.title("📝 本地 LaTeX 公式识别工具")
st.caption("设备: Intel Core Ultra 5 125H | 加速: OpenVINO (Intel Arc GPU)")

# ==========================================
# 2. 模型加载 (缓存机制)
# ==========================================
@st.cache_resource
def load_model():
    model_dir = "./qwen_ov_model"
    # device="GPU" 对应你的 Intel Arc 核显
    model = OVModelForVisualCausalLM.from_pretrained(
        model_dir, 
        device="GPU", 
        ov_config={"PERFORMANCE_HINT": "LATENCY"}
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor

try:
    model, processor = load_model()
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.stop()

# ==========================================
# 3. 核心工具函数：精确提取 LaTeX
# ==========================================
def extract_latex(text):
    """
    从模型输出中提取 LaTeX 核心内容，剔除多余解释
    """
    # 匹配优先级：```latex 代码块 > $$ 双美元 > $ 单美元
    patterns = [
        r"```latex\s*([\s\S]*?)\s*```",
        r"\$\$\s*([\s\S]*?)\s*\$\$",
        r"\$\s*([\s\S]*?)\s*\$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # 如果没匹配到，尝试剔除常见的引导词
    clean_text = re.sub(r'^(这里是识别结果：|Here is the LaTeX:|LaTeX:)', '', text, flags=re.IGNORECASE)
    return clean_text.strip()

# ==========================================
# 4. 主界面交互
# ==========================================
st.write("---")
# 粘贴按钮
paste_result = paste_image_button(label="📋 点击此处粘贴剪贴板截图 (Ctrl+V)")

if paste_result.image_data is not None:
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("🖼️ 输入图像")
        image = paste_result.image_data
        st.image(image, use_container_width=True)
    
    with right_col:
        st.subheader("🔢 识别结果")
        with st.spinner("Intel Arc 正在全力计算中..."):
            try:
                # 构造对话消息
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Extract all mathematical formulas from this image and convert them to LaTeX format. Return only the LaTeX code wrapped in $$. No explanations."},
                        ],
                    }
                ]

                # 处理输入
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cpu")

                # 模型生成
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                raw_output = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0]

                # 提取纯净 LaTeX
                final_latex = extract_latex(raw_output)

                # 展示代码
                st.code(final_latex, language="latex")

                # 渲染预览
                st.write("---")
                st.write("**公式预览:**")
                if final_latex:
                    # 使用双美元符号包裹确保渲染成功率
                    st.latex(final_latex)
                else:
                    st.warning("未能提取到有效的 LaTeX 代码，请尝试重新截图。")

            except Exception as e:
                st.error(f"识别出错: {e}")

# ==========================================
# 5. 侧边栏：状态监控
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统配置")
    st.info(f"处理器: Core Ultra 5 125H\n\n显卡: Intel Arc Graphics\n\n内存: 32GB")
    if st.button("♻️ 清理缓存并重启"):
        st.cache_resource.clear()
        st.rerun()