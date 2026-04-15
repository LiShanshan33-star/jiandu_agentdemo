import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import io
import os
import json
import base64
from datetime import datetime
import plotly.express as px

# 1. 直接导入 AI 助手逻辑（不再通过 Flask 调用）
try:
    from ai_assistant import smart_correction, incomplete_completion, generate_report, polish_result
except ImportError:
    st.error("找不到 ai_assistant.py 文件，请确保该文件在仓库根目录中。")

# ========== 1. 模型加载 (云端部署核心：使用缓存) ==========
@st.cache_resource
def load_yolo_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error(f"找不到模型文件: {model_path}，请检查是否已上传至 GitHub。")
        return None
    return YOLO(model_path)

model = load_yolo_model()

# ========== 2. 后端逻辑本地化 (原 Flask 函数直接移植) ==========

def apply_clahe_preprocessing(image_bytes):
    """原 Flask 中的图像增强逻辑"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

def local_predict_logic(img_bytes, conf_thres):
    """替代原有的 call_predict_api 接口请求"""
    processed_rgb = apply_clahe_preprocessing(img_bytes)
    if processed_rgb is None:
        return None
    
    # YOLO 推理
    results = model.predict(source=processed_rgb, conf=conf_thres, save=False)
    res = results[0]
    
    predictions = []
    if res.boxes is not None:
        for box in res.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            char_name = model.names[class_id]
            xyxy = box.xyxy[0].tolist()
            predictions.append({
                "char": char_name,
                "confidence": round(conf, 4),
                "bbox": [int(x) for x in xyxy]
            })
    
    # 排序
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 模拟原 Flask 返回的 base64 图像，保持 home.py 逻辑兼容
    enhanced_pil = Image.fromarray(processed_rgb)
    buffered = io.BytesIO()
    enhanced_pil.save(buffered, format="JPEG")
    enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "status": "success",
        "count": len(predictions),
        "results": predictions,
        "enhanced_image_base64": enhanced_base64
    }

# ========== 3. Streamlit 界面配置 ==========
st.set_page_config(page_title="简牍文字智能识别系统", page_icon="🏺", layout="wide")

# 初始化 Session State
if 'history' not in st.session_state: st.session_state.history = []
if 'ocr_results' not in st.session_state: st.session_state.ocr_results = None
if 'ai_display_content' not in st.session_state: st.session_state.ai_display_content = None

# 辅助函数：绘图
def draw_boxes_on_image(image_pil, predictions):
    draw = ImageDraw.Draw(image_pil)
    try:
        # 尝试加载中文字体，云端如果没有会回退到默认
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    for pred in predictions:
        box = pred['bbox']
        draw.rectangle(box, outline="red", width=3)
        label = f"{pred['char']} {int(pred['confidence'] * 100)}%"
        draw.text((box[0], box[1] - 25), label, fill="red", font=font)
    return image_pil

# ========== 侧边栏导航 ==========
st.sidebar.title("🏺 简牍文字识别系统")
page = st.sidebar.radio("导航菜单", ["📖 简牍科普与背景", "📊 DeepJiandu 数据集", "🔍 智能上传与识别"])

# ========== 页面实现 ==========

if page == "📖 简牍科普与背景":
    st.title("📖 探秘简牍：穿越千年的文字密码")
    st.markdown("> **“书于竹帛，镂于金石。”**")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("简牍是记录中华文明薪火相传的“古代硬盘”。本项目利用 AI 技术对其进行数字化保护与释读。")
    with col2:
        if os.path.exists("jiandu.jpg"):
            st.image("jiandu.jpg", caption="出土简牍实物")

elif page == "📊 DeepJiandu 数据集":
    st.title("📊 数据集概览")
    # 模拟数据展示
    c1, c2, c3 = st.columns(3)
    c1.metric("总图像数", "7,416")
    c2.metric("字符类别", "2,242")
    c3.metric("总字符框", "99,852")
    # 如果有 csv 可以在这里展示 px.bar...

elif page == "🔍 智能上传与识别":
    st.title("🔍 在线体验：简牍文字智能识别")
    conf_thres = st.slider("置信度阈值", 0.01, 1.0, 0.25, 0.01)
    uploaded_file = st.file_uploader("上传简牍图像", type=['bmp','jpg','png','jpeg'])

    if uploaded_file:
        original_img = Image.open(uploaded_file).convert("RGB")
        col_left, col_mid, col_right = st.columns(3)
        with col_left:
            st.image(original_img, caption="原始图像", use_container_width=True)

        if st.button("🚀 开始识别", type="primary", use_container_width=True):
            with st.spinner("模型正在进行视觉分析..."):
                img_bytes = uploaded_file.getvalue()
                # 直接调用本地函数，不再发送 requests 请求！
                result = local_predict_logic(img_bytes, conf_thres)
                
                if result:
                    st.session_state.ocr_results = result
                    st.session_state.ai_display_content = None
                    # 保存历史
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "filename": uploaded_file.name,
                        "num_chars": result['count']
                    })

        # 结果渲染
        if st.session_state.ocr_results:
            res = st.session_state.ocr_results
            preds = res.get('results', [])
            
            # 显示增强图
            if res.get('enhanced_image_base64'):
                enhanced_img = Image.open(io.BytesIO(base64.b64decode(res['enhanced_image_base64'])))
                with col_mid: st.image(enhanced_img, caption="CLAHE 增强后", use_container_width=True)
            
            # 显示检测图
            annotated_img = draw_boxes_on_image(original_img.copy(), preds)
            with col_right: st.image(annotated_img, caption="识别结果", use_container_width=True)

            # AI 助手区
            st.markdown("---")
            st.subheader("🤖 AI 智能释读助手 (DeepSeek 驱动)")
            char_sequence = [p['char'] for p in preds]
            low_conf_indices = [i for i, p in enumerate(preds) if p['confidence'] < 0.6]

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔍 智能纠错与重校准", use_container_width=True):
                    raw = smart_correction(char_sequence, low_conf_indices) # 直接调用函数
                    st.session_state.ai_display_content = polish_result(raw, "correction")
            with c2:
                if st.button("🩹 残损字符智能补全", use_container_width=True):
                    raw = incomplete_completion(char_sequence)
                    st.session_state.ai_display_content = polish_result(raw, "completion")
            with c3:
                if st.button("📝 一键生成释读报告", use_container_width=True):
                    raw = generate_report(char_sequence, preds)
                    st.session_state.ai_display_content = raw.get("report")

            if st.session_state.ai_display_content:
                with st.container(border=True):
                    st.markdown("### 🏺 专家释读结论")
                    st.write(st.session_state.ai_display_content)
                    if st.button("🗑️ 关闭显示"):
                        st.session_state.ai_display_content = None
                        st.rerun()

# 历史记录展示
if st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 最近识别记录")
    st.sidebar.table(pd.DataFrame(st.session_state.history).tail(5))
