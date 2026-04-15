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
import time
from datetime import datetime
import plotly.express as px

# --- 核心：直接导入 AI 助手的所有功能函数 ---
try:
    from ai_assistant import smart_correction, incomplete_completion, generate_report, polish_result
except ImportError:
    st.error("❌ 严重错误：未能在当前目录找到 ai_assistant.py，请检查 GitHub 仓库文件。")

# ==================== 1. 系统底层配置与模型加载 ====================

st.set_page_config(
    page_title="简牍文字智能识别系统", 
    page_icon="🏺", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_yolo_model():
    """在云端缓存模型，避免重复加载导致内存溢出"""
    MODEL_PATH = 'best.pt'
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.error(f"无法找到模型文件 {MODEL_PATH}")
        return None

model = load_yolo_model()

# ==================== 2. 图像处理与识别核心逻辑 (原 Flask 移植) ====================

def apply_clahe_preprocessing(image_bytes):
    """CLAHE 增强算法实现"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    # 转换为 RGB 供 PIL/Streamlit 使用
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

def local_prediction(img_bytes, conf_thres):
    """直接在 Streamlit 内部驱动 YOLO 推理，不再请求本地 7000 端口"""
    processed_rgb = apply_clahe_preprocessing(img_bytes)
    if processed_rgb is None:
        return {"status": "error", "message": "图片解码失败"}

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

    # 按置信度排序
    predictions.sort(key=lambda x: x['confidence'], reverse=True)

    # 将增强后的图像转为 base64 以兼容原有 UI 渲染逻辑
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

# ==================== 3. 辅助功能函数 ====================

def load_metrics():
    if os.path.exists("eval_metrics.json"):
        with open("eval_metrics.json", "r") as f:
            return json.load(f)
    return {"mAP50": 0.2724, "mAP50-95": 0.18, "precision": 0.51, "recall": 0.23}

def load_category_freq():
    if os.path.exists("category_freq.csv"):
        return pd.read_csv("category_freq.csv")
    return None

def draw_boxes_on_image(image_pil, predictions, font_path=None):
    draw = ImageDraw.Draw(image_pil)
    try:
        # 云端 Linux 环境尝试加载常用中文字体
        font = ImageFont.truetype("simsun.ttc", 20)
    except:
        font = ImageFont.load_default()
        
    for pred in predictions:
        box = pred['bbox']
        draw.rectangle(box, outline="red", width=3)
        label = f"{pred['char']} {int(pred['confidence'] * 100)}%"
        draw.text((box[0], box[1] - 25), label, fill="red", font=font)
    return image_pil

# ==================== 4. 初始化 Session State ====================

if 'history' not in st.session_state:
    st.session_state.history = []
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = None
if 'ai_display_content' not in st.session_state:
    st.session_state.ai_display_content = None

# ==================== 5. 侧边栏布局 ====================

st.sidebar.title("🏺 简牍文字识别系统")
st.sidebar.markdown("---")
page = st.sidebar.radio("导航菜单", ["📖 简牍科普与背景", "📊 DeepJiandu 数据集", "🔍 智能上传与识别"])
st.sidebar.markdown("---")
st.sidebar.info("📌 **项目信息**\n\n参赛赛道：大数据实践赛\n\n核心架构：Streamlit + YOLOv8 + CLAHE + DeepSeek AI")

# ==================== 6. 页面一：简牍科普与背景 ====================

if page == "📖 简牍科普与背景":
    st.title("📖 探秘简牍：穿越千年的文字密码与文化传承")
    st.markdown("""
    > **“书于竹帛，镂于金石。”**
    在纸张发明并大规模普及之前，中国先民最主要的文字载体是简牍。它们深埋地下两千年，是记录中华文明薪火相传的“古代硬盘”。
    """)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🏺 旷世遗珍：什么是简牍？
        **“简”** 指狭长的竹木片；**“牍”** 指较宽的木板。本项目依托源自著名的**西北汉简**。这些简牍记录了汉代丝绸之路上的戍边生活、律令诏书与驿站往来。

        ### 🕰️ 岁月侵蚀：千年的痛点
        历经千年的地下水土浸泡，出土简牍面临墨迹脱落、木纹干扰等严峻问题。传统的释读工作极度依赖专家肉眼辨认，效率低下。

        ### 🚀 科技赋能：跨越时空的智能对话
        本项目构建了一套融合 **CLAHE 增强** 与 **YOLOv8** 的智能 OCR 流水线，并引入 **DeepSeek 大模型** 作为智能释读助手，实现对古代汉字的精准定位、识别与语义理解。
        """)
    with col2:
        if os.path.exists("jiandu.jpg"):
            st.image("jiandu.jpg", caption="出土简牍实物：岁月斑驳，墨迹隐现", use_container_width=True)
        else:
            st.warning("⚠️ 提示：请确保 GitHub 仓库中有 jiandu.jpg 以展示科普图片。")

# ==================== 7. 页面二：数据集展示 ====================

elif page == "📊 DeepJiandu 数据集":
    st.title("📊 DeepJiandu 数据集概览")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总图像数", "7,416")
    col2.metric("总字符框", "99,852")
    col3.metric("字符类别", "2,242")
    col4.metric("划分比例", "8:1:1")

    st.subheader("类别频率分布（Top 20）")
    freq_df = load_category_freq()
    if freq_df is not None:
        top20 = freq_df.head(20)
        fig = px.bar(top20, x='char', y='count', title="高频字符出现次数", color='count')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 暂无类别频率数据，请将 category_freq.csv 上传至仓库。")

    st.subheader("样本真值展示")
    meta_path = "data/gallery_samples/gallery_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            gallery = json.load(f)
        options = {f"[{item['id']}] {item['image_name']}": item for item in gallery}
        sel = st.selectbox("选择样本预览", list(options.keys()))
        item = options[sel]
        st.info(f"📜 专家释文：`{item['text_content']}`")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(item['raw_path']):
                st.image(item['raw_path'], caption="原始图像", width=400)
        with c2:
            if os.path.exists(item['annotated_path']):
                st.image(item['annotated_path'], caption="标注真值", width=400)
    else:
        st.warning("⚠️ 未找到 gallery_meta.json 或对应图片，请检查 data/gallery_samples 路径。")

# ==================== 8. 页面三：智能上传与识别 ====================

elif page == "🔍 智能上传与识别":
    st.title("🔍 在线体验：简牍文字智能识别")
    
    # 交互控制区
    conf_thres = st.slider("识别置信度阈值", 0.01, 1.0, 0.25, 0.01)
    uploaded_file = st.file_uploader("上传简牍图像进行智能扫描", type=['bmp','jpg','png','jpeg'])
    
    if uploaded_file:
        original_img = Image.open(uploaded_file).convert("RGB")
        col_left, col_mid, col_right = st.columns(3)
        with col_left:
            st.image(original_img, caption="原始图像", use_container_width=True)
        
        # --- 识别逻辑执行 ---
        if st.button("🚀 开始识别", type="primary", use_container_width=True):
            with st.spinner("正在执行 CLAHE 增强处理与 YOLOv8 深度推理..."):
                img_bytes = uploaded_file.getvalue()
                # 调用本地合并的逻辑，不再请求外部 API
                result = local_prediction(img_bytes, conf_thres)
                
                if result and result.get('status') == 'success':
                    st.session_state.ocr_results = result
                    st.session_state.ai_display_content = None
                    
                    # 历史记录保存
                    preds = result.get('results', [])
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "num_chars": len(preds),
                        "chars": " ".join([p['char'] for p in preds])
                    })
                else:
                    st.error("识别过程出错，请检查日志。")

        # --- 结果渲染展示 ---
        if st.session_state.ocr_results:
            res = st.session_state.ocr_results
            preds = res.get('results', [])
            
            # 1. 渲染增强图像
            if res.get('enhanced_image_base64'):
                enhanced_bytes = base64.b64decode(res['enhanced_image_base64'])
                enhanced_img = Image.open(io.BytesIO(enhanced_bytes))
                with col_mid:
                    st.image(enhanced_img, caption="CLAHE 增强效果", use_container_width=True)
            
            # 2. 渲染检测标注图
            annotated_img = original_img.copy()
            annotated_img = draw_boxes_on_image(annotated_img, preds)
            with col_right:
                st.image(annotated_img, caption="智能识别标注结果", use_container_width=True)
            
            # 3. 数据分析图表
            if preds:
                st.markdown("---")
                st.subheader("📊 字符置信度与详细数据")
                df_pred = pd.DataFrame(preds)
                fig = px.bar(df_pred, x='char', y='confidence', color='confidence', 
                             title="各识别字符置信度分布", text='confidence',
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("查看详细坐标与置信度列表"):
                    st.dataframe(df_pred[['char', 'confidence', 'bbox']], use_container_width=True)
            else:
                st.warning("⚠️ 未能在此置信度下检测到任何有效字符，请尝试调低阈值。")

            # ==================== 9. AI 助手交互区 (DeepSeek 驱动) ====================
            
            st.markdown("---")
            st.subheader("🤖 AI 智能释读助手")
            st.caption("由 DeepSeek-V3 提供语义推理与古籍大数据校准支持")
            
            char_sequence = [p['char'] for p in preds]
            low_conf_indices = [i for i, p in enumerate(preds) if p['confidence'] < 0.6]

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔍 智能纠错与重校准", use_container_width=True, help="针对形近字进行上下文逻辑修正"):
                    with st.spinner("专家正在进行语义校准..."):
                        raw = smart_correction(char_sequence, low_conf_indices)
                        if raw:
                            st.session_state.ai_display_content = polish_result(raw, "correction")
                        st.rerun()

            with c2:
                if st.button("🩹 残损字符智能补全", use_container_width=True, help="基于历史语境推测缺失文本"):
                    with st.spinner("正在检索古籍数据库补全语义..."):
                        raw = incomplete_completion(char_sequence)
                        if raw:
                            st.session_state.ai_display_content = polish_result(raw, "completion")
                        st.rerun()

            with c3:
                if st.button("📝 一键生成释读报告", use_container_width=True, help="生成包含统计与结论的专业鉴定报告"):
                    with st.spinner("系统正在撰写综合鉴定报告..."):
                        raw = generate_report(char_sequence, preds)
                        if raw and raw.get("report"):
                            st.session_state.ai_display_content = raw.get("report")
                        st.rerun()
            
            # 渲染 AI 结论文本框
            if st.session_state.ai_display_content:
                st.write("") 
                with st.container(border=True):
                    st.markdown("### 🏺 专家释读结论")
                    st.markdown("---")
                    st.markdown(str(st.session_state.ai_display_content))
                    st.write("") 
                    if st.button("🗑️ 关闭释读结果"):
                        st.session_state.ai_display_content = None
                        st.rerun()

    # --- 历史记录卡片 ---
    if st.session_state.history:
        st.markdown("---")
        with st.expander("📋 查看最近识别历史"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.table(hist_df.tail(10))
            if st.button("清空历史数据"):
                st.session_state.history = []
                st.rerun()
