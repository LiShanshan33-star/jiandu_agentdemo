# home.py
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import base64
import json
import os
import time
from datetime import datetime
import plotly.express as px

# ========== 配置 ==========
API_URL = "http://127.0.0.1:7000/predict"
API_SMART_CORRECTION = "http://127.0.0.1:7000/smart_correction"
API_INCOMPLETE_COMPLETION = "http://127.0.0.1:7000/incomplete_completion"
API_GENERATE_REPORT = "http://127.0.0.1:7000/generate_report"
API_POLISH = "http://127.0.0.1:7000/polish" 
# --------------------

st.set_page_config(page_title="简牍文字智能识别系统", page_icon="🏺", layout="wide", initial_sidebar_state="expanded")

# ========== 侧边栏 ==========
st.sidebar.title("🏺 简牍文字识别系统")
st.sidebar.markdown("---")
page = st.sidebar.radio("导航菜单", ["📖 简牍科普与背景", "📊 DeepJiandu 数据集", "🔍 智能上传与识别"])
# st.sidebar.markdown("---")
# st.sidebar.info("📌 **项目信息**\n\n参赛赛道：大数据实践赛\n\n核心架构：Flask + YOLOv8 + CLAHE + DeepSeek AI")

# 全局初始化历史记录与状态
if 'history' not in st.session_state:
    st.session_state.history = []
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = None
if 'ai_display_content' not in st.session_state:
    st.session_state.ai_display_content = None

# ========== 辅助函数 ==========
def load_metrics():
    if os.path.exists("eval_metrics.json"):
        with open("eval_metrics.json", "r") as f:
            return json.load(f)
    return {"mAP50": 0.75, "mAP50-95": 0.48, "precision": 0.78, "recall": 0.69}


def load_category_freq():
    if os.path.exists("category_freq.csv"):
        return pd.read_csv("category_freq.csv")
    return None


def call_predict_api(img_bytes, conf_thres, return_heatmap=False):
    files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
    data = {"conf": conf_thres, "heatmap": str(return_heatmap).lower()}
    try:
        response = requests.post(API_URL, files=files, data=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API错误: {response.text}")
            return None
    except Exception as e:
        st.error(f"连接后端失败: {e}")
        return None


def call_ai_function(endpoint, payload):
    """通用AI功能调用"""
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"AI服务返回错误: {response.text}")
            return None
    except Exception as e:
        st.error(f"连接AI服务失败: {e}")
        return None


def draw_boxes_on_image(image_pil, predictions, font_path=None):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("simsun.ttc", 20) if font_path is None else ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()
    for pred in predictions:
        box = pred['bbox']
        draw.rectangle(box, outline="red", width=3)
        label = f"{pred['char']} {int(pred['confidence'] * 100)}%"
        draw.text((box[0], box[1] - 25), label, fill="red", font=font)
    return image_pil


# ========== 页面一：简牍科普与背景 ==========
if page == "📖 简牍科普与背景":
    st.title("📖 探秘简牍：穿越千年的文字密码与文化传承")
    st.markdown("""
    > **“书于竹帛，镂于金石。”**
    在纸张发明并大规模普及之前，简牍是中国先民最主要的文字载体。它们深埋地下两千年，是记录中华文明薪火相传的“古代硬盘”。
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
        try:
            st.image("jiandu.jpg", caption="出土简牍实物：岁月斑驳，墨迹隐现", use_container_width=True)
        except:
            st.warning("⚠️ 建议放置一张 jiandu.jpg 增加美观度。")

# ========== 页面二：数据集展示 ==========
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
        fig = px.bar(top20, x='char', y='count', title="高频字符出现次数")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无类别频率数据，请运行预处理生成 category_freq.csv")

    st.subheader("样本真值展示")
    meta_path = "data/gallery_samples/gallery_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            gallery = json.load(f)
        options = {f"[{item['id']}] {item['image_name']}": item for item in gallery}
        sel = st.selectbox("选择样本", list(options.keys()))
        item = options[sel]
        st.info(f"📜 专家释文：`{item['text_content']}`")
        c1, c2 = st.columns(2)
        with c1:
            st.image(item['raw_path'], caption="原始图像", width=400)
        with c2:
            st.image(item['annotated_path'], caption="标注真值", width=400)
    else:
        st.warning("未找到 gallery_meta.json")

elif page == "🔍 智能上传与识别":
    st.title("🔍 在线体验：简牍文字智能识别")
    conf_thres = st.slider("置信度阈值", 0.01, 1.0, 0.25, 0.01)
    uploaded_file = st.file_uploader("上传简牍图像", type=['bmp','jpg','png','jpeg'])
    
    if uploaded_file:
        original_img = Image.open(uploaded_file).convert("RGB")
        col_left, col_mid, col_right = st.columns(3)
        with col_left:
            st.image(original_img, caption="原始图像", use_container_width=True)
        
        # ================= 第一步：按钮仅负责获取数据并存入 Session State =================
        if st.button("🚀 开始识别", type="primary", use_container_width=True):
            with st.spinner("调用后端 API 进行 CLAHE 增强与 YOLOv8 推理..."):
                img_bytes = uploaded_file.getvalue()
                result = call_predict_api(img_bytes, conf_thres)
                
                if result and result.get('status') == 'success':
                    # 将识别结果存入全局变量，防止刷新丢失
                    st.session_state.ocr_results = result
                    st.session_state.ai_display_content = None # 重新识别时清空之前的AI结论
                    
                    # 历史记录保存逻辑移到这里，确保每次成功识别只保存一次
                    preds = result.get('results', [])
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "num_chars": len(preds),
                        "chars": " ".join([p['char'] for p in preds]),
                        "confidences": [p['confidence'] for p in preds]
                    })
                else:
                    st.error("识别失败，请检查后端服务")

        # ================= 第二步：只要有数据，就独立渲染结果区（不缩进在按钮下） =================
        if st.session_state.ocr_results:
            res = st.session_state.ocr_results
            preds = res.get('results', [])
            
            # 显示增强图像
            if 'enhanced_image_base64' in res and res['enhanced_image_base64']:
                enhanced_img = Image.open(io.BytesIO(base64.b64decode(res['enhanced_image_base64'])))
                with col_mid:
                    st.image(enhanced_img, caption="CLAHE 增强后", use_container_width=True)
            else:
                with col_mid:
                    st.info("增强图像未返回")
            
            # 绘制识别结果
            annotated_img = original_img.copy()
            annotated_img = draw_boxes_on_image(annotated_img, preds)
            with col_right:
                st.image(annotated_img, caption="识别结果", use_container_width=True)
            
            # 置信度条形图
            if preds:
                st.subheader("识别字符置信度")
                df_pred = pd.DataFrame(preds)
                fig = px.bar(df_pred, x='char', y='confidence', color='confidence', 
                             title="各字符置信度", text='confidence')
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("查看详细识别结果"):
                    st.dataframe(df_pred[['char','confidence','bbox']])
            else:
                st.warning("未检测到字符")

            # --- AI 助手交互区 ---
            st.markdown("---")
            st.subheader("🤖 AI 智能释读助手")
            
            char_sequence = [p['char'] for p in preds]
            low_conf_indices = [i for i, p in enumerate(preds) if p['confidence'] < 0.6]

            # 按钮排列
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔍 智能纠错与重校准", use_container_width=True):
                    with st.spinner("专家正在会诊..."):
                        raw = call_ai_function(API_SMART_CORRECTION, {"char_sequence": char_sequence, "low_conf_indices": low_conf_indices})
                        if raw and "error" not in raw:
                            polished = call_ai_function(API_POLISH, {"raw_data": raw, "task_type": "correction"})
                            st.session_state.ai_display_content = polished.get("polished_text") if polished else f"⚠️ 润色无返回内容: {raw}"
                        else:
                            st.session_state.ai_display_content = f"❌ 纠错失败: {raw}"
                    st.rerun() 

            with c2:
                if st.button("🩹 残损字符智能补全", use_container_width=True):
                    with st.spinner("正在对比古籍语境..."):
                        raw = call_ai_function(API_INCOMPLETE_COMPLETION, {"char_sequence": char_sequence})
                        if raw and "error" not in raw:
                            polished = call_ai_function(API_POLISH, {"raw_data": raw, "task_type": "completion"})
                            st.session_state.ai_display_content = polished.get("polished_text") if polished else f"⚠️ 润色无返回内容: {raw}"
                        else:
                            st.session_state.ai_display_content = f"❌ 补全失败: {raw}"
                    st.rerun() 

            with c3:
                if st.button("📝 一键生成释读报告", use_container_width=True):
                    with st.spinner("撰写鉴定报告中..."):
                        raw = call_ai_function(API_GENERATE_REPORT, {"char_sequence": char_sequence, "detections": preds})
                        if raw and raw.get("report") and raw.get("report") != "生成报告失败":
                            st.session_state.ai_display_content = raw.get("report")
                        else:
                            st.session_state.ai_display_content = f"❌ 报告生成失败: {raw}"
                    st.rerun() 
                    
            # 渲染 AI 结果
            if st.session_state.ai_display_content:
                st.write("") 
                with st.container(border=True):
                    st.markdown("### 🏺 专家释读结论")
                    st.markdown("---")
                    st.markdown(str(st.session_state.ai_display_content))
                    st.write("") 
                    
                    if st.button("🗑️ 关闭结果显示", key="close_ai_result"):
                        st.session_state.ai_display_content = None
                        st.rerun() 

    # 历史记录展示
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📋 识别历史记录")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("导出历史记录为 CSV", csv, "recognition_history.csv", "text/csv")
        if st.button("清空历史"):
            st.session_state.history = []
            st.rerun()
