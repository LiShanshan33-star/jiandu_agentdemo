# ai_assistant.py
import os
import re
import json
import requests

# DeepSeek API 配置（建议从环境变量读取）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-84ee1e1197fe45b6ab28a2245052be58")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def call_deepseek(messages, temperature=0.7):
    """通用 DeepSeek API 调用函数"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"DeepSeek API 请求失败: {e}")
        return None

def smart_correction(char_sequence, low_conf_indices):
    """智能纠错与置信度重校准"""
    prompt = f"""你是一位简牍文字释读专家。下面是一个简牍文字识别序列，其中包含一些低置信度的字符（用□代替）。
请根据上下文语义，推断出□处最可能的汉字，并给出修正后的完整句子和理由。

识别序列：{" ".join(char_sequence)}
低置信度字符位置（索引从0开始）：{low_conf_indices}

请严格按以下JSON格式输出（不要包含其他解释）：
{{
    "corrected_sentence": "修正后的完整句子",
    "corrections": [
        {{"position": 位置索引, "original": "原识别字符或□", "corrected": "修正后的字符", "confidence": 0.85, "reason": "修正理由"}}
    ]
}}
"""
    messages = [{"role": "user", "content": prompt}]
    result = call_deepseek(messages, temperature=0.5)
    if result:
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"error": "解析AI返回的JSON失败", "raw_response": result}
    return {"error": "AI服务调用失败"}

def incomplete_completion(char_sequence):
    """残损字符智能补全与上下文验证"""
    prompt = f"""你是一位简牍文字释读专家。下面是一个从简牍图像中识别出的不完整字符序列（包含缺失部分，用□代替）。
请根据你的历史知识和上下文，补全这些缺失的字符，并评估整个句子的通顺度和合理性。

字符序列：{" ".join(char_sequence)}

请严格按以下JSON格式输出：
{{
    "completed_sentence": "补全缺失字符后的完整句子",
    "missing_chars": [
        {{"position": 位置索引, "possible_chars": ["候选字1", "候选字2"], "confidence": "高/中/低"}}
    ],
    "sentence_evaluation": "对补全后句子的通顺度和合理性的简要评估。"
}}
"""
    messages = [{"role": "user", "content": prompt}]
    result = call_deepseek(messages, temperature=0.6)
    if result:
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"error": "解析AI返回的JSON失败", "raw_response": result}
    return {"error": "AI服务调用失败"}

def generate_report(char_sequence, detections):
    """自动生成释读报告"""
    # 构建检测结果的文本描述
    detections_text = ""
    for det in detections:
        detections_text += f"- 字符：{det['char']}，置信度：{det['confidence']:.2f}，位置：{det['bbox']}\n"
    prompt = f"""你是一位简牍文字释读专家，现在需要根据以下从简牍图像中识别出的字符序列和检测详情，生成一份结构化的释读报告。

字符序列：{" ".join(char_sequence)}

检测详情：
{detections_text}

报告必须包含以下部分（使用Markdown格式）：
1. **释读概览**：整体描述简牍的内容主题。
2. **逐字释读明细**：以表格形式列出每个字符、其置信度和简要说明（如是否为残字、异体字等）。
3. **释读难点与存疑**：指出识别过程中置信度较低或语义不通的字符，并提供可能的原因分析。
4. **简要总结**：总结该简牍的主要信息和历史价值。

请确保报告语言专业、结构清晰。
"""
    messages = [{"role": "user", "content": prompt}]
    report = call_deepseek(messages, temperature=0.7)
    return {"report": report if report else "生成报告失败"}

# ai_assistant.py (局部修正)

def polish_result(raw_data, task_type):
    """调用 DeepSeek 将 JSON 数据转化为自然语言报告"""
    prompts = {
        "correction": "你是一位文字修复专家。请根据以下纠错 JSON 数据，写一段口语化、专业且易懂的修复说明。要求：先给出修正后的句子，再简述理由。",
        "completion": "你是一位古籍补全专家。请根据以下残损补全 JSON 数据，描述你如何通过上下文‘复原’了这些文字，并评价句子的完整性。",
        "report": "你是一位古籍研究员，请将以下结构化的释读报告进行语言润色，使其显得更正式、更有学术深度。" # 补全这一行
    }
    
    # 增加兜底逻辑，防止 get 找不到 key
    base_prompt = prompts.get(task_type, "请解读以下数据：")
    prompt = f"{base_prompt}\n数据内容：{json.dumps(raw_data, ensure_ascii=False)}"
    
    messages = [{"role": "user", "content": prompt}]
    return call_deepseek(messages, temperature=0.7)