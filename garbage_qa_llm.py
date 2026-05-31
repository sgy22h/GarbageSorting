import requests
import json
import streamlit as st

DASHSCOPE_API_KEY = st.secrets.get("LLM_API_KEY", "")

def call_real_llm(prompt, timeout=8):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "input": {
            "messages": [{"role": "user", "content": prompt}]
        },
        "parameters": {
            "temperature": 0.0,
            "result_format": "text"
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["output"]["text"].strip()
    except Exception as e:
        raise Exception(f"大模型调用失败：{str(e)}")

def garbage_qa_llm(user_question: str, timeout: int = 8):
    try:
        prompt = f"""你是专业的垃圾分类助手，只回答垃圾分类相关问题，简洁、准确、正式回答。
问题：{user_question}"""

        answer = call_real_llm(prompt, timeout=timeout)

        if not answer:
            return {
                "code": 201,
                "msg": "暂无答案",
                "answer": "抱歉，我无法回答该问题。"
            }

        return {
            "code": 200,
            "msg": "成功",
            "answer": answer
        }

    except Exception as e:
        return {
            "code": 500,
            "msg": f"服务异常：{str(e)}",
            "answer": "服务暂时不可用"
        }