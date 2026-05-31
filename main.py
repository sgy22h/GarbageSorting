import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import numpy as np
from agent import ui_to_agent

# ====================== 初始化 ======================
# 注意：不要使用 st.session_state.clear()，只定义业务变量
if "recognize_result" not in st.session_state:
    st.session_state.recognize_result = None
if "qa_result" not in st.session_state:
    st.session_state.qa_result = ""
if "img_file_path" not in st.session_state:
    st.session_state.img_file_path = ""
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0   # 用于重置摄像头组件的 key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # 用于重置上传组件

os.makedirs("upload_imgs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ====================== 数据保存（可选，如果 agent 已经自动保存可删除） ======================
def save_recognize_record(record):
    csv_path = os.path.join("data", "recognize_history.csv")
    df = pd.DataFrame([record])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8")

def save_qa_record(record):
    csv_path = os.path.join("data", "qa_log.csv")
    df = pd.DataFrame([record])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8")

# ====================== 核心识别处理 ======================
def agent_task(input_type, data):
    try:
        if input_type == "image":
            result = ui_to_agent("image", data)
            if result.get("garbage_name") is None:
                return {"code": -1, "msg": "未识别到垃圾，请重新拍摄"}
            if result["code"] != 0:
                return {"code": -1, "msg": result["msg"]}

            # 如果 agent 已自动保存，此处可注释掉保存代码
            record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "garbage_name": result["garbage_name"],
                "confidence": result["confidence"],
                "category": result["category"]
            }
            save_recognize_record(record)

            return {
                "code": 0,
                "garbage_name": result["garbage_name"],
                "confidence": result["confidence"],
                "category": result["category"],
                "save_status": "保存成功"
            }

        elif input_type == "text":
            result = ui_to_agent("text", data)
            if result["code"] != 0:
                return {"code": -1, "msg": result["msg"]}
            answer = result["answer"]
            record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": data,
                "answer": answer
            }
            save_qa_record(record)
            return {"code": 0, "answer": answer, "save_status": "保存成功"}

        else:
            return {"code": -1, "msg": "不支持的任务类型"}
    except Exception as e:
        return {"code": -1, "msg": f"系统异常: {str(e)}"}

# ====================== 辅助函数 ======================
def translate_name(name):
    name_map = {
        "plastic": "塑料瓶", "plastic_bottle": "塑料瓶", "can": "易拉罐",
        "food_waste": "厨余垃圾", "battery": "电池", "paper": "纸巾",
        "carton": "纸箱", "cup": "一次性杯子", "lid": "瓶盖",
        "straw": "吸管", "glass bottle": "玻璃瓶"
    }
    return name_map.get(name.lower(), name)

def get_garbage_detail(garbage_type):
    details = {
        "可回收物": "**♻️ 可回收物**\n🔹 投放：蓝色桶\n🔹 包含：易拉罐、塑料瓶、纸箱、玻璃瓶等",
        "厨余垃圾": "**🍚 厨余垃圾**\n🔹 投放：绿色桶\n🔹 包含：果皮、剩菜、食物残渣等",
        "有害垃圾": "**☠️ 有害垃圾**\n🔹 投放：红色桶\n🔹 包含：电池、过期药品、灯管等",
        "其他垃圾": "**🗑️ 其他垃圾**\n🔹 投放：灰色桶\n🔹 包含：纸巾、烟头、一次性餐具等"
    }
    return details.get(str(garbage_type), "暂无信息")

# ====================== 页面布局 ======================
st.set_page_config(page_title="垃圾分类小助手", layout="wide")
st.markdown("""
<style>
.gradient-title { font-size: 24px; font-weight: 800; background: linear-gradient(90deg, #22c55e, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="gradient-title">♻️ 智能垃圾分类小助手</p>', unsafe_allow_html=True)
st.caption("📸 上传/拍照 → 智能识别 → 分类知识 → 问答")
st.divider()

col_left, col_right = st.columns([2.5, 7.5])

# ====================== 左侧控制面板 ======================
with col_left:
    st.markdown("### 🛡️ 控制面板")
    st.divider()

    st.markdown("#### 1️⃣ 照片输入")
    # 使用 radio 切换输入方式，且利用 key 强制重建组件
    mode = st.radio("输入方式", ["本地上传", "摄像头拍摄"], horizontal=True, key="input_mode")

    if mode == "本地上传":
        uploaded_file = st.file_uploader(
            "JPG/PNG", type=["jpg", "png"],
            key=f"uploader_{st.session_state.uploader_key}"  # 动态 key 可重置
        )
        if uploaded_file:
            # 保存图片到本地并记录路径
            fn = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
            fp = os.path.join("upload_imgs", fn)
            with open(fp, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.img_file_path = fp
            st.success("✅ 上传成功")
        else:
            st.session_state.img_file_path = ""
    else:  # 摄像头拍摄
        camera_photo = st.camera_input(
            "拍摄",
            key=f"camera_{st.session_state.camera_key}"  # 动态 key
        )
        if camera_photo:
            fn = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_cam.jpg"
            fp = os.path.join("upload_imgs", fn)
            with open(fp, "wb") as f:
                f.write(camera_photo.getbuffer())
            st.session_state.img_file_path = fp
            st.success("✅ 拍摄成功")
        else:
            st.session_state.img_file_path = ""

    st.divider()
    st.markdown("#### 2️⃣ 开始识别")
    btn_rec = st.button("🔍 开始识别", type="primary", use_container_width=True,
                        disabled=not st.session_state.img_file_path)
    btn_clear = st.button("🧹 清空", use_container_width=True)

    st.divider()
    st.markdown("#### 3️⃣ 在线问答")
    q = st.text_area("输入问题", placeholder="例：电池是什么垃圾？", key="qa_input")
    btn_q = st.button("💬 提交问题", use_container_width=True, disabled=not q.strip())

# ====================== 清空逻辑（无 rerun 冲突） ======================
if btn_clear:
    # 重置所有业务状态
    st.session_state.recognize_result = None
    st.session_state.qa_result = ""
    st.session_state.img_file_path = ""
    # 更新 key 强制重建组件（清除残留的 DOM）
    st.session_state.camera_key += 1
    st.session_state.uploader_key += 1
    # 不要调用 st.rerun()，让自然流程重新运行脚本
    # 由于按钮触发导致脚本重新执行，组件会因 key 变化而完全重建，不会报错

# ====================== 右侧展示区 ======================
with col_right:
    st.markdown("### 📋 识别展示区")
    st.divider()

    st.markdown("#### 🖼️ 图片预览")
    with st.container(border=True):
        if st.session_state.img_file_path and os.path.exists(st.session_state.img_file_path):
            img = Image.open(st.session_state.img_file_path)
            st.image(img, width=350)
        else:
            st.info("请上传或拍摄图片")

    st.divider()
    st.markdown("#### 🎯 识别结果")
    with st.container(border=True):
        if btn_rec and st.session_state.img_file_path:
            with st.spinner("识别中..."):
                res = agent_task("image", st.session_state.img_file_path)
            if res["code"] == 0:
                st.session_state.recognize_result = res
                show_name = translate_name(res['garbage_name'])
                st.success("识别完成")
                st.write(f"物品名称：{show_name}")
                st.write(f"分类：{res['category']}")
                st.write(f"置信度：{res['confidence']:.2%}")
                st.write(f"保存状态：{res['save_status']}")
            else:
                st.error(res["msg"])
        elif st.session_state.recognize_result:
            res = st.session_state.recognize_result
            show_name = translate_name(res['garbage_name'])
            st.success(f"""
**历史结果**
物品名称：{show_name}
分类：{res['category']}
置信度：{res['confidence']:.2%}
""")
        else:
            st.info("点击识别按钮获取结果")

    st.divider()
    st.markdown("#### 📖 分类小知识")
    with st.container(border=True):
        if st.session_state.recognize_result:
            st.markdown(get_garbage_detail(st.session_state.recognize_result["category"]))
        else:
            st.info("识别后显示")

    st.divider()
    st.markdown("#### 💬 问答解答")
    with st.container(border=True):
        if btn_q and q.strip():
            with st.spinner("思考中..."):
                res = agent_task("text", q)
            if res["code"] == 0:
                st.markdown(f"🔎 **问题**：{q}\n\n✅ **回答**：{res['answer']}\n\n💾 保存状态：{res['save_status']}")
            else:
                st.error(res["msg"])
        else:
            st.info("输入问题获取解答")