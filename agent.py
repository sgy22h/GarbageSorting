# ==============================
# 智能体调度核心模块
# 功能：任务路由 + 统一调用 + 异常处理 + 数据保存 + 自动预处理
# ==============================
from garbage_image_recognize import garbage_image_recognize
from garbage_classify_judge import garbage_classify_judge
from garbage_qa_llm import garbage_qa_llm
from preprocess import image_capture_preprocess  # 新增：导入预处理模块
import csv
import os
from datetime import datetime

# ======================
# 5. 数据记录保存接口（任务5）
# ======================
def data_record_save(record: dict) -> bool:
    """
    统一数据保存接口
    支持：识别记录 / 问答记录
    返回：True=成功 False=失败
    """
    try:
        # 自动创建文件夹
        os.makedirs("data", exist_ok=True)

        # 保存识别记录（新增suggestion字段）
        if "garbage_name" in record:
            path = "data/recognize_records.csv"
            # ✅ 修正：加上suggestion字段
            header = ["time", "garbage_name", "category", "confidence", "suggestion", "msg"]
            # 写入
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if not os.path.exists(path):
                    w.writeheader()
                w.writerow(record)

        # 保存问答记录
        elif "question" in record:
            path = "data/qa_records.csv"
            header = ["time", "question", "answer"]
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if not os.path.exists(path):
                    w.writeheader()
                w.writerow(record)

        return True
    except Exception as e:
        print(f"保存失败：{str(e)}")
        return False

# ======================
# 1. 智能体调度核心函数（任务1）
# 2. 任务类型自动路由（任务2）
# 3. 统一异常捕获 + 结果封装（任务3）
# ======================
def agent_receive_task(task_type: str, data):
    """
    总调度入口
    :param task_type: image / text
    :param data: 图像文件路径 或 文本问题
    :return: 统一格式结果
    """
    try:
        # ----------------------
        # 图像任务 → 自动预处理 + 识别 + 分类
        # ----------------------
        if task_type == "image":
            # ✅ 新增：自动执行图像预处理（强制嵌入调度流程，不会被遗忘）
            standard_image = image_capture_preprocess(data)
            
            # 调用图像识别（输入为预处理后的标准图像）
            rec_res = garbage_image_recognize(standard_image)
            
            if rec_res["name"] is None:
                return {
                    "code": -1,
                    "msg": rec_res["msg"]
                }

            # 调用分类判断
            judge_res = garbage_classify_judge(rec_res["name"])

            # 封装最终结果
            result = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "garbage_name": rec_res["name"],
                "category": judge_res["category"],
                "confidence": rec_res["confidence"],
                "suggestion": judge_res["suggestion"],
                "msg": rec_res["msg"]
            }

            # 自动保存数据
            data_record_save(result)

            return {
                "code": 0,
                **result
            }

        # ----------------------
        # 文本任务 → 智能问答
        # ----------------------
        elif task_type == "text":
            # 调用问答接口
            answer = garbage_qa_llm(data)

            if answer["code"] != 200:
                return {
                    "code": -1,
                    "msg": answer["msg"]
                }

            # 封装结果（只取纯文本答案）
            result = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": data,
                "answer": answer["answer"]
            }

            # 自动保存
            data_record_save(result)

            return {
                "code": 0,
                **result
            }

        # 不支持的任务
        else:
            return {
                "code": -1,
                "msg": "不支持的任务类型"
            }

    # 全局统一异常处理（任务3）
    except Exception as e:
        return {
            "code": -1,
            "msg": f"系统异常：{str(e)}"
        }

# ======================
# 4. 界面→智能体接口（任务4）
# ======================
def ui_to_agent(task_type: str, data):
    """界面层调用智能体的统一入口"""
    return agent_receive_task(task_type, data)


# ======================
# 测试代码（可直接运行）
# ======================
if __name__ == "__main__":
    # ✅ 测试图像接口
    print(ui_to_agent("image", "test1.jpg"))

    # 测试问答接口
    print(ui_to_agent("text", "电池是什么垃圾？"))

  
