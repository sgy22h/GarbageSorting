def garbage_classify_judge(garbage_name: str) -> dict:
    """
    根据识别出的垃圾名称，判断其所属的标准垃圾分类。
    
    Args:
        garbage_name: 图像识别模块输出的垃圾类别名称（如 "Can", "Cup"）
        
    Returns:
        dict: 包含分类结果、投放建议的字典
    """
    # 1. 分类映射表（根据你的YOLO类别来定义）
    category_map = {
         # 可回收物
        "Aluminium foil": "可回收物",
        "Bottle": "可回收物",
        "Broken glass": "可回收物",
        "Can": "可回收物",
        "Carton": "可回收物",
        "Lid": "可回收物",
        "Other plastic": "可回收物",
        "Paper": "可回收物",
        "Plastic container": "可回收物",
        "Pop tab": "可回收物",
        
        # 其他垃圾
        "Bottle cap": "其他垃圾",
        "Cigarette": "其他垃圾",
        "Cup": "其他垃圾",
        "Other litter": "其他垃圾",
        "Plastic bag - wrapper": "其他垃圾",
        "Straw": "其他垃圾",
        "Styrofoam piece": "其他垃圾",
        "Unlabeled litter": "其他垃圾",
        
        # 厨余垃圾
        "banana_peel": "厨余垃圾",
        "food_waste": "厨余垃圾",
        
        # 有害垃圾
        "battery": "有害垃圾"
    }

    # 2. 投放建议（按四大分类）
    suggestion_map = {
        "可回收物": "请投放至蓝色可回收物收集容器，注意保持清洁干燥。",
        "厨余垃圾": "请投放至绿色厨余垃圾收集容器，沥干水分后投放。",
        "有害垃圾": "请投放至红色有害垃圾收集容器，注意密封，避免泄漏。",
        "其他垃圾": "请投放至灰色其他垃圾收集容器。"
    }

    # 3. 判断分类
    category = category_map.get(garbage_name, "其他垃圾")
    suggestion = suggestion_map.get(category, "请按当地垃圾分类指引投放。")

    return {
        "code": 200,
        "garbage_name": garbage_name,
        "category": category,
        "suggestion": suggestion
    }