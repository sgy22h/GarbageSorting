import os
os.system("chcp 65001")
import cv2
import numpy as np

def image_capture_preprocess(image_input):
    """
    接口名：image_capture_preprocess（完全按文档）
    功能：图像采集 + 预处理
    输入：支持两种类型
        1. str: 本地图片文件路径（兼容原逻辑）
        2. np.ndarray: 内存中的图像数组（兼容Streamlit摄像头拍照）
    输出：符合YOLOv8要求的标准图像（ndarray，BGR格式，640×640×3）
    """
    # 1. 读取/获取图像
    if isinstance(image_input, str):
        # 输入为文件路径：使用OpenCV读取（BGR格式，与训练数据完全一致）
        img = cv2.imread(image_input)
        if img is None:
            raise Exception("图像文件损坏或无法读取")
    elif isinstance(image_input, np.ndarray):
        # 输入为numpy数组：自动转换为BGR格式（兼容PIL/RGB摄像头数据）
        img = image_input
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        raise Exception("输入类型不支持，仅支持文件路径字符串或numpy数组")

    # 2. 格式检查
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise Exception("仅支持JPG/PNG格式彩色图片")

    # 3. 统一缩放到640×640（YOLO标准输入）
    target_size = 640
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # 4. 灰边填充到640×640（保持原始宽高比，避免拉伸变形）
    img_padded = np.ones((target_size, target_size, 3), dtype=np.uint8) * 114
    img_padded[:new_h, :new_w, :] = img_resized

    # 5. 高斯去噪（文档要求）
    img_denoised = cv2.GaussianBlur(img_padded, (3, 3), 0)

    return img_denoised

if __name__ == "__main__":
    # 测试1：原逻辑-本地文件路径
    test_img_path = image_capture_preprocess("test.jpg")
    print("本地文件预处理完成，标准图像尺寸：", test_img_path.shape)
    
    # 测试2：摄像头数组逻辑（模拟Streamlit拍照输入）
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        test_img_cam = image_capture_preprocess(frame)
        print("摄像头图像预处理完成，标准图像尺寸：", test_img_cam.shape)