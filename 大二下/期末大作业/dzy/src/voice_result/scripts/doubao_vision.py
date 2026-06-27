import requests
import base64
import cv2

# ===================== 配置区（复制你自己的参数）=====================
ARK_API_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
ARK_AUTH_TOKEN = "ark-ef377b91-01ce-4f42-804c-96b753d7608e-9e43b"
MODEL_ID = "doubao-seed-2-0-pro-260215"
# ==================================================================

def image_to_base64(img_path=None, cv_frame=None):
    """
    两种传图方式：1.本地图片路径 2. OpenCV内存帧（摄像头实时画面）
    返回base64编码字符串
    """
    if cv_frame is not None:
        # 内存帧转base64（ROS摄像头图像专用）
        ret, buf = cv2.imencode(".jpg", cv_frame)
        img_bytes = buf.tobytes()
    else:
        # 本地文件转base64
        with open(img_path, "rb") as f:
            img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")

def detect_image_content(base64_img, prompt="详细描述图片里的物体、场景，用中文输出"):
    """调用豆包多模态识图，返回文字结果"""
    headers = {
        "Authorization": f"Bearer {ARK_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_base64": base64_img
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    try:
        resp = requests.post(ARK_API_URL, json=payload, headers=headers, timeout=10)
        resp_data = resp.json()
        # 提取AI返回的中文描述
        result_text = resp_data["output"]["choices"][0]["message"]["content"][0]["text"]
        return result_text
    except Exception as e:
        return f"API调用失败：{str(e)}"

# 本地图片测试入口
if __name__ == "__main__":
    # 替换成你的测试图片路径
    img_base64 = image_to_base64(img_path="./test.jpg")
    res = detect_image_content(img_base64)
    print("豆包识图结果：\n", res)