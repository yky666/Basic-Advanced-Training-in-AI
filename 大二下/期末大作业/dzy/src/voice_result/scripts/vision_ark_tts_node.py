#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import os
import base64
import asyncio
from cv_bridge import CvBridge

# 导入火山引擎官方SDK
from volcenginesdkarkruntime import AsyncArk

# ===================== 配置区 =====================
ARK_API_KEY = "ark-ef377b91-01ce-4f42-804c-96b753d7608e-9e43b"
MODEL_ID = "doubao-seed-2-0-pro-260215"
# ==================================================

bridge = CvBridge()
latest_cv_frame = None
frame_received = False # 用于打印首次收到图像的提示

# 全局SDK客户端
ark_client = AsyncArk(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY
)

# 图像订阅回调：提取最新帧
def img_callback(msg):
    global latest_cv_frame, frame_received
    try:
        if not frame_received:
            rospy.loginfo("📸 成功接收到第一帧相机画面，准备开始识别！")
            frame_received = True
            
        raw_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = raw_frame.shape[:2]
        # 压缩尺寸，极大提升Base64编码和网络传输速度
        latest_cv_frame = cv2.resize(raw_frame, (int(w * 0.4), int(h * 0.4)))
    except Exception as e:
        rospy.logwarn(f"图像转换失败: {str(e)}")

# 异步识图核心函数 (使用 Base64 内存直传，彻底抛弃本地临时文件)
async def ark_vision_base64(cv_img):
    # 1. OpenCV 图像直接在内存中转为 JPG 格式的 Base64 编码
    _, buffer = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    try:
        # 2. 调用官方标准的 Chat Completions 接口 (多模态)
        resp = await ark_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "识别画面中人或者水果，人只有邓紫棋、刘亦菲、任贤齐、撒贝宁四种，水果只有苹果、香蕉、葡萄、橙子；如果是人，是谁？如果是水果，是什么水果？并且用中文描述整个画面，控制15个字以内"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ]
        )
        # 3. 提取AI返回的中文描述
        text_result = resp.choices[0].message.content.strip()
        return text_result
    except Exception as err:
        return f"识图异常：{str(err)}"

# HDMI语音播报
def voice_broadcast(text):
    # 过滤掉换行和引号，防止破坏 bash 命令
    clean_text = text.replace("\n", "").replace('"', "").replace("'", "")
    os.system(f'espeak-ng -v zh "{clean_text}"')

if __name__ == "__main__":
    rospy.init_node("doubao_sdk_vision_tts_node")
    
    # 订阅相机图像话题
    topic_name = "/camera/color/image_raw"
    rospy.Subscriber(topic_name, Image, img_callback, queue_size=1)
    result_pub = rospy.Publisher("/vision_result", String, queue_size=5)
    
    rospy.loginfo("===== 火山SDK Base64直传识图节点启动 =====")
    rospy.loginfo(f"⏳ 正在等待相机话题: {topic_name}")

    loop_rate = rospy.Rate(0.2)  # 5秒识别一次，防止刷爆API额度
    
    while not rospy.is_shutdown():
        if latest_cv_frame is not None:
            rospy.loginfo("🚀 开始调用豆包大模型识图...")
            
            # 锁定当前帧进行推理（防止在推理时被回调函数覆盖）
            frame_to_infer = latest_cv_frame.copy()
            latest_cv_frame = None # 清空缓存，等待下一轮
            
            # 异步调用SDK
            recognize_text = asyncio.run(ark_vision_base64(frame_to_infer))

            # 发布话题并播报
            rospy.loginfo(f"🎯 识别结果：{recognize_text}")
            result_pub.publish(recognize_text)
            voice_broadcast(recognize_text)
            
        loop_rate.sleep()