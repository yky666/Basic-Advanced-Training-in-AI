# Windows 语音识别 → 发送文字给小车
# 不用装 socket！自带的！
import speech_recognition as sr
import socket

# ====================== 重要：改成你小车的IP ======================
CAR_IP = "10.62.39.89"   # 你的小车IP
CAR_PORT = 10888          # 保持不变
# =================================================================

def send_text(text):
    try:
        s = socket.socket()
        s.connect((CAR_IP, CAR_PORT))
        s.send(text.encode("utf-8"))
        s.close()
        print(f"✅ 发送成功：{text}")
    except:
        print("❌ 发送失败，请检查小车IP")

def main():
    r = sr.Recognizer()
    mic = sr.Microphone()

    print("✅ Windows 语音控制已启动！")
    print("请说话，例如：往前走一米、左转、停止\n")

    with mic as source:
        r.adjust_for_ambient_noise(source)
        while True:
            try:
                print("🎤 聆听中...")
                audio = r.listen(source, phrase_time_limit=3)
                text = r.recognize_google(audio, language="zh-CN")
                print(f"🗣️ 识别：{text}")
                send_text(text)
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print("错误：", e)

if __name__ == "__main__":
    main()