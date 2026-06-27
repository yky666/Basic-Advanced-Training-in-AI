import os

def tts_speak(text):
    # 直接调用espeak-ng，自动使用HDMI默认输出
    os.system(f'espeak-ng -v zh "{text}"')

if __name__ == "__main__":
    tts_speak("我是中山大学智工小趴虎")