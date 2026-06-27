#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import re
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from text_preprocessor import TextPreprocessor

try:
    import rospy
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
from std_msgs.msg import String

class ParamParser:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.relative = {'一点':0.3, '一点点':0.2, '稍微':0.3, '很多':1.5, '非常':1.5}
        self.intent_map = {
            'forward':['distance'], 'backward':['distance'],
            'left':['angle'], 'right':['angle'], 'turn_around':['angle'],
            'stop':[], 'wait':['time'], 'speed_up':[], 'slow_down':[]
        }
        self.last_text = None
        self.last_intent = None
        self.cmd_queue = []

        if HAS_ROS:
            rospy.Subscriber('/text_command', String, self.text_cb)
            rospy.Subscriber('/intent_result', String, self.intent_cb)
            self.cmd_pub = rospy.Publisher('/cmd_params', String, queue_size=10)
            rospy.loginfo("✅ 支持复合指令·参数解析器启动")

    def text_cb(self, msg):
        self.last_text = msg.data
        rospy.loginfo(f"📥 文本: {self.last_text}")
        self.try_parse()

    def intent_cb(self, msg):
        self.last_intent = msg.data
        rospy.loginfo(f"🔍 意图: {self.last_intent}")
        self.try_parse()

    def try_parse(self):
        if not self.last_text or not self.last_intent:
            return
        text = self.last_text
        intent = self.last_intent

        # 复合指令切分
        parts = re.split('然后|接着|再|之后', text)
        for p in parts:
            p = p.strip()
            if not p: continue
            self._parse_single(p)

        # 执行指令队列
        self._run_queue()
        self.last_text = None
        self.last_intent = None

    def _parse_single(self, text):
        text = self.preprocessor.normalize_numbers(text)
        rospy.loginfo(f"✅ 归一化: {text}")
        scale = 1.0
        for k, v in self.relative.items():
            if k in text: scale = v

        nums = re.findall(r'\d+\.?\d*', text)
        val = float(nums[0])*scale if nums else None

        # 判断意图
        if '前进' in text or '往前走' in text: intent = 'forward'
        elif '后退' in text or '往后' in text: intent = 'backward'
        elif '左转' in text: intent = 'left'
        elif '右转' in text: intent = 'right'
        elif '掉头' in text: intent = 'turn_around'
        elif '加速' in text: intent = 'speed_up'
        elif '减速' in text: intent = 'slow_down'
        elif '停' in text: intent = 'stop'
        elif '等待' in text or '延时' in text: intent = 'wait'
        else: intent = 'stop'

        # 拼接指令
        if intent in ['forward','backward']:
            cmd = f"{intent},distance={val}" if val else intent
        elif intent in ['left','right','turn_around']:
            cmd = f"{intent},angle={val}" if val else intent
        elif intent == 'wait':
            cmd = f"wait,time={val}" if val else 'wait'
        else:
            cmd = intent
        self.cmd_queue.append(cmd)

    def _run_queue(self):
        for cmd in self.cmd_queue:
            rospy.loginfo(f"🚀 队列指令: {cmd}")
            self.cmd_pub.publish(cmd)
            rospy.sleep(0.5)
        self.cmd_queue = []

def main():
    if HAS_ROS:
        rospy.init_node('param_parser_node')
        ParamParser()
        rospy.spin()

if __name__ == '__main__':
    main()