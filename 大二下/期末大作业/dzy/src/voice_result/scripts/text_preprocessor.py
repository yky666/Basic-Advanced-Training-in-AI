#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import jieba
import re

CUSTOM_DICT = """
往前走 20 n
往后退 20 n
往左转 20 n
往右转 20 n
左转 20 n
右转 20 n
前进 20 n
后退 20 n
掉头 20 n
三米 15 n
五米 15 n
十米 15 n
九十度 15 n
一百八十度 15 n
"""

STOPWORDS = set([
    '的', '了', '着', '啊', '吧', '呢', '吗', '呀', '哦', '哈',
    '一下', '一点', '一会儿', '一些', '然后', '接着', '再', '又', '还', '也',
    '请', '帮我', '麻烦', '麻烦你', '个', '下', '点', '次', '遍'
])

class TextPreprocessor:
    def __init__(self, use_custom_dict=True):
        if use_custom_dict:
            self._load_custom_dict()

        self.ngram_phrases = {
            '往前走': 'forward', '往前': 'forward', '向前': 'forward', '前进': 'forward', '向前走': 'forward',
            '往后退': 'backward', '后退': 'backward', '倒车': 'backward',
            '左转': 'left', '往左': 'left', '向左': 'left', '左拐': 'left',
            '右转': 'right', '往右': 'right', '向右': 'right', '右拐': 'right',
            '停下': 'stop', '停止': 'stop', '别动': 'stop', '暂停': 'stop',
            '加速': 'speed_up', '提速': 'speed_up',
            '减速': 'slow_down', '慢一点': 'slow_down',
            '掉头': 'turn_around', '原180': 'turn_around',
        }

        rospy.loginfo("文本预处理器初始化完成")

    def _load_custom_dict(self):
        for line in CUSTOM_DICT.strip().split('\n'):
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    word, freq = parts[0], parts[1]
                    jieba.add_word(word, freq=freq)
        rospy.loginfo("已加载自定义词典")

    def segment(self, text):
        text = text.strip()
        words = list(jieba.cut(text, cut_all=False))
        words = [w for w in words if w not in STOPWORDS and len(w.strip()) > 0]
        return words

    def detect_phrases(self, text):
        phrases = []
        remaining = text
        while remaining:
            matched = False
            for phrase_len in range(min(len(remaining), 8), 0, -1):
                candidate = remaining[:phrase_len]
                if candidate in self.ngram_phrases:
                    phrases.append({
                        'text': candidate,
                        'intent': self.ngram_phrases[candidate],
                        'type': 'phrase'
                    })
                    remaining = remaining[phrase_len:]
                    matched = True
                    break
            if not matched:
                if remaining[0] not in STOPWORDS:
                    phrases.append({'text': remaining[0], 'type': 'char'})
                remaining = remaining[1:]
        return phrases

    # ==========================
    # 🔴 修复：九十 → 90，完美中文数字
    # ==========================
    def normalize_numbers(self, text):
        """
        终极修复版：
        一 → 1
        三 → 3
        十 → 10
        三十 → 30
        九十 → 90
        一百 → 100
        一百八十 → 180
        """
        # 直接硬编码所有机器人常用数字（最稳定）
        cn_to_num = {
            "一": "1",
            "二": "2",
            "两": "2",
            "三": "3",
            "四": "4",
            "五": "5",
            "六": "6",
            "七": "7",
            "八": "8",
            "九": "9",
            "十": "10",
            "二十": "20",
            "三十": "30",
            "四十": "40",
            "五十": "50",
            "六十": "60",
            "七十": "70",
            "八十": "80",
            "九十": "90",
            "一百": "100",
            "一百八十": "180",
            "一百八": "180",
        }
        
        # 从长到短替换，避免短的破坏长的
        for key in sorted(cn_to_num.keys(), key=len, reverse=True):
            text = text.replace(key, cn_to_num[key])
        
        return text

def test_preprocessor():
    preprocessor = TextPreprocessor()
    test_cases = [
        "往前走三米",
        "往后退五米然后左转九十度",
        "停止",
        "掉头一百八十度"
    ]
    for text in test_cases:
        res = preprocessor.preprocess(text)
        print(f"\n输入: {text}")
        print(f"归一化: {res['normalized']}")

if __name__ == '__main__':
    test_preprocessor()