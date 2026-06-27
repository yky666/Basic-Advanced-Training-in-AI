#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
意图分类节点 - 三层融合版本
功能：融合规则匹配 + TF-IDF + Word2Vec 进行意图分类
参考《2.2》第九章：当前主流文本表示方法

三层融合权重（可调）：
    - 规则匹配: 0.3
    - TF-IDF分类: 0.3
    - Word2Vec语义: 0.4
"""

import rospy
import numpy as np
from std_msgs.msg import String, Float32
import os
import pickle
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from text_preprocessor import TextPreprocessor

class ThreeLayerIntentClassifier:
    """三层融合意图分类器"""
    
    def __init__(self):
        """初始化三层分类器"""

        rospy.init_node('intent_classifier_node', anonymous=True)
        # 初始化文本预处理器
        self.preprocessor = TextPreprocessor()
        
        # 意图标签定义
        self.intents = ['forward', 'backward', 'left', 'right', 'stop', 'speed_up', 'slow_down', 'turn_around']
        
        # ============ 第一层：规则匹配 ============
        self.rule_intents = {
            '往前走': 'forward', '往前': 'forward', '前进': 'forward', '向前': 'forward',
            '往后退': 'backward', '后退': 'backward', '倒车': 'backward',
            '左转': 'left', '往左': 'left', '向左': 'left', '左拐': 'left',
            '右转': 'right', '往右': 'right', '向右': 'right', '右拐': 'right',
            '停止': 'stop', '停下': 'stop', '别动': 'stop', '暂停': 'stop',
            '加速': 'speed_up', '提速': 'speed_up',
            '减速': 'slow_down', '慢一点': 'slow_down',
            '掉头': 'turn_around', '原180': 'turn_around'
        }
        
        # ============ 第二层：TF-IDF + SVC ============
        self.tfidf_vectorizer = None
        self.tfidf_classifier = None
        self._init_tfidf_classifier()
        
        # ============ 第三层：Word2Vec ============
        self.word2vec_model = None
        self._init_word2vec_model()
        
        # ============ 融合权重 ============
        self.fusion_weights = {
            'rule': 0.3,
            'tfidf': 0.3,
            'word2vec': 0.4
        }
        # 创建订阅者和发布者
        rospy.Subscriber('/text_command', String, self.command_callback)
        self.intent_pub = rospy.Publisher('/intent_result', String, queue_size=10)
        self.confidence_pub = rospy.Publisher('/intent_confidence', Float32, queue_size=10)
        
        rospy.loginfo("三层融合意图分类器已启动")
    
    def _init_tfidf_classifier(self):
        """初始化TF-IDF分类器（参考《2.2》6.2节 TF-IDF）"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import SVC
            
            # 加载预训练模型（如果存在）
            model_path = os.path.join(os.path.dirname(__file__), '../models/tfidf_model.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.tfidf_vectorizer = data['vectorizer']
                    self.tfidf_classifier = data['classifier']
                rospy.loginfo("已加载TF-IDF预训练模型")
            else:
                # 使用基础训练数据初始化
                self._train_tfidf_classifier()
                
        except ImportError as e:
            rospy.logwarn(f"sklearn未安装，TF-IDF分类器不可用: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_classifier = None
    
    def _train_tfidf_classifier(self):
        """训练TF-IDF分类器"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import SVC
        # 训练数据（54个文本 = 54个标签）
        train_texts = [
            # forward - 9个
            "往前走", "往前", "前进", "向前", "往前走一点",
            "往前挪", "向前走", "向前进", "往前开",
            
            # backward - 7个
            "往后退", "后退", "倒车", "往后", "往后退一点",
            "后退走", "倒车一点",
            
            # left - 7个
            "左转", "往左", "向左", "左拐", "左转一下",
            "往左转", "左转弯",
            
            # right - 7个
            "右转", "往右", "向右", "右拐", "右转一下",
            "往右转", "右转弯",
            
            # stop - 6个
            "停止", "停下", "别动", "暂停", "停止运动", "站住",
            
            # speed_up - 6个
            "加速", "提速", "快一点", "快点", "加快", "加速前进",
            
            # slow_down - 5个
            "减速", "慢一点", "慢速", "减速运动", "慢下来",
            
            # turn_around - 7个
            "掉头", "原180", "掉个头", "转180", "回头", "转过来", "掉头转向",
        ]
        
        train_labels = ['forward']*9 + ['backward']*7 + ['left']*7 + ['right']*7 +['stop']*6 + ['speed_up']*6 + ['slow_down']*5 + ['turn_around']*7
        # 9+7+7+7+6+6+5+7 = 54 个 ✅
        
        # TF-IDF向量化
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=500
        )
        X_train = self.tfidf_vectorizer.fit_transform(train_texts)
        
        # 训练SVC分类器
        self.tfidf_classifier = SVC(probability=True)
        self.tfidf_classifier.fit(X_train, train_labels)
        
        rospy.loginfo("TF-IDF分类器训练完成")
    
    def _init_word2vec_model(self):
        """初始化Word2Vec模型（参考《2.2》8.1节 Word2Vec）"""
        try:
            from gensim.models import Word2Vec
            
            # 加载预训练模型（如果存在）
            model_path = os.path.join(os.path.dirname(__file__), '../models/word2vec_model.bin')
            
            if os.path.exists(model_path):
                self.word2vec_model = Word2Vec.load(model_path)
                rospy.loginfo("已加载Word2Vec预训练模型")
            else:
                # 训练基础模型
                self._train_word2vec_model()
                
        except ImportError as e:
            rospy.logwarn(f"gensim未安装，Word2Vec不可用: {e}")
            self.word2vec_model = None
    
    def _train_word2vec_model(self):
        """训练Word2Vec模型"""
        from gensim.models import Word2Vec
        
        # 训练语料（机器人控制领域的句子）
        sentences = [
            # forward相关
            ["往前走", "往前", "前进", "向前", "向前走"],
            ["往前走一点", "往前挪", "往前开"],
            # backward相关
            ["往后退", "后退", "倒车", "往后"],
            ["后退一点", "倒车一点"],
            # left相关
            ["左转", "往左", "向左", "左拐"],
            ["左转一下", "往左转", "左转弯"],
            # right相关
            ["右转", "往右", "向右", "右拐"],
            ["右转一下", "往右转", "右转弯"],
            # 组合指令
            ["往前走", "左转", "右转", "停止"],
            ["前进", "后退", "掉头"]
        ]
        
        # 训练Word2Vec模型（参考《2.2》8.1节）
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=100,  # 100维词向量
            window=5,         # 窗口大小5
            min_count=1,      # 最小词频1
            workers=4,        # 4个线程
            epochs=100        # 训练轮次
        )
        
        rospy.loginfo("Word2Vec模型训练完成")
    
    def _rule_classify(self, text):
        """
        第一层：规则匹配分类
        
        参考《2.2》6.1节 规则基础方法
        """
        # 预处理：归一化
        normalized = self.preprocessor.normalize_numbers(text)
        
        # 短语检测
        phrases = self.preprocessor.detect_phrases(normalized)
        
        # 查找匹配的意图
        for phrase in phrases:
            if phrase['type'] == 'phrase' and phrase['text'] in self.rule_intents:
                return self.rule_intents[phrase['text']], 0.9
        
        return None, 0.0
    
    def _tfidf_classify(self, text):
        """
        第二层：TF-IDF分类
        
        参考《2.2》6.2节 TF-IDF + 文本分类
        """
        if self.tfidf_vectorizer is None or self.tfidf_classifier is None:
            return None, 0.0
        
        try:
            # 预处理
            normalized = self.preprocessor.normalize_numbers(text)
            words = self.preprocessor.segment(normalized)
            seg_text = ' '.join(words)
            
            # TF-IDF向量化
            X = self.tfidf_vectorizer.transform([seg_text])
            
            # 预测
            intent = self.tfidf_classifier.predict(X)[0]
            prob = self.tfidf_classifier.predict_proba(X)[0]
            confidence = max(prob)  # 取最大概率
            
            return intent, confidence
            
        except Exception as e:
            rospy.logwarn(f"TF-IDF分类失败: {e}")
            return None, 0.0
    
    def _word2vec_classify(self, text):
        """
        第三层：Word2Vec语义分类
        
        参考《2.2》8.1节 Word2Vec 语义相似度
        """
        if self.word2vec_model is None:
            return None, 0.0
        
        try:
            # 预处理
            normalized = self.preprocessor.normalize_numbers(text)
            words = self.preprocessor.segment(normalized)
            # 计算文本的词向量（平均）
            vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    vectors.append(self.word2vec_model.wv[word])
            
            if not vectors:
                return None, 0.0
            
            # 平均词向量作为文本表示
            text_vector = np.mean(vectors, axis=0)
            
            # 计算与各意图标签的相似度
            best_intent = None
            best_score = 0.0
            
            for intent in self.intents:
                if intent in self.word2vec_model.wv:
                    # 余弦相似度（参考《2.2》向量空间模型）
                    similarity = np.dot(text_vector, self.word2vec_model.wv[intent])
                    similarity = similarity / (np.linalg.norm(text_vector) * np.linalg.norm(self.word2vec_model.wv[intent]))
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = intent
            
            # 归一化相似度到[0,1]
            confidence = (best_score + 1) / 2
            
            return best_intent, confidence
            
        except Exception as e:
            rospy.logwarn(f"Word2Vec分类失败: {e}")
            return None, 0.0
    
    def classify(self, text):
        """
        三层融合分类
        
        参考《2.2》第九章 多方法融合策略
        返回:
            (intent, confidence, details): 意图、置信度、详细信息
        """
        # 第一层：规则匹配
        rule_intent, rule_conf = self._rule_classify(text)
        
        # 第二层：TF-IDF
        tfidf_intent, tfidf_conf = self._tfidf_classify(text)
        
        # 第三层：Word2Vec
        w2v_intent, w2v_conf = self._word2vec_classify(text)
        
        # ============ 融合决策 ============
        # 统计各意图的加权得分
        intent_scores = {}
        for intent in self.intents:
            score = 0.0
            count = 0
            
            if rule_intent == intent:
                score += self.fusion_weights['rule'] * rule_conf
                count += 1
            if tfidf_intent == intent:
                score += self.fusion_weights['tfidf'] * tfidf_conf
                count += 1
            if w2v_intent == intent:
                score += self.fusion_weights['word2vec'] * w2v_conf
                count += 1
            
            if count > 0:
                intent_scores[intent] = score / count  # 平均得分
        
        # 选择最高分意图
        if intent_scores:
            final_intent = max(intent_scores, key=intent_scores.get)
            final_confidence = intent_scores[final_intent]
        else:
            final_intent = 'stop'
            final_confidence = 0.5
            # 详细信息
        details = {
            'rule': {'intent': rule_intent, 'confidence': rule_conf},
            'tfidf': {'intent': tfidf_intent, 'confidence': tfidf_conf},
            'word2vec': {'intent': w2v_intent, 'confidence': w2v_conf},
            'final': {'intent': final_intent, 'confidence': final_confidence}
        }
        
        return final_intent, final_confidence, details
    
    def command_callback(self, msg):
        """ROS回调函数"""
        text = msg.data
        rospy.loginfo(f"接收到指令: {text}")
        
        # 三层融合分类
        intent, confidence, details = self.classify(text)
        
        # 发布结果
        self.intent_pub.publish(intent)
        self.confidence_pub.publish(float(confidence))
        
        # 日志输出详细信息
        rospy.loginfo(f"分类结果: {intent} (置信度: {confidence:.2f})")
        rospy.loginfo(f"  - 规则匹配: {details['rule']['intent']} ({details['rule']['confidence']:.2f})")
        rospy.loginfo(f"  - TF-IDF: {details['tfidf']['intent']} ({details['tfidf']['confidence']:.2f})")
        rospy.loginfo(f"  - Word2Vec: {details['word2vec']['intent']} ({details['word2vec']['confidence']:.2f})")
    
    def save_models(self):
        """保存训练好的模型"""
        import pickle
        from gensim.models import Word2Vec
        
        # 保存TF-IDF模型
        model_dir = os.path.join(os.path.dirname(__file__), '../models')
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'tfidf_model.pkl'), 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'classifier': self.tfidf_classifier
            }, f)
        
        # 保存Word2Vec模型
        self.word2vec_model.save(os.path.join(model_dir, 'word2vec_model.bin'))
        
        rospy.loginfo("模型已保存")

def main():
    
    classifier = ThreeLayerIntentClassifier()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass