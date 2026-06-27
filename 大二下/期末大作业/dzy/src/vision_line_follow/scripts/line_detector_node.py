#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge


class LineDetectorNode:
    def __init__(self):
        rospy.init_node("line_detector_node")

        self.bridge = CvBridge()

        # -----------------------------
        # ROS 参数
        # -----------------------------
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")

        # ROI 参数：只看图像中下方地面区域
        self.roi_top_ratio = rospy.get_param("~roi_top_ratio", 0.55)
        self.roi_bottom_ratio = rospy.get_param("~roi_bottom_ratio", 1.0)
        self.roi_left_ratio = rospy.get_param("~roi_left_ratio", 0.05)
        self.roi_right_ratio = rospy.get_param("~roi_right_ratio", 0.95)

        # -----------------------------
        # 红色 HSV 阈值
        # -----------------------------
        # OpenCV HSV: H 范围 0~179
        self.red1_lower = np.array(rospy.get_param("~red1_lower", [0, 80, 80]), dtype=np.uint8)
        self.red1_upper = np.array(rospy.get_param("~red1_upper", [12, 255, 255]), dtype=np.uint8)

        self.red2_lower = np.array(rospy.get_param("~red2_lower", [168, 80, 80]), dtype=np.uint8)
        self.red2_upper = np.array(rospy.get_param("~red2_upper", [179, 255, 255]), dtype=np.uint8)

        # -----------------------------
        # 白色胶带 HSV 阈值
        # -----------------------------
        # 为了避免浅灰地面被大量吃进去，这里默认比上一版稍微收紧：
        # 白色特点：S低、V高
        self.white_lower = np.array(rospy.get_param("~white_lower", [0, 0, 160]), dtype=np.uint8)
        self.white_upper = np.array(rospy.get_param("~white_upper", [179, 85, 255]), dtype=np.uint8)

        # -----------------------------
        # 红白同时存在判定参数
        # -----------------------------
        # 核心逻辑：
        # 1. 红色必须靠近白色
        # 2. 白色必须靠近红色
        # 3. 同一个候选轮廓内必须同时包含足够红色和白色
        self.require_red_white_pair = rospy.get_param("~require_red_white_pair", True)

        # 红色向外膨胀，用于寻找附近白色
        self.red_dilate_size = rospy.get_param("~red_dilate_size", 29)

        # 白色向外膨胀，用于判断红色是否靠近白色
        self.white_dilate_size = rospy.get_param("~white_dilate_size", 29)

        # 合成线区域后的形态学处理
        self.line_close_size = rospy.get_param("~line_close_size", 13)
        self.line_open_size = rospy.get_param("~line_open_size", 3)

        # -----------------------------
        # 轮廓过滤参数
        # -----------------------------
        self.min_contour_area = rospy.get_param("~min_contour_area", 80)
        self.max_contour_area = rospy.get_param("~max_contour_area", 100000)

        # 不再严格过滤高度，因为真实线可能在图像里形成长条
        self.max_contour_height_ratio = rospy.get_param("~max_contour_height_ratio", 1.0)
        self.max_bbox_area = rospy.get_param("~max_bbox_area", 300000)

        # 一个候选线区域内至少要包含多少红色像素
        self.min_red_pixels_per_contour = rospy.get_param("~min_red_pixels_per_contour", 8)

        # 新增：一个候选线区域内至少要包含多少白色像素
        self.min_white_pixels_per_contour = rospy.get_param("~min_white_pixels_per_contour", 25)

        # 新增：白色像素占候选轮廓面积的最低比例
        self.min_white_ratio_per_contour = rospy.get_param("~min_white_ratio_per_contour", 0.01)

        # 新增：白色像素相对红色像素的最低比例
        # 用于过滤“几乎全红、只有极少白色噪声”的纯红障碍物
        self.min_white_to_red_ratio = rospy.get_param("~min_white_to_red_ratio", 0.10)

        # -----------------------------
        # 中心线提取参数
        # -----------------------------
        # 每隔多少像素扫描一条水平带
        self.row_step = rospy.get_param("~row_step", 8)

        # 扫描带半高
        self.band_half_height = rospy.get_param("~band_half_height", 3)

        # 一条扫描带里至少有多少个线区域像素，才认为有效
        self.min_pixels_per_row = rospy.get_param("~min_pixels_per_row", 25)

        # 将一条扫描带里的 x 像素按连续段分组，超过这个 gap 认为是不同段
        self.segment_gap = rospy.get_param("~segment_gap", 8)

        # 每个连续段至少多少像素
        self.min_segment_width = rospy.get_param("~min_segment_width", 12)

        # 相邻扫描行中心点允许的最大跳变比例
        self.max_center_jump_ratio = rospy.get_param("~max_center_jump_ratio", 0.35)

        # -----------------------------
        # 拟合与预瞄参数
        # -----------------------------
        self.min_points_for_fit = rospy.get_param("~min_points_for_fit", 3)

        # 拟合阶数：1 更稳，2 更适合弯道
        self.poly_degree = rospy.get_param("~poly_degree", 2)

        # 默认把预瞄点放在 ROI 底部附近，让小车中心尽量压线
        self.lookahead_at_bottom = rospy.get_param("~lookahead_at_bottom", True)

        # 如果不使用底部预瞄，则用这个比例
        self.lookahead_ratio_in_roi = rospy.get_param("~lookahead_ratio_in_roi", 0.55)

        # 拟合曲线允许向底部外推的最大像素
        self.max_extrapolate_px = rospy.get_param("~max_extrapolate_px", 50)

        # -----------------------------
        # ROS 通信
        # -----------------------------
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        self.error_pub = rospy.Publisher("/line_follow/error", Float32, queue_size=10)
        self.valid_pub = rospy.Publisher("/line_follow/valid", Bool, queue_size=10)
        self.debug_pub = rospy.Publisher("/line_follow/debug_image", Image, queue_size=1)

        self.red_mask_pub = rospy.Publisher("/line_follow/red_mask", Image, queue_size=1)
        self.white_mask_pub = rospy.Publisher("/line_follow/white_mask", Image, queue_size=1)
        self.line_mask_pub = rospy.Publisher("/line_follow/line_mask", Image, queue_size=1)

        rospy.loginfo("line_detector_node started.")
        rospy.loginfo("Subscribing image topic: %s", self.image_topic)

    def make_odd(self, value, min_value=1):
        value = int(value)
        value = max(value, min_value)
        if value % 2 == 0:
            value += 1
        return value

    def split_x_segments(self, xs):
        """
        将一条扫描带上的 x 像素分成若干连续段。
        返回 [(x_start, x_end, count), ...]
        """
        if len(xs) == 0:
            return []

        xs = np.sort(xs)
        segments = []

        start = xs[0]
        prev = xs[0]
        count = 1

        for x in xs[1:]:
            if x - prev > self.segment_gap:
                if count >= self.min_segment_width:
                    segments.append((int(start), int(prev), int(count)))
                start = x
                count = 1
            else:
                count += 1

            prev = x

        if count >= self.min_segment_width:
            segments.append((int(start), int(prev), int(count)))

        return segments

    def choose_segment_center(self, segments, prev_cx, roi_w):
        """
        从一条扫描带的多个连续段中选择当前要跟踪的线段。
        - 第一行：优先选择靠近图像中心、且较宽的段
        - 后续行：优先选择与上一行中心连续的段
        """
        if not segments:
            return None

        image_center = roi_w / 2.0

        candidates = []
        for x_start, x_end, count in segments:
            cx = 0.5 * (x_start + x_end)
            candidates.append((cx, count, x_start, x_end))

        if prev_cx is None:
            best = min(
                candidates,
                key=lambda item: abs(item[0] - image_center) - 0.02 * item[1]
            )
            return int(best[0])

        max_jump = roi_w * self.max_center_jump_ratio

        candidates = sorted(candidates, key=lambda item: abs(item[0] - prev_cx))
        best = candidates[0]

        if abs(best[0] - prev_cx) > max_jump:
            return None

        return int(best[0])

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge conversion failed: %s", str(e))
            return

        h, w = frame.shape[:2]

        # -----------------------------
        # 1. 截取 ROI
        # -----------------------------
        x1 = int(w * self.roi_left_ratio)
        x2 = int(w * self.roi_right_ratio)
        y1 = int(h * self.roi_top_ratio)
        y2 = int(h * self.roi_bottom_ratio)

        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        roi = frame[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]

        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # -----------------------------
        # 2. HSV 分割红色与白色候选
        # -----------------------------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, self.red1_lower, self.red1_upper)
        red_mask2 = cv2.inRange(hsv, self.red2_lower, self.red2_upper)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        # 红色轻微去噪
        red_open_kernel = np.ones((3, 3), np.uint8)
        red_close_kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, red_open_kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, red_close_kernel)

        # 白色轻微去噪
        white_open_kernel = np.ones((3, 3), np.uint8)
        white_close_kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, white_open_kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, white_close_kernel)

        # -----------------------------
        # 3. 红白共存约束
        # -----------------------------
        red_dilate_size = self.make_odd(self.red_dilate_size, 3)
        white_dilate_size = self.make_odd(self.white_dilate_size, 3)

        red_dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (red_dilate_size, red_dilate_size)
        )
        white_dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (white_dilate_size, white_dilate_size)
        )

        red_near_area = cv2.dilate(red_mask, red_dilate_kernel, iterations=1)
        white_near_area = cv2.dilate(white_mask, white_dilate_kernel, iterations=1)

        # 只有靠近白色的红色才保留
        red_near_white = cv2.bitwise_and(red_mask, white_near_area)

        # 只有靠近红色的白色才保留
        white_near_red = cv2.bitwise_and(white_mask, red_near_area)

        if self.require_red_white_pair:
            # 严格模式：最终线区域只由“红白相互邻近”的区域组成
            line_mask = cv2.bitwise_or(red_near_white, white_near_red)
        else:
            # 非严格模式：兼容旧逻辑
            line_mask = cv2.bitwise_or(red_mask, white_near_red)

        # -----------------------------
        # 4. 形态学处理完整线区域
        # -----------------------------
        close_size = self.make_odd(self.line_close_size, 3)
        open_size = self.make_odd(self.line_open_size, 3)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_size, close_size)
        )
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (open_size, open_size)
        )

        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, close_kernel)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, open_kernel)

        # -----------------------------
        # 5. 轮廓过滤，得到 filtered_line_mask
        # -----------------------------
        contours, _ = cv2.findContours(
            line_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_line_mask = np.zeros_like(line_mask)
        kept_contours = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_contour_area:
                continue

            if area > self.max_contour_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            if bh > roi_h * self.max_contour_height_ratio:
                continue

            if bw * bh > self.max_bbox_area:
                continue

            local_contour_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted_cnt = cnt - np.array([[[x, y]]])
            cv2.drawContours(local_contour_mask, [shifted_cnt], -1, 255, thickness=-1)

            local_red = red_near_white[y:y + bh, x:x + bw]
            red_inside = cv2.bitwise_and(local_red, local_contour_mask)
            red_pixels = cv2.countNonZero(red_inside)

            local_white = white_near_red[y:y + bh, x:x + bw]
            white_inside = cv2.bitwise_and(local_white, local_contour_mask)
            white_pixels = cv2.countNonZero(white_inside)

            contour_area_for_ratio = max(1, cv2.countNonZero(local_contour_mask))
            white_ratio = float(white_pixels) / float(contour_area_for_ratio)
            white_to_red_ratio = float(white_pixels) / float(max(1, red_pixels))

            # 候选区域必须包含足够红色
            if red_pixels < self.min_red_pixels_per_contour:
                continue

            # 新增：候选区域必须包含足够白色
            if self.require_red_white_pair:
                if white_pixels < self.min_white_pixels_per_contour:
                    continue

                if white_ratio < self.min_white_ratio_per_contour:
                    continue

                if white_to_red_ratio < self.min_white_to_red_ratio:
                    continue

            cv2.drawContours(filtered_line_mask, [cnt], -1, 255, thickness=-1)
            kept_contours += 1

            # debug：画候选线区域外框
            cv2.rectangle(
                debug,
                (x + x1, y + y1),
                (x + bw + x1, y + bh + y1),
                (0, 255, 255),
                2
            )

            # debug：显示该区域红白像素数量，方便判断是否误检
            cv2.putText(
                debug,
                "R:{} W:{}".format(red_pixels, white_pixels),
                (x + x1, max(y + y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1
            )

        # -----------------------------
        # 6. 在完整线 mask 上按行扫描，提取中心线点
        # -----------------------------
        centers = []
        prev_cx = None

        for yy in range(roi_h - 1, 0, -self.row_step):
            y_low = max(0, yy - self.band_half_height)
            y_high = min(roi_h, yy + self.band_half_height + 1)

            band = filtered_line_mask[y_low:y_high, :]
            _, xs = np.where(band > 0)

            if len(xs) < self.min_pixels_per_row:
                continue

            segments = self.split_x_segments(xs)
            cx = self.choose_segment_center(segments, prev_cx, roi_w)

            if cx is None:
                continue

            cy = yy
            centers.append((cx, cy))
            prev_cx = cx

            cv2.circle(debug, (cx + x1, cy + y1), 4, (0, 255, 0), -1)

        # -----------------------------
        # 7. 拟合中心线并计算误差
        # -----------------------------
        valid = False
        error_norm = 0.0

        image_center_x_in_roi = roi_w // 2
        center_global_x = x1 + image_center_x_in_roi

        # 图像中心线
        cv2.line(debug, (center_global_x, y1), (center_global_x, y2), (255, 0, 0), 2)

        if len(centers) >= self.min_points_for_fit:
            centers = sorted(centers, key=lambda p: p[1])

            pts = np.array(centers, dtype=np.float32)
            ys = pts[:, 1]
            xs = pts[:, 0]

            try:
                fit_degree = min(self.poly_degree, len(centers) - 1)
                coeff = np.polyfit(ys, xs, deg=fit_degree)

                y_min = int(np.min(ys))
                y_max = int(np.max(ys))

                if self.lookahead_at_bottom:
                    desired_y = roi_h - 1
                    if desired_y - y_max > self.max_extrapolate_px:
                        lookahead_y = y_max
                    else:
                        lookahead_y = desired_y
                else:
                    lookahead_y = int(roi_h * self.lookahead_ratio_in_roi)

                lookahead_y = int(np.clip(lookahead_y, 0, roi_h - 1))
                lookahead_x = int(np.polyval(coeff, lookahead_y))
                lookahead_x = int(np.clip(lookahead_x, 0, roi_w - 1))

                error = lookahead_x - image_center_x_in_roi
                error_norm = float(error) / float(roi_w / 2.0)
                error_norm = float(np.clip(error_norm, -1.0, 1.0))

                valid = True

                # 画拟合曲线：只画在可靠范围附近，避免远处乱外推
                curve_y_start = max(0, y_min)
                curve_y_end = min(roi_h - 1, y_max + self.max_extrapolate_px)

                curve_points = []
                for yy in range(curve_y_start, curve_y_end + 1, 5):
                    xx = int(np.polyval(coeff, yy))
                    if 0 <= xx < roi_w:
                        curve_points.append((xx + x1, yy + y1))

                for i in range(1, len(curve_points)):
                    cv2.line(
                        debug,
                        curve_points[i - 1],
                        curve_points[i],
                        (255, 0, 255),
                        3
                    )

                # 画预瞄点
                gx = lookahead_x + x1
                gy = lookahead_y + y1
                cv2.circle(debug, (gx, gy), 10, (0, 0, 255), -1)

                # 画误差线
                cv2.line(debug, (center_global_x, gy), (gx, gy), (0, 0, 255), 2)

            except Exception as e:
                rospy.logwarn_throttle(1.0, "polyfit failed: %s", str(e))
                valid = False
                error_norm = 0.0

        else:
            valid = False
            error_norm = 0.0

        # -----------------------------
        # 8. 发布结果
        # -----------------------------
        self.error_pub.publish(Float32(data=error_norm))
        self.valid_pub.publish(Bool(data=valid))

        cv2.putText(
            debug,
            "valid: {}  error: {:.3f}  centers: {}  contours: {}".format(
                valid,
                error_norm,
                len(centers),
                kept_contours
            ),
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if valid else (0, 0, 255),
            2
        )

        cv2.putText(
            debug,
            "mode: red-white coexist line mask",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

            red_msg = self.bridge.cv2_to_imgmsg(red_mask, encoding="mono8")
            red_msg.header = msg.header
            self.red_mask_pub.publish(red_msg)

            white_msg = self.bridge.cv2_to_imgmsg(white_mask, encoding="mono8")
            white_msg.header = msg.header
            self.white_mask_pub.publish(white_msg)

            line_msg = self.bridge.cv2_to_imgmsg(filtered_line_mask, encoding="mono8")
            line_msg.header = msg.header
            self.line_mask_pub.publish(line_msg)

        except Exception as e:
            rospy.logwarn("debug image publish failed: %s", str(e))

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = LineDetectorNode()
    node.spin()