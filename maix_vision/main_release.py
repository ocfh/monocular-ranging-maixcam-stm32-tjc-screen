# ====================================================================
# 项目说明：MaixCam 黑色四边形轮廓检测、单目视觉测距、串口UART上报距离
# 功能：1. HSV黑色阈值分割 2. 轮廓筛选+多边形判定 3. 四点排序透视矫正
#      4. 相似三角形单目测距 5. 串口节流+防抖上报数据给单片机
#      6. 双画面拼接显示（原图+矫正俯视图）
# 串口协议格式：$D:距离*100,W:像素宽,H:像素高#\r\n
# ====================================================================
# 模块导入
# ====================================================================
from maix import camera, display, app, image
from maix.peripheral import uart
import cv2
import numpy as np
import time
# 20cm:505.6153
# 15cm:500.76
# ===================== 全局参数配置区（可根据场景修改） =====================
REAL_WIDTH_CM = 6.5        # 被测物体实际物理宽度(厘米)，测距标定基准
FOCAL_LENGTH = 500.76       # 相机像素焦距，需实物标定调整，直接影响测距精度
SEND_INTERVAL_MS = 200     # 串口最小发送间隔，防止高频刷屏导致单片机阻塞
last_send_time = 0         # 记录上一次串口发送时间戳，用于节流控制
last_obj_width = 0         # 上一帧物体像素宽度，用于宽度防抖滤波
scale = 0                  # 矫正图缩放比例缓存
# ====================================================================
# 1. 硬件初始化：摄像头、显示屏、串口UART0
# ====================================================================
# 初始化摄像头 分辨率640*480，输出BGR888彩色图像
cam = camera.Camera(640, 480, image.Format.FMT_BGR888)
# 初始化屏幕显示
disp = display.Display()

# 串口初始化增加异常捕获，接线错误不会直接崩溃程序
serial = None
try:
    device = "/dev/ttyS0"  # Maix开发板硬件串口0设备路径
    serial = uart.UART(device, 115200)  # 波特率115200，与单片机匹配
    print("串口初始化成功 UART0 115200")
except Exception as e:
    print("串口初始化失败，请检查引脚/设备路径:", e)

# ====================================================================
# 2. HSV颜色阈值：黑色物体分割范围
# HSV：H色相[0-180] S饱和度[0-255] V亮度[0-255]
# 黑色：低饱和度、低亮度区间
# ====================================================================
hsv_arr_min = np.array([0, 0, 0])
hsv_arr_max = np.array([180, 255, 60])

# ====================================================================
# 3. 顶点排序函数（四边形4个角点标准化排序）
# 输入：4个无序轮廓顶点数组
# 返回：按顺序 [左上,右上,右下,左下] 规整坐标
# 原理：x+y最小=左上；x+y最大=右下；y-x最小=右上；y-x最大=左下
# ====================================================================
def sort_xy(nu2):
    temp = np.zeros((4, 2), dtype=np.float32)
    s = nu2.sum(axis=1)       # 每个点x+y求和
    temp[0] = nu2[np.argmin(s)]   # 左上：坐标和最小
    temp[2] = nu2[np.argmax(s)]   # 右下：坐标和最大
    c = np.diff(nu2, axis=-1)     # 每个点y-x差值
    temp[1] = nu2[np.argmin(c)]   # 右上：差值最小
    temp[3] = nu2[np.argmax(c)]   # 左下：差值最大
    return temp

# ====================================================================
# 4. 主循环：图像采集、预处理、轮廓检测、测距、串口发送、画面渲染
# ====================================================================
while not app.need_exit():
    # 读取摄像头一帧图像（Maix image对象）
    img = cam.read()
    # Maix图像转OpenCV BGR格式，不拷贝节省内存
    imgcv_bgr = image.image2cv(img, ensure_bgr=False, copy=False)
    # BGR转HSV色彩空间，方便颜色阈值分割
    imghsv = cv2.cvtColor(imgcv_bgr, cv2.COLOR_BGR2HSV)
    # HSV二值掩码：只保留黑色区域，其余全黑
    imgcv_mask = cv2.inRange(imghsv, hsv_arr_min, hsv_arr_max)
    # 查找所有外层轮廓，压缩轮廓点减少运算量
    contours, _ = cv2.findContours(imgcv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 右侧矫正图画布：固定240*320黑色背景，无四边形时全黑
    pic_img = np.zeros((240, 320, 3), dtype=np.uint8)

    # 存在轮廓才执行后续检测逻辑
    if contours:
        # 筛选面积最大轮廓，过滤小噪声色块
        c = max(contours, key=cv2.contourArea)
        # 轮廓面积阈值过滤，小于500像素判定为噪声丢弃
        if cv2.contourArea(c) > 500:
            # 计算轮廓周长
            lenc = cv2.arcLength(c, True)
            # 多边形逼近，简化轮廓顶点，0.02倍周长作为精度阈值
            approx = cv2.approxPolyDP(c, lenc * 0.02, True)
            vertex_count = len(approx)  # 获取多边形顶点数量

            # 根据顶点数量区分图形类型
            if vertex_count == 3:
                text = "Triangle"    # 三角形
            elif vertex_count == 4:
                text = "quadrilateral" # 四边形（目标检测对象）
            else:
                text = "Other"       # 其他不规则图形

            # 在原图绘制轮廓绿色边框
            cv2.drawContours(imgcv_bgr, [approx], -1, (0, 255, 0), 2)
            # 获取轮廓第一个顶点坐标
            x, y = approx[0][0]
            # 在轮廓左上角绘制图形类型文字
            cv2.putText(imgcv_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            # ★ 仅四边形执行透视矫正、测距、串口上报逻辑
            if vertex_count == 4:
                # 重塑顶点数组为4行2列坐标
                new_array = approx.reshape((4, 2))
                # 调用函数标准化四点顺序：左上、右上、右下、左下
                rect = sort_xy(new_array)

                # 遍历4个角点，绘制圆点+序号标记
                for i, pt in enumerate(rect):
                    center = tuple(pt.astype(int))
                    cv2.circle(imgcv_bgr, center, 5, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(imgcv_bgr, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                # 计算上下两条边像素宽度，取最大值作为物体像素宽度
                widthA = np.linalg.norm(rect[1] - rect[0])
                widthB = np.linalg.norm(rect[2] - rect[3])
                max_width = max(int(widthA), int(widthB))

                # 计算左右两条边像素高度，取最大值作为物体像素高度
                heightA = np.linalg.norm(rect[3] - rect[0])
                heightB = np.linalg.norm(rect[2] - rect[1])
                max_height = max(int(heightA), int(heightB))

                # ============ 串口测距发送逻辑 ==========
                pixel_width = max_width  # 物体成像像素宽度

                # 防除零保护，避免分母为0报错
                if pixel_width >= 1:
                    # 相似三角形测距公式：距离(cm) = (真实宽度 * 焦距) / 像素宽度
                    distance_cm = (REAL_WIDTH_CM * FOCAL_LENGTH) / pixel_width
                    dis_int = int(distance_cm*100)  # 放大100倍转整数，减少串口浮点传输

                    now_ms = int(time.time() * 1000)  # 当前毫秒时间戳
                    width_diff = abs(pixel_width - last_obj_width) # 前后帧宽度变化量

                    # 双重防抖节流：1.发送间隔达标 2.物体宽度变化超3像素才发送
                    if (now_ms - last_send_time) > SEND_INTERVAL_MS and width_diff > 3:
                        last_send_time = now_ms      # 更新发送时间戳
                        last_obj_width = pixel_width # 更新缓存像素宽度

                        # 自定义串口协议：$起始符 #结束符，单片机可快速分割解析
                        data_str = f"$D:{ dis_int},W:{max_width},H:{max_height}#\r\n"

                        # 串口存在时执行发送，捕获发送异常防止卡死
                        if serial is not None:
                            try:
                                serial.write(data_str.encode("utf-8"))
                                print(data_str.strip()) # 控制台打印发送数据用于调试
                            except Exception as e:
                                print("串口发送异常:", e)

                # 定义透视变换目标画布四个顶点（标准俯视矩形）
                dest_jx = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]
                ], dtype=np.float32)
                # 计算透视变换矩阵
                M = cv2.getPerspectiveTransform(rect.astype(np.float32), dest_jx)
                # 执行透视变换，矫正倾斜四边形为俯视平面图
                img_rt = cv2.warpPerspective(imgcv_bgr, M, (max_width, max_height))

                # ---------- 将矫正图等比缩放并居中放入右侧画布 ----------
                if max_width > 0 and max_height > 0:
                    # 计算缩放系数，保证图像完整放入320*240画布不拉伸
                    scale = min(320 / max_width, 240 / max_height)
                    new_w = int(max_width * scale)
                    new_h = int(max_height * scale)
                    img_resized = cv2.resize(img_rt, (new_w, new_h))
                    # 计算居中偏移坐标
                    x_offset = (320 - new_w) // 2
                    y_offset = (240 - new_h) // 2
                    # 将缩放后的矫正图贴入黑色画布居中位置
                    pic_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    # ---------- 左右双画面拼接 ----------
    # 原图缩放到320*240尺寸
    left_img = cv2.resize(imgcv_bgr, (320, 240))
    # 水平拼接：左=原图，右=透视矫正图
    combined = np.hstack((left_img, pic_img))

    # ========== 画面文字标识渲染配置 ==========
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color_white = (255, 255, 255)   # BGR白色文字
    color_black = (0, 0, 0)          # 黑色半透明背景

    # 半透明黑色遮罩，提升文字可读性
    overlay = combined.copy()
    # 左侧文字背景框
    cv2.rectangle(overlay, (5, 5), (120, 35), (0, 0, 0), -1)
    # 右侧文字背景框
    cv2.rectangle(overlay, (325, 5), (440, 35), (0, 0, 0), -1)
    # 原图与遮罩50%透明度混合
    combined = cv2.addWeighted(overlay, 0.5, combined, 0.5, 0)

    # 左侧画面标注原图
    cv2.putText(combined, "Original", (10, 30), font, font_scale, color_white, thickness, cv2.LINE_AA)
    # 右侧画面标注矫正图
    cv2.putText(combined, "Warped", (330, 30), font, font_scale, color_white, thickness, cv2.LINE_AA)

    # 将OpenCV图像转回Maix图像并刷新屏幕显示
    disp.show(image.cv2image(combined, bgr=True, copy=False))

# # ====================================================================
# # 模块导入
# # ====================================================================
# from maix import camera, display, app, image
# from maix.peripheral import uart
# import cv2
# import numpy as np
# import time

# #全局参数配置区
# # 测距标定参数
# REAL_WIDTH_CM = 8.5        # 现实中物体的真实宽度 (cm)
# FOCAL_LENGTH = 800        # 相机焦距参数 (需根据实际标定调整)
# SEND_INTERVAL_MS = 200     # 串口最小发送间隔 (ms)，防止刷屏淹没单片机
# last_send_time = 0         # 记录上次发送时间
# last_obj_width = 0         # 防抖：上一帧物体宽度
# scale = 0
# # ====================================================================
# # 1. 硬件初始化
# # ====================================================================
# cam = camera.Camera(640, 480, image.Format.FMT_BGR888)
# disp = display.Display()

# # 串口初始化加捕获，接线错不会直接导致程序崩溃
# serial = None
# try:
#     #指定的设备路径格式初始化串口0
#     device = "/dev/ttyS0"
#     serial = uart.UART(device, 115200)
#     print("串口初始化成功 UART0 115200")
# except Exception as e:
#     print("串口初始化失败，请检查引脚/设备路径:", e)
# # ====================================================================
# # 2. HSV 阈值（黑色）
# # ====================================================================
# hsv_arr_min = np.array([0, 0, 0])
# hsv_arr_max = np.array([180, 255, 60])



# # ====================================================================
# # 3. 顶点排序函数（专用于四边形）
# # ====================================================================
# def sort_xy(nu2):
#     temp = np.zeros((4, 2), dtype=np.float32)
#     s = nu2.sum(axis=1)
#     temp[0] = nu2[np.argmin(s)]   # 左上
#     temp[2] = nu2[np.argmax(s)]   # 右下
#     c = np.diff(nu2, axis=-1)
#     temp[1] = nu2[np.argmin(c)]   # 右上
#     temp[3] = nu2[np.argmax(c)]   # 左下
#     return temp

# # ====================================================================
# # 4. 主循环
# # ====================================================================
# while not app.need_exit():
#     img = cam.read()
#     imgcv_bgr = image.image2cv(img, ensure_bgr=False, copy=False)

#     imghsv = cv2.cvtColor(imgcv_bgr, cv2.COLOR_BGR2HSV)
#     imgcv_mask = cv2.inRange(imghsv, hsv_arr_min, hsv_arr_max)

#     contours, _ = cv2.findContours(imgcv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 每帧重新创建画布（右侧显示矫正图，全黑背景）
#     pic_img = np.zeros((240, 320, 3), dtype=np.uint8)

#     if contours:
#         c = max(contours, key=cv2.contourArea)
#         if cv2.contourArea(c) > 500:
#             lenc = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, lenc * 0.02, True)
#             vertex_count = len(approx)

#             # 形状分类
#             if vertex_count == 3:
#                 text = "Triangle"
#             elif vertex_count == 4:
#                 text = "quadrilateral"
#             else:
#                 text = "Other"

#             cv2.drawContours(imgcv_bgr, [approx], -1, (0, 255, 0), 2)
#             x, y = approx[0][0]
#             cv2.putText(imgcv_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

#             # ★ 只有四边形才执行透视变换
#             if vertex_count == 4:
#                 new_array = approx.reshape((4, 2))
#                 rect = sort_xy(new_array)

#                 for i, pt in enumerate(rect):
#                     center = tuple(pt.astype(int))
#                     cv2.circle(imgcv_bgr, center, 5, (0, 0, 0), 2, cv2.LINE_AA)
#                     cv2.putText(imgcv_bgr, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

#                 widthA = np.linalg.norm(rect[1] - rect[0])
#                 widthB = np.linalg.norm(rect[2] - rect[3])
#                 max_width = max(int(widthA), int(widthB))

#                 heightA = np.linalg.norm(rect[3] - rect[0])
#                 heightB = np.linalg.norm(rect[2] - rect[1])
#                 max_height = max(int(heightA), int(heightB))


#                 # ============ 串口测距发送逻辑 ==========
#                 #最大像素宽度
#                 pixel_width = max_width 
                
#                 # 【防除零】双重保险
#                 if pixel_width >= 1:
#                 #if pixel_width >= 1:e
#                     distance_cm = (REAL_WIDTH_CM * FOCAL_LENGTH) / pixel_width
#                     dis_int = int(distance_cm*100)
#                     now_ms = int(time.time() * 1000)

#                     #时间节流 + 宽度防抖（变化大于3像素才发送）绝对值
#                     width_diff = abs(pixel_width - last_obj_width)
#                     if (now_ms - last_send_time) > SEND_INTERVAL_MS and width_diff > 3:
#                         last_send_time = now_ms
#                         last_obj_width = pixel_width

#                         # 规范协议，增加起始符方便单片机解析
#                         #data_str = f"$D:{distance_cm:.1f},W:{max_width},H:{max_height}#\r\n"
#                         data_str = f"$D:{ dis_int},W:{max_width},H:{max_height}#\r\n"
#                         if serial is not None:
#                             try:
#                                 serial.write(data_str.encode("utf-8"))
#                                 #print("发送距离数据:", data_str.strip())
#                                 print(data_str.strip())
#                             except Exception as e:
#                                 print("串口发送异常:", e)

#                 dest_jx = np.array([
#                     [0, 0],
#                     [max_width - 1, 0],
#                     [max_width - 1, max_height - 1],
#                     [0, max_height - 1]
#                 ], dtype=np.float32)

#                 M = cv2.getPerspectiveTransform(rect.astype(np.float32), dest_jx)
#                 img_rt = cv2.warpPerspective(imgcv_bgr, M, (max_width, max_height))

#                 # ---------- 将矫正图等比缩放并居中放入画布 ----------
#                 #if max_height>0:
#                 if max_width > 0 and max_height > 0:
#                     scale = min(320 / max_width, 240 / max_height)
#                     new_w = int(max_width * scale)
#                     new_h = int(max_height * scale)
#                     img_resized = cv2.resize(img_rt, (new_w, new_h))

#                     x_offset = (320 - new_w) // 2
#                     y_offset = (240 - new_h) // 2

#                     pic_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

#     # ---------- 拼接并显示 ----------
#     left_img = cv2.resize(imgcv_bgr, (320, 240))
#     combined = np.hstack((left_img, pic_img))

#     # ========== ★ 新增：添加文字标识 ==========
#     # 定义文字参数
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.8
#     thickness = 2
#     color_white = (255, 255, 255)   # BGR 白色
#     color_black = (0, 0, 0)          # 黑色

#     # 为左侧“原图”添加半透明背景框（可选）
#     # 画一个半透明矩形作为背景，使文字更清晰
#     overlay = combined.copy()
#     # 左侧背景：从 (5,5) 到 (120, 35)
#     cv2.rectangle(overlay, (5, 5), (120, 35), (0, 0, 0), -1)   # 黑色填充
#     # 右侧背景：从 (325, 5) 到 (440, 35)
#     cv2.rectangle(overlay, (325, 5), (440, 35), (0, 0, 0), -1)
#     # 混合透明度（alpha=0.5）
#     combined = cv2.addWeighted(overlay, 0.5, combined, 0.5, 0)

#     # 在左侧写 "原图"
#     cv2.putText(combined, "Original", (10, 30), font, font_scale, color_white, thickness, cv2.LINE_AA)
#     # 在右侧写 "矫正图" (如果右侧有图，都显示，无图则显示"Warped"也无妨)
#     cv2.putText(combined, "Warped", (330, 30), font, font_scale, color_white, thickness, cv2.LINE_AA)

#     # ==========================================

#     # 显示最终图像
#     disp.show(image.cv2image(combined, bgr=True, copy=False))



