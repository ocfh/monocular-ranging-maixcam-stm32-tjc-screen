
# 导入所需的库
from maix import image, display, camera, uart,nn,touchscreen,app,gpio, pinmap, time
import struct
import cv2
import numpy as np
import math
from collections import defaultdict


# 定义黑色的HSV颜色范围（H: 0-180, S: 0-255, V: 0-60）
black_lower = np.array([0, 0, 0])    # HSV下限
black_upper = np.array([180, 255, 93])  # HSV上限
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 单组颜色阈值
thresholds = {
    "lower": [0, 0, 0],
    "upper": [180, 115, 100]
}

distance = 0 #距离
IsRun = True #是否进行阈值修改标志位

#--------------------------------------------------------------------------------------串口发送
def send_float_rate(serial, dis, d):
    print(dis,d)
    D = int(dis*100) 
    L = int(d*100)
    print(D,L)                
    if L<65535 and D<65535:
        packed_data = struct.pack(">BHHB", 0XFE, D, L, 0XFF)  # 打包找到D的消息
        serial.write(packed_data)  # 发送打包后的数据
        print(f"发送数据: {packed_data}")  # 打印发送的数据



#-------------------------------------------------------------------------------------"""找到图像中最大和第二大的四边形轮廓并返回其四个顶点"""(前置处理支持)
def get_rectangle_points(input_image, hsv_lower=None, hsv_upper=None):

    """
    找到图像中所有四边形轮廓并返回面积最大和第二大的四边形顶点
    
    参数:
        input_image: 输入图像（可以是BGR图像或二值图）
        hsv_lower: HSV下限（仅当输入为BGR图像时使用）
        hsv_upper: HSV上限（仅当输入为BGR图像时使用）
    
    返回:
        largest_points: 最大四边形的顶点坐标或None
        second_largest_points: 第二大四边形的顶点坐标或None
        mask: 使用的二值图像
    """
    # 自动检测输入图像类型
    if len(input_image.shape) == 3:  # 彩色图像
        if hsv_lower is None or hsv_upper is None:
            raise ValueError("彩色图像需要提供hsv_lower和hsv_upper参数")
            
        # 转换为HSV并进行颜色过滤
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        # mask = cv2.erode(mask, None, iterations=1)
    elif len(input_image.shape) == 2:  # 二值图像
        mask = input_image.copy()
    else:
        raise ValueError("输入图像必须是3通道(BGR)或单通道(二值)")
    
    # 查找轮廓
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    quadrilaterals = []  # 存储所有四边形及其面积
    
    for contour in contours:
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 多边形近似
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果是四边形
        if len(approx) == 4:
            # 计算当前四边形面积
            area = cv2.contourArea(approx)
            
            # 获取四个顶点
            points = approx.reshape(4, 2)
            
            # 按顺序排列顶点(左上、右上、右下、左下)
            points = sorted(points, key=lambda x: x[0])  # 按x坐标排序
            left_points = sorted(points[:2], key=lambda x: x[1])  # 左侧点按y排序
            right_points = sorted(points[2:], key=lambda x: x[1])  # 右侧点按y排序
            ordered_points = np.array([left_points[0], right_points[0], 
                                     right_points[1], left_points[1]])
            
            quadrilaterals.append((area, ordered_points))
    
    # 按面积降序排序
    quadrilaterals.sort(reverse=True, key=lambda x: x[0])
    
    # 准备返回值
    largest = quadrilaterals[0][1] if len(quadrilaterals) > 0 else None
    second_largest = quadrilaterals[1][1] if len(quadrilaterals) > 1 else None
    
    return largest, second_largest, mask


#------------------------------------------------------------------------------------------------ """将矩形向内缩小指定像素"""（图像前置处理使用）
def shrink_rect(points, shrink_pixels=6):
    center = np.mean(points, axis=0)
    vectors = points - center
    lengths = np.sqrt(np.sum(vectors**2, axis=1))
    lengths[lengths == 0] = 1
    unit_vectors = vectors / lengths[:, np.newaxis]
    return (points - (unit_vectors * shrink_pixels)).astype(np.int32)

#------------------------------------------------------------------------------------------------ """矫正图像"""（图像前置处理使用）
def Correction(input_image, points):
    """执行透视变换校正"""
    if points is None or len(points) != 4:
        return input_image
    
    try:
        # 确保点坐标是float32类型
        src_points = np.array(points, dtype=np.float32)
        
        # 目标点坐标 (根据实际需要调整)
        dst_points = np.float32([
            [150, 0], 
            [489, 0], 
            [489, 480],
            [150, 480]
        ])
        
        # 计算变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        filtered = cv2.warpPerspective(input_image, matrix, (640, 480),flags=cv2.INTER_NEAREST)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)  # 先开运算去白噪点
        # filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)  # 再闭运算补黑空洞

        # 连通域分析（直接过滤小区域）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered)
        min_area = 0  # 根据实际调整最小像素面积
        filtered_binary = np.zeros_like(filtered)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_binary[labels == i] = 255
        # # 侵蚀操作（消除白噪声）
        # eroded = cv2.erode(corrected_img, kernel, iterations=1)
        return filtered_binary
    except Exception as e:
        print("校正失败:", e)
        return input_image
    
#---------------------------------------------------------------------------------------------------------------获取主要图形（图像前置处理使用）
def get_shapes(input_image,points,reduced_pixels = 7):
    #缩小
    new_points = shrink_rect(points,reduced_pixels)
    # 全黑背景
    black_bg = np.zeros_like(input_image)  
    # inner_mask = cv2.rectangle(np.zeros_like(input_image), (new_points[0][0], new_points[0][1]), 
    #                                                         (new_points[2][0], new_points[2][1]), 
    #                                                         255, -1)  # 内部区域=255
    inner_mask = cv2.fillPoly(black_bg.copy(), [new_points], color=255, lineType=cv2.LINE_AA)
    black_bg[inner_mask == 255] = input_image[inner_mask == 255]  # 内部填充result
    # opening = cv2.morphologyEx(black_bg, cv2.MORPH_OPEN, kernel)
    return black_bg

#---------------------------------------------------------------------------------------------------------------四个有顺序的点的到最长线段（测距使用）
def get_Longest(quad_points):
    # 确保输入是4个点
    if quad_points is None :
        print("算长度时没找到四个点")
        return 0
    if len(quad_points) != 4:
        print("算长度时不够四个点")
        return 0
    # 转换为numpy数组
    points = np.array(quad_points, dtype=np.float32)
    # 计算四条边的长度
    edge_lengths = [
        np.linalg.norm(points[1] - points[0]),  # 上边
        np.linalg.norm(points[2] - points[1]),  # 右边
        np.linalg.norm(points[3] - points[2]),  # 下边
        np.linalg.norm(points[0] - points[3])   # 左边
    ]
    # 找出最长边
    edge_lengths = sorted(edge_lengths)
    L = (edge_lengths[2]+edge_lengths[3])/2
    # print("L:",L)
    return L

#------------------------------------------------------------------------------------------------------------------测距得到距离（测距使用）
def get_distance(quad_points, k_list, qx_list):
    distance = 0.0  # 默认值
    L = get_Longest(quad_points)  # 获取最长边（或其他计算）
    
    # 检查 L 是否有效
    if L <= 0:
        return 0.0  # 直接返回 0，避免无效计算
    
    # 遍历 qx_list（避免越界，只到 len(qx_list)-1）
    for i in range(len(qx_list) - 1):  # 注意：-1 防止 qx_list[i+1] 越界
        threshold = qx_list[i] + (qx_list[i+1] - qx_list[i]) / 2
        if L > threshold:
            distance = k_list[i] * 29.7 / L
            # print(f"正在使用的k值: {k_list[i]}, 像素值: {qx_list[i]}, i值: {i}")
            return distance  # 直接返回，避免后续覆盖
    
    # 如果所有条件都不满足，检查最后一个区间（单独处理，避免越界）
    if L > qx_list[-1]:  # 检查是否大于最后一个 qx_list 值
        distance = k_list[-1] * 29.7 / L
        # print(f"正在使用的k值: {k_list[-1]}, 像素值: {qx_list[-1]}, i值: {len(qx_list)-1}")
    else:
        distance = 200  # 默认值（如果 L 不大于任何阈值）
    
    return distance

#--------------------------------------------------------------------------------------------------功能一     识别基本图像
def Mode_1_sjx(input_image):
    print("正在寻找三角形")
    x, y =0, 0
    D = 0
    out_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        epsilon = 0.06 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        i = 0
        
        # 多边形物体处理
        if len(approx) == 3 :
            cv2.drawContours(out_image, [approx], 0, (0, 255, 0), 2)
            side_lengths_pixels = []
            # 遍历所有顶点，计算相邻点距离
            for i in range(len(approx)):
                # 当前点
                x1, y1 = approx[i][0]  
                # 下一个点（如果是最后一个点，则连接回第一个点）
                x2, y2 = approx[(i + 1) % len(approx)][0]  
                
                # 计算欧氏距离（像素单位）
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                side_lengths_pixels.append(length)
                
                # 在图像上标注边长（可选）
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(
                    out_image, 
                    f"{length:.1f}", 
                    (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 255), 
                    1
                )
            D = sum(side_lengths_pixels) / len(approx) * 29.7 / 480
    else:
        D = 0

    return out_image,D
def Mode_1_zfx(input_image):
    print("正在寻找正方形")
    x, y =0, 0
    D = 0
    out_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        epsilon = 0.09 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        i = 0
        
        # 多边形物体处理
        if len(approx) == 4 :
            cv2.drawContours(out_image, [approx], 0, (0, 255, 0), 2)
            side_lengths_pixels = []
            # 遍历所有顶点，计算相邻点距离
            for i in range(len(approx)):
                # 当前点
                x1, y1 = approx[i][0]  
                # 下一个点（如果是最后一个点，则连接回第一个点）
                x2, y2 = approx[(i + 1) % len(approx)][0]  
                
                # 计算欧氏距离（像素单位）
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                side_lengths_pixels.append(length)
                
                # 在图像上标注边长（可选）
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(
                    out_image, 
                    f"{length:.1f}", 
                    (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 255), 
                    1
                )
            D = sum(side_lengths_pixels) / len(approx) * 29.7 / 480

        
    else:
        D = 0

    return out_image,D
def Mode_1_yx(input_image):
    print("正在寻找圆形")
    x, y =0, 0
    D = 0
    out_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        i = 0
        
        # 圆形物体处理
        if len(approx) > 4 :
            (x1, y1), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(out_image, (int(x1), int(y1)), int(r), (255, 0, 255), 2)
            D = 0 if 480 == 0 else 29.7 * 2 * r / 480
        

    else:
        D = 0

    return out_image,D


def Mode_1_all(input_image):
    print("正在寻找所有图形")
    x, y =0, 0
    D = 0
    out_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        epsilon = 0.09 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        i = 0
        
        # 多边形物体处理
        if len(approx) == 4 :
            cv2.drawContours(out_image, [approx], 0, (0, 255, 0), 2)
            side_lengths_pixels = []
            # 遍历所有顶点，计算相邻点距离
            for i in range(len(approx)):
                # 当前点
                x1, y1 = approx[i][0]  
                # 下一个点（如果是最后一个点，则连接回第一个点）
                x2, y2 = approx[(i + 1) % len(approx)][0]  
                
                # 计算欧氏距离（像素单位）
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                side_lengths_pixels.append(length)
                
                # 在图像上标注边长（可选）
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(
                    out_image, 
                    f"{length:.1f}", 
                    (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 255), 
                    1
                )
            D = sum(side_lengths_pixels) / len(approx) * 29.7 / 480
    
        if len(approx) == 3 :
            cv2.drawContours(out_image, [approx], 0, (0, 255, 0), 2)
            side_lengths_pixels = []
            # 遍历所有顶点，计算相邻点距离
            for i in range(len(approx)):
                # 当前点
                x1, y1 = approx[i][0]  
                # 下一个点（如果是最后一个点，则连接回第一个点）
                x2, y2 = approx[(i + 1) % len(approx)][0]  
                
                # 计算欧氏距离（像素单位）
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                side_lengths_pixels.append(length)
                
                # 在图像上标注边长（可选）
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(
                    out_image, 
                    f"{length:.1f}", 
                    (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 255), 
                    1
                )
            D = sum(side_lengths_pixels) / len(approx) * 29.7 / 480


        
        # 圆形物体处理
        if len(approx) > 4 :
            (x1, y1), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(out_image, (int(x1), int(y1)), int(r), (255, 0, 255), 2)
            D = 0 if 480 == 0 else 29.7 * 2 * r / 480


    
    else:
        D = 0

    return out_image,D


#--------------------------------------------------------------------------------------------------得到邻角和对角信息（多边形支持函数）
def find_parallel_segments(outlines):
    """
    从轮廓数据中找出满足特定角度关系的点对
    
    参数:
        outlines: 轮廓数据列表，每个轮廓由多个点组成，每个点包含以下信息:
            [0]: 当前点x坐标
            [1]: 当前点y坐标
            [2]: 前一个点x坐标
            [3]: 前一个点y坐标
            [4]: 下一个点x坐标
            [5]: 下一个点y坐标
            [6]: 前一线段长度
            [7]: 后一线段长度
            [8]: 内角角度(0-360°)
            [9]: 平分线角度(0-360°)
    
    返回:
        tuple: 包含两个列表的元组:
            - 第一个列表: 所有轮廓中平分线角度差接近180°的点对(平行关系)
            - 第二个列表: 所有轮廓中平分线角度差接近90°的点对(垂直关系)
            每个匹配点对包含两个点的坐标、索引和平分线角度信息
    """
    all_contour_matches_parallel = []  # 存储平行关系点对
    all_contour_matches_perpendicular = []  # 存储垂直关系点对
    
    for outline in outlines:
        contour_matches_parallel = []  # 当前轮廓的平行点对
        contour_matches_perpendicular = []  # 当前轮廓的垂直点对
        
        n = len(outline)
        for i in range(n):
            # 只处理近似直角的内角(90±20度范围内)(可调)
            if abs(outline[i][8] - 90) < 20:
                for j in range(i + 1, n):
                    # 同样只处理近似直角
                    if abs(outline[j][8] - 90) < 30:
                        # 计算平分线角度差(考虑360°循环)
                        angle_diff = abs((outline[i][9] - outline[j][9]) % 360)
                        angle_diff = min(angle_diff, 360 - angle_diff)  # 取最小差
                        
                        # 检查是否为近似180度(180±20度范围内)
                        if abs(angle_diff - 180) < 20:
                            match = {
                                'point1': {
                                    'index': i,
                                    'x': outline[i][0],
                                    'y': outline[i][1],
                                    'bisector_angle': outline[i][9]
                                },
                                'point2': {
                                    'index': j,
                                    'x': outline[j][0],
                                    'y': outline[j][1],
                                    'bisector_angle': outline[j][9]
                                }
                            }
                            contour_matches_parallel.append(match)
                            
                        # 检查是否为近似90度(90±20度范围内)
                        if abs(angle_diff - 90) < 20:
                            match = {
                                'point1': {
                                    'index': i,
                                    'x': outline[i][0],
                                    'y': outline[i][1],
                                    'bisector_angle': outline[i][9]
                                },
                                'point2': {
                                    'index': j,
                                    'x': outline[j][0],
                                    'y': outline[j][1],
                                    'bisector_angle': outline[j][9]
                                }
                            }
                            contour_matches_perpendicular.append(match)
                            
        all_contour_matches_parallel.append(contour_matches_parallel)
        all_contour_matches_perpendicular.append(contour_matches_perpendicular)
        
    return all_contour_matches_parallel, all_contour_matches_perpendicular

#----------------------------------------------------------------------------------已知一个正方形的对角坐标，得到四个顶点坐标（多边形支持函数）
def find_square_vertices_dui(diag1, diag2):
    """
    根据正方形对角坐标计算四个顶点（优化版）
    
    参数:
        diag1: 第一个对角点，格式为 (x, y)
        diag2: 第二个对角点，格式为 (x, y)
    
    返回:
        包含四个顶点坐标的numpy数组，形状为(4,2)，数据类型为np.int32，
        可直接用于OpenCV绘图函数，顶点顺序为顺时针或逆时针连续排列
    """
    # 输入验证
    if not (isinstance(diag1, (tuple, list))) or not (isinstance(diag2, (tuple, list))):
        raise ValueError("输入坐标必须是元组或列表格式")
    if len(diag1) != 2 or len(diag2) != 2:
        raise ValueError("坐标必须是二维点")
    
    # 转换为numpy数组并转换为float类型以提高计算精度
    A = np.array(diag1, dtype=np.float32)
    C = np.array(diag2, dtype=np.float32)
    
    # 计算中心点
    O = (A + C) / 2.0
    
    # 计算从中心到A的向量
    OA = A - O
    
    # 计算垂直向量（旋转90度）
    OB = np.array([-OA[1], OA[0]], dtype=np.float32)  # 逆时针旋转90度
    
    # 计算四个顶点（确保闭合顺序）
    B = O + OB
    D = O - OB  # 等价于 O + np.array([OA[1], -OA[0]])
    
    # 将四个顶点组合并转换为整数坐标
    vertices = np.array([A, B, C, D], dtype=np.float32)
    
    # 确保顶点顺序是连续的（对绘制闭合图形很重要）
    # 重新排序顶点，使其形成连续的环
    ordered_vertices = np.zeros((4, 2), dtype=np.int32)
    ordered_vertices[0] = np.round(A).astype(np.int32)
    ordered_vertices[1] = np.round(B).astype(np.int32)
    ordered_vertices[2] = np.round(C).astype(np.int32)
    ordered_vertices[3] = np.round(D).astype(np.int32)
    
    return ordered_vertices

#-------------------------------------------------------------------------------已知一个正方形的邻角坐标，得到四个顶点坐标（多边形支持函数）
def find_square_vertices_lin(corner1, corner2):
    """
    给定正方形的一条边的两个顶点，返回可能的正方形顶点集合。
    由于一条边可以对应两个正方形（分别在边的两侧），因此返回两个可能的正方形顶点列表。

    参数:
        corner1: 第一个顶点的坐标，格式为 (x1, y1)
        corner2: 第二个顶点的坐标，格式为 (x2, y2)

    返回:
        一个包含两个列表的列表，每个列表是对应的正方形四个顶点的坐标 [A, B, C, D]
    """
    # 确保输入是数值类型
    x1, y1 = float(corner1[0]), float(corner1[1])
    x2, y2 = float(corner2[0]), float(corner2[1])

    # 计算向量 AB
    dx = x2 - x1
    dy = y2 - y1

    # 情况1：逆时针旋转90度得到垂直向量 (-dy, dx)
    c1 = (x2 - dy, y2 + dx)
    d1 = (x1 - dy, y1 + dx)
    
    # 情况2：顺时针旋转90度得到垂直向量 (dy, -dx)
    c2 = (x2 + dy, y2 - dx)
    d2 = (x1 + dy, y1 - dx)

    # 转换为整数坐标并构建正方形顶点列表
    square1 = [
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (int(round(c1[0])), int(round(c1[1]))),
        (int(round(d1[0])), int(round(d1[1])))
    ]
    
    square2 = [
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (int(round(c2[0])), int(round(c2[1]))),
        (int(round(d2[0])), int(round(d2[1])))
    ]

    # 返回两种可能的正方形顶点列表
    return [square1, square2]



#---------------------------------------------------------------------------------------检查指定图形是否在目标图形中（多边形支持函数）
def is_polygon_inside_mask_optimized(vertices, mask, sample_points=20):
    """
    优化版多边形mask检查（采样+边界检查）
    参数:
        vertices: 多边形顶点坐标
        mask: 二值mask
        sample_points: 每条边采样点数
    """
    """使用 OpenCV 的 pointPolygonTest + 边界框优化"""
    h, w = mask.shape[:2]
    vertices = np.array(vertices, dtype=np.int32)
    
    # 1. 检查外接矩形是否越界
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)
    if x_min < 0 or y_min < 0 or x_max >= w or y_max >= h:
        return False
    
    # 2. 检查所有顶点是否在 mask 内（白区）
    for (x, y) in vertices:
        if mask[y, x] != 255:
            return False
    
    # 3. 检查边缘采样点（更严格）
    for i in range(len(vertices)):
        pt1 = vertices[i]
        pt2 = vertices[(i + 1) % len(vertices)]
        edge_length = np.linalg.norm(pt2 - pt1)
        num_samples = max(2, int(edge_length / 5))  # 动态采样
        for t in np.linspace(0, 1, num_samples):
            x = int(pt1[0] + t * (pt2[0] - pt1[0]))
            y = int(pt1[1] + t * (pt2[1] - pt1[1]))
            if mask[y, x] != 255:
                return False
    return True



#--------------------------------------------------------------------------------------------------功能二   寻找多边形
def Mode_2(input_image,expand_pixels=0):
    print("正在寻找多边形")
    D = 0 #真实最短长度
    num = 0
    d ,m_d = 100,100

    # 创建初始掩模（全黑）
    mask = np.zeros(input_image.shape[:2], dtype=np.uint8)
    #可画图片
    out_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2BGR)
    #找轮廓
    cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    outlines = [] #装每个轮廓的各个点的信息
    # 遍历每个检测到的轮廓    
    # print("len(cnts):",len(cnts))
    for cnt in cnts:
        # 计算多边形近似的epsilon值（基于轮廓周长）
        # 0.00775是经验系数，值越大近似越粗略
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        # 使用Douglas-Peucker算法进行多边形近似
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 将近似轮廓点从3维数组重塑为2维数组(N,2)
        approx = approx.reshape(-1, 2)
        # print("     len(approx):",len(approx))

        # 在掩模图像上填充近似轮廓（白色填充）
        mask = cv2.drawContours(mask, [approx], -1, 255, -1)
        
        # === 向量化计算部分 ===
        # 获取每个点的下一个点（循环移位）
        approx_rolled = np.roll(approx, -1, axis=0)
        # 获取每个点的前一个点（循环移位）
        prev_pts = np.roll(approx, 1, axis=0)
        # 下一个点就是rolled后的点
        next_pts = approx_rolled
        
        # 计算每条边的长度（当前点到下一个点的距离）
        edge_lengths = np.linalg.norm(approx_rolled - approx, axis=1)
        # 获取前一条边的长度（循环移位）
        prev_lengths = np.roll(edge_lengths, 1)
        
        # === 角度计算部分 ===
        # 计算当前点到下一个点的向量
        vecs = approx_rolled - approx
        # 计算前一个点到当前点的向量
        prev_vecs = approx - prev_pts
        
        # 计算两个向量的叉积（用于判断角度方向）
        cross = np.cross(prev_vecs, vecs)
        # 计算两个向量的点积（用于计算夹角）
        dot = np.einsum('ij,ij->i', prev_vecs, vecs)
        
        # 计算向量的模（长度）
        norms = np.linalg.norm(vecs, axis=1)
        prev_norms = np.linalg.norm(prev_vecs, axis=1)
        
        # 计算夹角的余弦值（限制在[-1,1]范围内）
        cos_theta = dot / (prev_norms * norms)
        # 将余弦值转换为角度（0-180度）
        angles = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        # 根据叉积符号确定是内角还是外角（0-360度）
        internal_angles = np.where(cross < 0, angles, 360 - angles)
        
        # === 平分线角度计算 ===
        # 归一化前一个向量
        norm_prev_vecs = prev_vecs / prev_norms[:, None]
        # 归一化当前向量
        norm_vecs = vecs / norms[:, None]
        # 计算角平分线向量（两个单位向量相加）
        bisector_vecs = norm_prev_vecs + norm_vecs
        # 计算平分线向量的角度（0-360度）
        bisector_angles = np.degrees(np.arctan2(bisector_vecs[:,1], bisector_vecs[:,0])) % 360
        # 将平分线角度旋转90度（垂直于平分线方向）
        bisector_angles = (bisector_angles - 90) % 360
        
        # === 构建输出数据结构 ===
        # 将所有信息按列堆叠成一个数组，然后转换为列表
        current_outline = np.column_stack([
            approx,              # 当前点坐标(x,y)
            prev_pts,            # 前一个点坐标(x,y)
            next_pts,            # 下一个点坐标(x,y)
            prev_lengths[:, None],  # 前一条边长度
            edge_lengths[:, None],  # 当前边长度
            internal_angles[:, None],  # 内角度数
            bisector_angles[:, None]   # 平分线角度
        ]).tolist()
        # 在绘制图像上标记所有近似轮廓点（绿色圆点）
        for pt in approx:
            cv2.circle(out_image, (int(pt[0]), int(pt[1])), 5, (0,255,0), thickness=-1)

        # 将当前轮廓的所有点信息添加到输出列表
        outlines.append(current_outline)

        if expand_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (expand_pixels, expand_pixels))
            mask = cv2.dilate(mask, kernel, iterations=1)
    if(len(outlines)):
        matches_dui,matches_lin = find_parallel_segments(outlines)
        # 处理对角
        for contour_idx, contour_matches in enumerate(matches_dui):
            for match in contour_matches:
                p1, p2 = match['point1'], match['point2']
                # 已知对角点画正方形
                diag1 = (p1['x'], p1['y'])
                diag2 = (p2['x'], p2['y'])
                vertices_dui = find_square_vertices_dui(diag1, diag2)
                
                Is_in = is_polygon_inside_mask_optimized(vertices_dui, mask, sample_points=100) #最后一个参数是一条边检测多少个点
                if Is_in:
                    # 绘制正方形
                    cv2.line(out_image, vertices_dui[0], vertices_dui[1], (0,0,255), 4)
                    cv2.line(out_image, vertices_dui[1], vertices_dui[2], (0,0,255), 4)
                    cv2.line(out_image, vertices_dui[2], vertices_dui[3], (0,0,255), 4)
                    cv2.line(out_image, vertices_dui[3], vertices_dui[0], (0,0,255), 4)
                    
                    # 计算四条边的长度
                    edge1 = np.linalg.norm(np.array(vertices_dui[0]) - np.array(vertices_dui[1]))
                    edge2 = np.linalg.norm(np.array(vertices_dui[1]) - np.array(vertices_dui[2]))
                    edge3 = np.linalg.norm(np.array(vertices_dui[2]) - np.array(vertices_dui[3]))
                    edge4 = np.linalg.norm(np.array(vertices_dui[3]) - np.array(vertices_dui[0]))
                    
                    # 计算四条边的平均长度
                    avg_edge_length = (edge1 + edge2 + edge3 + edge4) / 4
                    
                    # 计算实际长度D（单位：cm）
                    D = avg_edge_length * 29.7 / 480
                    
                    # 更新最小距离
                    if D > 4:  # 过滤掉过小的距离
                        m_d = min(D, m_d)
                # else:
                #     print("这个不行")
        # 处理邻角
        for contour_idx, contour_matches in enumerate(matches_lin):
            for match in contour_matches:
                p1, p2 = match['point1'], match['point2']
                # 已知边上的两个点画正方形
                corner1 = (int(p1['x']), int(p1['y']))  # 确保坐标是整数
                corner2 = (int(p2['x']), int(p2['y']))
                possible_squares = find_square_vertices_lin(corner1, corner2)
                
                for square_vertices in possible_squares:
                    # 检查正方形是否在mask内
                    Is_in = is_polygon_inside_mask_optimized(square_vertices, mask, sample_points=100) #最后一个参数是一条边检测多少个点
                    
                    if Is_in:
                        # 绘制完整的正方形
                        for i in range(4):
                            pt1 = (int(square_vertices[i][0]), int(square_vertices[i][1]))  # 确保坐标是整数
                            pt2 = (int(square_vertices[(i+1)%4][0]), int(square_vertices[(i+1)%4][1]))
                            cv2.line(out_image, pt1, pt2, (0,0,255), 4)
                        
                        # 计算四条边的长度
                        edge_lengths = []
                        for i in range(4):
                            pt1 = np.array(square_vertices[i])
                            pt2 = np.array(square_vertices[(i+1)%4])
                            edge_length = np.linalg.norm(pt2 - pt1)
                            edge_lengths.append(edge_length)
                        
                        # 计算平均边长
                        avg_edge_length = sum(edge_lengths) / 4
                        
                        # 计算实际长度D（单位：cm）
                        D = avg_edge_length * 29.7 / 480
                        
                        # 更新最小距离（过滤掉过小的距离）
                        if D > 4:
                            m_d = min(D, m_d)
                            # d = d / math.sqrt(2)
                        else:
                            pass
                            # print("这个不行")
                            # print(corner1, corner2)


    return out_image,m_d

#--------------------------------------------------------------------------------------------------功能三     旋转正方形
def Mode_3(input_image):
    x, y =0, 0
    D = 0
    out_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        epsilon = 0.09 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        i = 0
        
        
        
        # 找正方形物体处理
        if len(approx) == 4:
            cv2.drawContours(out_image, [approx], 0, (0, 255, 0), 2)
            side_lengths_pixels = []
            # 遍历所有顶点，计算相邻点距离
            for i in range(len(approx)):
                # 当前点
                x1, y1 = approx[i][0]  
                # 下一个点（如果是最后一个点，则连接回第一个点）
                x2, y2 = approx[(i + 1) % len(approx)][0]  
                
                # 计算欧氏距离（像素单位）
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                side_lengths_pixels.append(length)
                
                # 在图像上标注边长（可选）
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(
                    out_image, 
                    f"{length:.1f}", 
                    (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 255), 
                    1
                )
            D = sum(side_lengths_pixels) / len(approx) * 29.7 / 480
    else:
        D = 0

    return out_image,D

#-----------------------------------------------------------------------------------------------------------------按键调阈值部分\1/
# 按钮标签和位置
buttons = {
    "EXIT": [560, 379, 79, 100],  # x, y, width, height
    "-": [10, 200, 90, 90],
    "+": [280, 200, 90, 90],
    "Hmin": [0, 370, 70, 60],
    "Hmax": [70, 370, 70, 60],
    "Smin": [140, 370, 70, 60],
    "Smax": [210, 370, 70, 60],
    "Vmin": [280, 370, 70, 60],
    "Vmax": [350, 370, 70, 60],
    "CAL": [560, 1, 79, 79]  # 校准按钮
}



# 状态变量
current_threshold_index = None  # 当前选择的阈值索引
show_mask = False  # 是否显示掩膜
current_distance = 0
distance_options = [100, 120, 140, 160, 180, 200]
# k_values = [358.8529968261719, 308.188232421875, 271.1042785644531, 238.5777130126953, 213.58428955078125, 193.57876586914062]
l_ce = 0
# avg_k = 1315.94 #焦距平均值
k_list = [1208.2592485729692, 1245.2049794823233, 1277.9326262297454, 1285.267140809133, 1294.4502397017045, 1303.560712923506]
# k_list = []
qx_list = [369.14105224609375, 312.66644287109375, 271.6077880859375, 240.5748291015625, 215.11624145507812, 195.07818603515625]
#--------------------------------------------------------------------------------------------------------------------处理按键对应的功能
def button_clicked(button_name):
    global IsRun, current_threshold_index, show_mask, current_distance, k_values, l_ce, avg_k, k_list, qx_list
    
    if button_name == "EXIT":
        if show_mask:  # 如果当前显示掩膜，则只退出掩膜模式
            show_mask = False
            print("退出掩膜显示模式")
        else:  # 如果不显示掩膜，则退出程序
            IsRun = False
            if k_list:
                # avg_k = sum(k_values) / len(k_values)
                print(f"K值列表: {k_list}")
            else :
                print("没记录k列表")
    elif button_name in ["Hmin", "Hmax", "Smin", "Smax", "Vmin", "Vmax"]:
        # 映射按钮到索引 (0-5)
        index_map = {
            "Hmin": 0, "Hmax": 1,
            "Smin": 2, "Smax": 3,
            "Vmin": 4, "Vmax": 5
        }
        current_threshold_index = index_map[button_name]
        show_mask = True  # 点击阈值按钮时显示掩膜
        print(f"Selected {button_name}")
    elif button_name in ["+", "-"]:
        if current_threshold_index is not None:
            change_threshold(button_name)
            show_mask = True  # 调整阈值时显示掩膜
    elif button_name == "CAL":
        current_distance = (current_distance + 1) % 6
        show_mask = False  # 校准模式显示正常图像
        print(f"切换到校准距离: {distance_options[current_distance]}cm")
        
        # 记录当前k值（如果检测到目标）
        if l_ce > 0:
            distance = distance_options[current_distance-1]
            k = (l_ce * distance) / 29.7
            # k_list.append(k)
            k_list[current_distance-1] = k
            # qx_list.append(l_ce)
            qx_list[current_distance-1] = l_ce
            print(f"记录数据: k={k:.2f} (距离={distance}cm)(l_ce:{l_ce})")

#----------------------------------------------------------------------------------------------------------------------------改变阈值
def change_threshold(button_name):
    step = 1
    if current_threshold_index is None:
        return
    
    # 确定是调整 lower 还是 upper
    is_lower = current_threshold_index % 2 == 0  # 偶数索引是min（lower）
    bound = "lower" if is_lower else "upper"
    hsv_index = current_threshold_index // 2  # 0:H, 1:S, 2:V
    
    if button_name == "-":
        # 减少值，考虑不同分量的范围限制
        if hsv_index == 0:  # H分量
            thresholds[bound][hsv_index] = max(0, thresholds[bound][hsv_index] - step)
        else:  # S或V分量
            thresholds[bound][hsv_index] = max(0, thresholds[bound][hsv_index] - step)
    elif button_name == "+":
        # 增加值，考虑不同分量的范围限制
        if hsv_index == 0:  # H分量
            thresholds[bound][hsv_index] = min(180, thresholds[bound][hsv_index] + step)
        else:  # S或V分量
            thresholds[bound][hsv_index] = min(255, thresholds[bound][hsv_index] + step)
    
    print(f"Threshold updated: lower={thresholds['lower']}, upper={thresholds['upper']}")

#-------------------------------------------------------------------------------------------------------------------------按键是否被按下
def is_in_button(x, y, button_pos):
    return (button_pos[0] <= x <= button_pos[0] + button_pos[2] and
            button_pos[1] <= y <= button_pos[1] + button_pos[3])





x, y, last_x, last_y, last_pressed =  0, 0, -1, -1, False
pressed_already = False

#硬件初始化
cam = camera.Camera(640, 480, image.Format.FMT_BGR888)
device = "/dev/ttyS0"
serial = uart.UART(device, 115200)
disp = display.Display()
ts = touchscreen.TouchScreen()

while IsRun:
    # 读取摄像头和触摸输入
    img = cam.read()
    frame = image.image2cv(img, copy=False)
    x, y, pressed = ts.read()
    # 处理图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_value = hsv[240, 480]  # 注意OpenCV是(height,width)顺序
    mask = cv2.inRange(hsv, np.array(thresholds["lower"]), np.array(thresholds["upper"]))
    
    # 根据状态决定显示原始图像还是掩膜
    if show_mask:
        display_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        display_frame = frame.copy()
        # 在正常图像模式下也做目标检测
        largest_points, second_largest_points, _  = get_rectangle_points(frame.copy(),np.array(thresholds["lower"]),np.array(thresholds["upper"])) 
        
        l_ce = 0  # 重置

        if largest_points is not None:
            l_ce = get_Longest(largest_points)
            for point in largest_points:
                cv2.circle(display_frame, tuple(point), 4, (0, 0, 255), -1)
        
        if second_largest_points is not None:
            for point in second_largest_points:
                cv2.circle(display_frame, tuple(point), 4, (255, 0, 0), -1)
                
    # 绘制所有按钮
    for button_name, button_pos in buttons.items():
        color = (0, 255, 0) if button_name in ["Hmin", "Hmax", "Smin", "Smax", "Vmin", "Vmax", "+", "-"] else (0, 0, 255)
        cv2.rectangle(display_frame, 
                      (button_pos[0], button_pos[1]), 
                      (button_pos[0]+button_pos[2], button_pos[1]+button_pos[3]), 
                      color, 2)
        cv2.putText(display_frame, button_name, 
                   (button_pos[0]+5, button_pos[1]+18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # 绘制中心十字
    cv2.line(display_frame, (320,0), (320,480), (255,0,0), 1)
    cv2.line(display_frame, (0,240), (640,240), (255,0,0), 1)

    # 显示当前状态
    status_color = (0, 255, 255) if show_mask else (0, 255, 0)

    mode_text = "MASK (EXIT:back)" if show_mask else "NORMAL (EXIT:exit)"
    cv2.putText(display_frame, f"MODE: {mode_text}", 
               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(display_frame, f"DIST: {distance_options[current_distance]}cm", 
               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(display_frame, f"DATA: {len(k_list)}", 
               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)    
    # 显示当前阈值
    cv2.putText(display_frame, f"LOW: {thresholds['lower']}", 
               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(display_frame, f"UP: {thresholds['upper']}", 
               (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color,2)
    cv2.putText(display_frame, f"HSV: {hsv_value}", 
               (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color,2)
    # 触摸事件处理
    if x != last_x or y != last_y or pressed != last_pressed:
        last_x = x
        last_y = y
        last_pressed = pressed
        
    if pressed:
        pressed_already = True
    else:
        if pressed_already:
            pressed_already = False
            # 检查点击了哪个按钮
            for button_name, button_pos in buttons.items():
                if is_in_button(x, y, button_pos):
                    button_clicked(button_name)
                    break
    
    disp.show(image.cv2image(display_frame, copy=False))

black_lower = np.array(thresholds["lower"])
black_upper = np.array(thresholds["upper"])
#-----------------------------------------------------------------------------------------------------------------------按键调阈值部分/1\
D = 0 #最后得出的边长
bofangcishu = 0
bian = 2
last_data =0xff
while True:
    data = serial.read()
    
    if data:
        print(1)
        if 0x01 <= data[0] <= 0x06:
            last_data = data[0]

    img = cam.read()
    frame = image.image2cv(img, copy=False)

    retangle_points_img = frame.copy()
    largest_points1, second_largest_points1, binary_image1 = get_rectangle_points(frame.copy(),black_lower,black_upper)

    if largest_points1 is not None:
        distance = get_distance(largest_points1,k_list,qx_list)
        for point in largest_points1:
            cv2.circle(retangle_points_img, tuple(point), 5, (0, 0, 255), -1)
        if second_largest_points1 is not None:
            for point in second_largest_points1:
                cv2.circle(retangle_points_img, tuple(point), 5, (0, 0, 255), -1)
            
            shapes_image = get_shapes(binary_image1,second_largest_points1,4)#最后一个参数是内框缩小值

            corrected_img = Correction(shapes_image,largest_points1)


            if 0x01 == last_data :
                resulte,D = Mode_1_sjx(corrected_img.copy())
                send_float_rate(serial,distance,D)
            elif   0x02 == last_data :
                resulte,D = Mode_1_zfx(corrected_img.copy())
                send_float_rate(serial,distance,D)
            elif 0x03 == last_data :
                resulte,D = Mode_1_yx(corrected_img.copy())
                send_float_rate(serial,distance,D)
            elif 0x04 == last_data :
                resulte,D = Mode_2(corrected_img.copy(),10)#最后一个参数是掩模版膨胀值
                send_float_rate(serial,distance,D)
            elif last_data == 0x05: 
                resulte,D = Mode_3(corrected_img.copy())
                send_float_rate(serial,distance,D)
            elif   0x06 == last_data :
                resulte,D = Mode_1_all(corrected_img.copy())
                send_float_rate(serial,distance,D)
            else:
                resulte = shapes_image
                print("没有数据")
        else:
            retangle_points_img = frame
            resulte = frame
            print("没识别到内框")   
    else:#----------------------------------------------------------------------------------------------------------------------什么都不识别到的话
        retangle_points_img = frame
        resulte = frame
        print("没识别到外框")

        
    if D is not None:
        cv2.putText(resulte, f'D: {D:.2f}cm', (10, 50), 
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
    else:
        D = 0

    cv2.putText(resulte, f'fps: {time.fps():.2f}', (10, 80), 
        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
    cv2.putText(resulte, f"distance: {distance:.3f}", (10, 20), 
        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
#-----------------------------------------------------------------------------------------stop
    # 显示原始图像和结果
    img_show = image.cv2image(resulte, copy=False)
    disp.show(img_show)
