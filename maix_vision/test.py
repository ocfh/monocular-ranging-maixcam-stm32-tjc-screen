from maix import camera,display,app,image
import cv2
import numpy as np
#相机初始化
cam = camera.Camera(640,480,image.Format.FMT_BGR888)
#屏幕初始化
disp = display.Display()
#存储HSV阈值--要识别的颜色 (黑色)
low_hsv = np.array([0, 0, 0])
top_hsv = np.array([180, 255, 60])

#顺序  左上 右上 右下 左下
#x+y 最小左上  x+y最大 右下  y-x最小右上  y-x最大左下
#[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]   nu2
def sort_xy(nu2):
    temp = np.zeros((4,2), dtype="float32")  #dtype=float
    s = nu2.sum(axis=1)
    temp[0] = nu2[np.argmin(s)]#左上
    temp[2] = nu2[np.argmax(s)]#右下
    c = np.diff(nu2,axis=1)
    temp[1] = nu2[np.argmin(c)]#右上
    temp[3] = nu2[np.argmax(c)]#左下
    return temp



while not app.need_exit():
    #读取一帧画面 Maix
    img = cam.read()
    #将画面从Maix转换成Opencv格式
    img_cv_bgr = image.image2cv(img,ensure_bgr=False, copy=False)
    #将BGR图片进行色彩交换成HSV格式
    img_cv_hsv = cv2.cvtColor(img_cv_bgr,cv2.COLOR_BGR2HSV)
    #进行掩模筛选，生成二值化图片
    cv_mask = cv2.inRange(img_cv_hsv,low_hsv,top_hsv)


    #创建另一视图的画布 (320*240)
    hb_image = np.zeros((240,320,3), dtype=np.uint8)

    #在二值图中找出所有得轮廓(白色区域得边界线)
    contours = cv2.findContours(cv_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    #条件判断
    if contours:
        c = max(contours,key=cv2.contourArea) #过滤
        if cv2.contourArea(c) > 500: #过滤比较小噪点
            #计算轮廓的周长
            zc = cv2.arcLength(c, True)
            #多边形拟合，取得顶点坐标
            #取周长的%2作为拟合参数
            approx = cv2.approxPolyDP(c, zc*0.02, True)
            #判断是什么形状
            cout = len(approx)
            text = "?"
            if cout == 3:
                text = "SJX"
            elif cout == 4: 
                text = "SBX"
            else:
                text = "Other"
            #描绘轮廓
            cv2.drawContours(img_cv_bgr,[approx],-1,(0, 255, 0),2)
            #文本显示
            x,y = approx[0][0]   #取第一个顶点坐标
            cv2.putText(img_cv_bgr,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)
            #x,y,w,h = cv2.boundingRect(c)
            #cv2.rectangle(img_cv_bgr,(x,y),(x+w,y+h),(0,0,255), 2)
            #将顶点坐标按顺序左上 右上 右下 左下排序
            if len(approx) == 4:
                new_array = approx.reshape(4, 2).astype(np.float32)
                rect = sort_xy(new_array)
                #循环在画面中将4个顶点画出圆点显示
                for i,t in enumerate(rect):
                    cv2.circle(img_cv_bgr,tuple(t.astype(int)),8,(0, 0, 255),-1)
                    cv2.putText(img_cv_bgr,str(i),tuple(t.astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)
                #计算拉平后的 最大宽 和 最大高  勾股定理
                weithA = np.linalg.norm(rect[1]-rect[0])
                weithB = np.linalg.norm(rect[2]-rect[3])
                max_weith = max(int(weithA),int(weithB))
                hightA = np.linalg.norm(rect[3]-rect[0])
                hightB = np.linalg.norm(rect[1]-rect[2])
                max_hight = max(int(hightA ),int(hightB))

                #构造目标矩阵
                dest_jx = np.array([
                    [0,0],
                    [max_weith-1,0],
                    [max_weith-1,max_hight-1],
                    [0,max_hight-1]
                ], dtype=np.float32)
                #透视变换,生成矩形规则
                M = cv2.getPerspectiveTransform(rect,dest_jx)
                #执行透视图片转换
                ts_image = cv2.warpPerspective(img_cv_bgr,M,(max_weith, max_hight))
                #将透视拉平后的矩形等比压缩进画布
                #压缩比例系数
                x = min(320/max_weith,240/max_hight)
                hbjx_w = int(max_weith*x)
                hbjx_h = int(max_hight*x)
                pj_image = cv2.resize(ts_image, (hbjx_w, hbjx_h))
                #居中显示
                x_jg = (320 - hbjx_w)//2
                y_jg = (240 - hbjx_h)//2
                #数组[y起始:y结束，x起始:x结束]    数组切片规则
                hb_image[y_jg:y_jg+hbjx_h,x_jg:x_jg+hbjx_w] = pj_image
            

    #将掩模画面显示到屏幕上
    #图片拼接 画面一分为二 一半原图一半显示透视拉平居中图
    #把原相机画面也压缩，方便后续拼接
    pj_image1 = cv2.resize(img_cv_bgr, (320, 240))
    new_image = np.hstack((pj_image1,hb_image))
    disp.show(image.cv2image(new_image,bgr=True, copy=False))