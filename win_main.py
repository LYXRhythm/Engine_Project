import sys
import cv2
import time
from algorithm.target2.yolo import YOLO as YOLO_target2
from algorithm.target1.yolo import YOLO as YOLO_target1

from win1 import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class My_mainwindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(My_mainwindow, self).__init__()
        self.setupUi(self)
        self.timer_video = QtCore.QTimer()  # 创建定时器
        self.init_slot()
        self.output_folder = 'output/'  # 输出的检测图片视频存在这里
        self.cap = cv2.VideoCapture()  # 读取视频
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.vid_writer = None
        self.target1_model = None
        self.target2_model = None
        self.model_index = 0  # 模型选择 0:划痕检测 1：异物检测
        self.init_model()  # 初始化模型们

    def init_slot(self):  # 信号与槽链接
        self.pushButton_detect_img.clicked.connect(self.detect_img)
        self.pushButton_detect_video.clicked.connect(self.detect_video)
        self.pushButton_pause.clicked.connect(self.video_pause)
        self.pushButton_stopdetect.clicked.connect(self.stopdetect)
        self.comboBox.currentIndexChanged.connect(self.change_model)
        self.timer_video.timeout.connect(self.show_video_frame)  # 定时器超时，将槽绑定至show_video_frame
        self.doubleSpinBox.valueChanged.connect(self.SpinBox_conf_changed)
        self.horizontalSlider.valueChanged.connect(self.Slider_conf_changed)
        self.checkBox.toggled.connect(self.checkBox_cuda)

    def init_model(self):  # 初始化模型
        #  初始化划痕检测模型
        self.target1_model = YOLO_target1()
        # self.target1_model.cuda = False
        #  初始化异物检测模型
        self.target2_model = YOLO_target2()
        # self.target2_model.cuda = False
        print("初始化模型成功")

    def change_model(self):
        self.model_index = self.comboBox.currentIndex()
        if self.model_index == 0:
            self.target1_model.confidence = 0.5
            self.horizontalSlider.setValue(50)
            self.doubleSpinBox.setValue(50)
            if self.checkBox.isChecked():
                self.target1_model.cuda = True
            else:
                self.target1_model.cuda = False
            print('当前为划痕检测模式')
            #  显示模型加载完成
            QtWidgets.QMessageBox.information(self, u"Notice", u"模型加载完成，当前为划痕检测模式", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
        elif self.model_index == 1:
            self.target2_model.confidence = 0.5
            self.horizontalSlider.setValue(50)
            self.doubleSpinBox.setValue(50)
            if self.checkBox.isChecked():
                self.target2_model.cuda = True
            else:
                self.target2_model.cuda = False
            print('当前为异物检测模式')
            #  显示模型加载完成
            QtWidgets.QMessageBox.information(self, u"Notice", u"模型加载完成，当前为异物检测模式", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            print('更换检测模式出错！')

    #  检测图片
    def detect_img(self):
        print('button_image_open')
        name_list = []
        try:  # 读取选中图片路径到 img_name
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        except OSError as reason:
            print('文件打开出错啦！核对路径是否正确' + str(reason))
        else:
            # 判断图片是否为空
            if not img_name:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                img = cv2.imread(img_name)  # 得到原图img
                print("img_name:", img_name)
                # 进行图像检测**********************此处加入检测算法******************
                # 进行图像检测**********************此处加入检测算法******************
                # 进行图像检测**********************此处加入检测算法******************
                # 得到检测图像
                if self.model_index == 0:  # 划痕检测
                    if self.target1_model is not None:
                        img_out = self.target1_model.detect_image_ori(img)
                        print('cuda:')
                        print(self.target1_model.cuda)
                    else:
                        img_out = img
                        print("choose no model")
                elif self.model_index == 1:  # 异物检测
                    if self.target2_model is not None:
                        img_out = self.target2_model.detect_image_ori(img)
                        print('cuda:')
                        print(self.target2_model.cuda)
                    else:
                        img_out = img
                        print("choose no model")
                else:
                    pass
                #  获取当前系统时间，作为img文件名
                now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
                print(now)
                file_extension = img_name.split('.')[-1]  # 获得文件后缀名
                new_filename = now + '.' + file_extension  # 获得文件后缀名
                #file_path = self.output_folder + 'img_output/' + new_filename
                #cv2.imwrite(file_path, img_out)

                # 原图 显示在界面
                self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                          QtGui.QImage.Format_RGB32)
                self.label_origin_output.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.label.setScaledContents(True)  # 设置图像自适应界面大小
                # 检测结果 显示在界面
                self.result1 = cv2.cvtColor(img_out, cv2.COLOR_BGR2BGRA)
                self.result1 = cv2.resize(self.result1, (640, 480), interpolation=cv2.INTER_AREA)
                self.QtImg1 = QtGui.QImage(self.result1.data, self.result1.shape[1], self.result1.shape[0],
                                           QtGui.QImage.Format_RGB32)
                self.label_detection_output.setPixmap(QtGui.QPixmap.fromImage(self.QtImg1))
                self.label.setScaledContents(True)  # 设置图像自适应界面大小

    #  检测视频
    def detect_video(self):
        print('button_video_open')
        try:  # 读取选中视频路径到 video_name
            video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        except OSError as reason:
            print('文件打开出错啦！核对路径是否正确' + str(reason))
        else:
            flag = self.cap.open(video_name)
            if not flag:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # -------------------------写入视频----------------------------------#
                #fps, w, h, save_path = self.set_video_name_and_path()
                #self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                #self.timer_video.start(30)  # 以30ms为间隔，启动或重启定时器
                # 进行视频识别时，关闭其他按键点击功能
                self.pushButton_detect_video.setDisabled(True)
                self.pushButton_detect_img.setDisabled(True)

    # 定义视频帧显示操作，在上一个函数 检测视频detect_video() 中被调用
    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()
        if img is not None:
            #  检测图片***********************此处加入检测算法*****************
            #  检测图片***********************此处加入检测算法*****************
            #  检测图片***********************此处加入检测算法*****************
            #  将检测完的图片存在 img_out
            if self.model_index == 0:  # 划痕检测
                if self.target1_model is not None:
                    img_out = self.target1_model.detect_image_ori(img)
                else:
                    img_out = img
                    print("choose no model")
            elif self.model_index == 1:  # 异物检测
                if self.target2_model is not None:
                    img_out = self.target2_model.detect_image_ori(img)
                else:
                    img_out = img
                    print("choose no model")
            else:
                pass
            self.vid_writer.write(img_out)  # 检测结果写入视频

            show = cv2.resize(img, (640, 480))  # 直接将原视频进行显示
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label_origin_output.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label_origin_output.setScaledContents(True)  # 设置图像自适应界面大小

            show1 = cv2.resize(img_out, (640, 480))  # 直接将检测结果进行显示
            self.result1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)
            showImage1 = QtGui.QImage(self.result1.data, self.result1.shape[1], self.result1.shape[0],
                                      QtGui.QImage.Format_RGB888)
            self.label_detection_output.setPixmap(QtGui.QPixmap.fromImage(showImage1))
            self.label_detection_output.setScaledContents(True)  # 设置图像自适应界面大小

        else:
            self.timer_video.stop()
            # 读写结束，释放资源
            self.cap.release()  # 释放video_capture资源
            self.vid_writer.release()  # 释放video_writer资源
            self.label_origin_output.clear()
            self.label_detection_output.clear()
            # 视频帧显示期间，禁用其他检测按键功能
            self.pushButton_detect_video.setDisabled(False)
            self.pushButton_detect_img.setDisabled(False)

    # 设定视频的保存路径、获取视频信息，在函数 检测视频detect_video() 中被调用
    def set_video_name_and_path(self):
        # 获取当前系统时间，作为img和video的文件名
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # 获取视频信息
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 视频检测结果存储位置
        save_path = self.output_folder + 'video_output/' + now + '.mp4'
        return fps, w, h, save_path

    #  视频的暂停与继续
    def video_pause(self):
        self.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.timer_video.isActive() == True and self.num_stop % 2 == 1:
            self.pushButton_pause.setText(u'暂停检测')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer_video.blockSignals(True)
        # 继续检测
        else:
            self.num_stop = self.num_stop + 1
            self.pushButton_pause.setText(u'继续检测')

    #  停止检测
    def stopdetect(self):
        # self.timer_video.stop()
        self.cap.release()  # 释放video_capture资源
        self.vid_writer.release()  # 释放video_writer资源
        self.label_origin_output.clear()  # 清空label画布
        self.label_detection_output.clear()  # 清空label画布
        # 启动其他检测按键功能
        self.pushButton_detect_video.setDisabled(False)
        self.pushButton_detect_img.setDisabled(False)

        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        if self.num_stop % 2 == 0:
            print("Reset stop/begin!")
            self.pushButton_pause.setText(u'暂停/继续')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)

    #  改变置信度
    def SpinBox_conf_changed(self):
        self.horizontalSlider.setValue(self.doubleSpinBox.value())
        conf = self.doubleSpinBox.value()
        if self.model_index == 0:
            self.target1_model.confidence = conf * 0.01
            print('当前置信度：' + str(self.target1_model.confidence))
        elif self.model_index == 1:
            self.target2_model.confidence = conf * 0.01
            print('当前置信度：' + str(self.target2_model.confidence))
        else:
            pass

    #  改变置信度
    def Slider_conf_changed(self):
        self.doubleSpinBox.setValue(self.horizontalSlider.value())
        conf = self.horizontalSlider.value()
        if self.model_index == 0:
            self.target1_model.confidence = conf * 0.01
            print('当前置信度：' + str(self.target1_model.confidence))
        elif self.model_index == 1:
            self.target2_model.confidence = conf * 0.01
            print('当前置信度：' + str(self.target2_model.confidence))
        else:
            pass

    def checkBox_cuda(self):  #  checkbox是否选中
        if self.checkBox.isChecked():
            if self.model_index == 0:
                self.target1_model.cuda = True
                print('cuda: True')
            elif self.model_index == 1:
                self.target2_model.cuda = True
                print('cuda: True')
            else:
                pass
        else:
            if self.model_index == 0:
                self.target1_model.cuda = False
                print('cuda: False')
            elif self.model_index == 1:
                self.target2_model.cuda = False
                print('cuda: False')
            else:
                pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywin1 = My_mainwindow()
    mywin1.show()
    sys.exit(app.exec())
