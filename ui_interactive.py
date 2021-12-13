#!/usr/bin/env python
# coding: utf-8

import sys
import os
import cv2
import time
import subprocess

from monitor import detect, LoadImages
from PyQt5.QtWidgets import QMainWindow, QApplication, QSlider, QFileDialog, QLabel, QPushButton, QFrame, QTextEdit, QProgressBar, QAction, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic, QtMultimedia

from qt_material import apply_stylesheet, list_themes

# from collections import deque

UNINITIALIZED = 0
INITIALIZED = 1
INFERENCE_INITIALIZED = 2
PLAYING = 3
STOPPED = 4
INFERENCE_FINISHED = 5
RECORD_FINISHED = 6

kor_cls = {
    "person": {'index': 0, 'kor': "사람"},
    "ship": {'index': 1, 'kor': "선박(단일)"},
    "ship_group": {'index': 2, 'kor': "선박(복수)"},
    "floating_house": {'index': 3, 'kor': "좌대"},
    "lighthouse": {'index': 4, 'kor': "등대"},
    "drone": {'index': 5, 'kor': "드론"},
    "floating_object": {'index': 6, 'kor': "부유물"},
    "construction_single": {'index': 7, 'kor': "구조물(단일)"},
    "construction_group": {'index': 8, 'kor': "구조물(복수)"},
    "attachment_normal": {'index': 9, 'kor': "부착물(정상)"},
    "attachment_abnormal": {'index': 10, 'kor': "부착물(비정상)"},
    "crack": {'index': 11, 'kor': "크랙"},
    "sunlightpannel_single": {'index': 12, 'kor': "태양광패널(단일)"},
    "sunlightpannel_group": {'index': 13, 'kor': "태양광패널(복수)"},
    "sunlightpannel_damage": {'index': 14, 'kor': "태양광패널(파손)"}
}


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, parent=None, id=0):
        QLabel.__init__(self, parent)
        self.id = id
        self.setFrameShape(QFrame.Box)
        self.setScaledContents(True)
        self.setFixedSize(100, 100)

    def mousePressEvent(self, event):
        self.clicked.emit(self.id)
        super(ClickableLabel, self).mousePressEvent(event)

        test = QTableWidgetItem('gg')
        test.setTextAlignment(Qt.AlignCenter)


class CenteredTableItem(QTableWidgetItem):

    def __init__(self, parent=None):
        QTableWidgetItem.__init__(self, parent)
        self.setTextAlignment(Qt.AlignCenter)


class IncodingThread(QThread):
    record_signal = pyqtSignal(int)
    record_progress = pyqtSignal(str)

    def __init__(self, val, parent=None):
        super(IncodingThread, self).__init__(parent)
        self.out_dir = val

    def run(self):
        t0 = time.time()
        iamges_path = f"./{self.out_dir}/%08d.jpg"
        codecs = "-c:v"
        out_file = "out.mp4"
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-r", "30", "-i", iamges_path, codecs, "h264", out_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in p.stdout:
            if line.startswith("frame="):
                self.record_progress.emit(line[:20])
        # while True:
        #     line = p.stdout.readline()
        #     if not line:
        #         break
        #     print(type(line))

        print(f'Done. ({time.time() - t0:.3f}s)')
        self.record_signal.emit(RECORD_FINISHED)


class VideoThread(QThread):
    changePixmap = pyqtSignal(list)
    frame_signal = pyqtSignal(int)

    def __init__(self, val, parent=None):
        super(VideoThread, self).__init__(parent)
        self.frame_n = 1
        self.instances = {}
        self.class_infos = {}
        self.total_frame = val.total_frame
        # self.instance_cls = val.instance_cls
        self.mode = val.mode
        self.inference_status = val.inference_status
        self.out_dir = val.out_dir

    def convert2qtimage(self, img_path, scale_x, scale_y):
        frame = cv2.imread(img_path)
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(
            rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(scale_x, scale_y, Qt.KeepAspectRatio)
        return p

    def update_imgage(self):
        # print(f"frame_n:{self.frame_n}")
        img_path = self.out_dir + f"/{self.frame_n:08d}.jpg"
        print(img_path)
        images = [self.convert2qtimage(img_path, 1280, 720)]
        if self.frame_n in self.instances:
            instance_data = self.instances[self.frame_n]
            # print(self.class_infos[self.frame_n])
            for i in instance_data:
                images.append(self.convert2qtimage(i, 400, 300))
                # instance_crop = baseQtImage.copy(QRect(*i))
                # images.append(instance_crop)
                # images.append(instance_crop.scaled(
                #     400, 300, Qt.KeepAspectRatio))
        # print(images)
        self.cur_images = images
        self.changePixmap.emit(images)
        self.frame_signal.emit(self.frame_n)

    def run(self):
        while 1:
            if self.mode == PLAYING:
                if self.frame_n < self.total_frame:
                    self.update_imgage()
                    self.frame_n = self.frame_n + 1
                    time.sleep(0.02)
            else:
                time.sleep(0.20)

    def set_frame(self, frame_d):
        self.frame_n = frame_d
        self.update_imgage()

    def stop(self):
        # self._isRunning = False
        self.threadactive = False
        # self.wait()


def classUpdate(item, cls):
    item.setProperty('class', cls)
    item.setStyleSheet('')


class InferenceThread(QThread):
    inference_signal = pyqtSignal(int)
    frame_info_signal = pyqtSignal(object)
    classes_signal = pyqtSignal(list)

    def __init__(self, val, parent=None):
        super(InferenceThread, self).__init__(parent)
        self.model = val.model
        # self.total_frame = val.total_frame
        self.dataset = val.dataset
        self.out_dir = val.out_dir
        self.threshold = val.threshold

    def run(self):
        print(self.out_dir)
        for frame_info in detect(self.model, self.dataset, self.out_dir, self.threshold):
            # print(frame_info)
            # if type(frame_info) == dict:
            #     if (frame_info['frame'] == 1):
            #         self.inference_signal.emit(INFERENCE_INITIALIZED)
            #     self.frame_info_signal.emit(
            #         [frame_info['frame'], frame_info['bboxes']])
            if type(frame_info) == dict:
                if (frame_info['frame'] == 1):
                    self.inference_signal.emit(INFERENCE_INITIALIZED)
                self.frame_info_signal.emit(frame_info)
            elif type(frame_info) == list:
                self.classes_signal.emit(frame_info)
        self.inference_signal.emit(INFERENCE_FINISHED)

    def stop(self):
        self._isRunning = False


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('main.ui', self)
        # self.setStyleSheet(open('stylesheet.css', 'r').read())

        self.imgLabel_0 = self.findChild(QLabel, 'imgLabel_0')
        self.imgLabel_6 = self.findChild(QLabel, 'imgLabel_6')
        self.imgLabel_1 = ClickableLabel(
            self.findChild(QLabel, 'imgLabel_1'), 1)
        self.imgLabel_2 = ClickableLabel(
            self.findChild(QLabel, 'imgLabel_2'), 2)
        self.imgLabel_3 = ClickableLabel(
            self.findChild(QLabel, 'imgLabel_3'), 3)
        self.imgLabel_4 = ClickableLabel(
            self.findChild(QLabel, 'imgLabel_4'), 4)
        self.imgLabel_5 = ClickableLabel(
            self.findChild(QLabel, 'imgLabel_5'), 5)

        self.inference_progress = self.findChild(QLabel, 'inference_progress')
        self.frame_progress = self.findChild(QLabel, 'frame_progress')
        self.total_instances = self.findChild(QLabel, 'total_instances')
        self.current_info_title = self.findChild(QLabel, 'current_info_title')
        self.record_status_title = self.findChild(
            QLabel, 'record_status_title')
        self.record_status_label = self.findChild(
            QLabel, 'record_status_label')

        self.STOP = self.findChild(QPushButton, 'STOP')
        self.RECORD = self.findChild(QPushButton, 'RECORD')
        self.RECORD.setText('결과 저장')
        self.PLAY = self.findChild(QPushButton, 'PLAY')
        self.REINITIATE = self.findChild(QPushButton, 'REINITIATE')
        self.REINITIATE.setText('초기화')

        # self.pickmodel = self.findChild(QPushButton, 'pickmodel')
        self.pickmodel2 = self.findChild(QComboBox, 'pickModel2')
        self.pickmodel2.addItem("항만감시")
        self.pickmodel2.addItem("수변감시")
        self.pickmodel2.addItem("태양광")
        self.pickmodel2.addItem("불법비행")
        self.pickmodel2.addItem("선용품")
        self.pickmodel2.addItem("시설물")
        # self.pickmodel2.activated[str].connect(self.style_choice)
        self.pickmodel2.activated[str].connect(lambda value: print(value))

        self.pickvideo = self.findChild(QPushButton, 'pickvideo')
        self.pickvideo.setText('동영상 입력')

        self.startInference = self.findChild(QPushButton, 'startInference')
        self.startInference.setText('객체 검출 시작')
        self.startInference.setProperty('class', 'warning')

        self.slider = self.findChild(QSlider, 'slider')

        self.pbar = self.findChild(QProgressBar, 'pbar')
        self.offset_input = self.findChild(QTextEdit, 'offset_input')
        self.offset_label = self.findChild(QLabel, 'offset_label')
        self.offset_label.setText('오프셋 프레임')
        self.threshold_input = self.findChild(QTextEdit, 'threshold_input')
        self.threshold_label = self.findChild(QLabel, 'threshold_label')
        self.threshold_label.setText('쓰레쉬홀드')
        self.actiondark_blue = self.findChild(QAction, 'actiondark_blue')
        self.actionlight_red = self.findChild(QAction, 'actionlight_red')
        self.instance_counter = self.findChild(
            QTableWidget, 'instance_counter')

        self.instance_counter.setHorizontalHeaderLabels(
            ["검출 수", "평균 검출률"])
        self.instance_counter.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.instance_counter.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        vertical_header = []
        for i in kor_cls.keys():
            vertical_header.append(kor_cls[i]['kor'])
        self.instance_counter.setVerticalHeaderLabels(vertical_header)
        # self.actiondark_blue.triggered.connect(self.changeStyle('dark_blue.xml'))
        self.actiondark_blue.triggered.connect(self.dark_blue)
        self.actionlight_red.triggered.connect(self.light_red)
        apply_stylesheet(self, list_themes()[1])
        # 1 14 15
        self.pbar.setProperty('class', 'warning')
        self.pbar.setStyleSheet(
            "QProgressBar::chunk {background-color: #00bcd4;width: 5px;margin: 1px;}")
        self.slider.setStyleSheet(
            "QSlider::handler {border-radius: 3px; border: 4px; width: 6px;}")
        self.show()

    def dark_blue(self):
        apply_stylesheet(self, 'dark_blue.xml')

    def light_red(self):
        apply_stylesheet(self, 'light_red.xml')


class MainWindow(QMainWindow):
    # class constructor

    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.re_ini()

        # self.ui.rbtn1.clicked.connect(self.radio_clicked)
        # self.ui.rbtn2.clicked.connect(self.radio_clicked)
        # self.ui.rbtn3.clicked.connect(self.radio_clicked)
        # self.ui.imgLabel_1.mousePressEvent = self.instance_clicked

    def re_ini(self):
        self.mode = STOPPED
        self.inference_status = UNINITIALIZED
        self.total_frame = 1
        self.frame_n = 1
        self.instances = {}
        self.frame_infos = {}
        self.threshold = 0.30

        if hasattr(self, 'th'):
            self.th.mode = STOPPED
            self.th.stop()
        if hasattr(self, 'inference'):
            # self.inference.frame_info_signal.disconnect(self.addFrameInfo)
            # self.inference.inference_signal.disconnect(
            #     self.updateInferenceStatus)
            self.inference.stop()
        self.ui = Ui()

        # default offset
        self.offset = 10000

        self.out_dir = '../inference/output/video'
        self.model = '/home/edgar/Desktop/yolov5/best.pt'
        self.inputVideo = '/home/edgar/Desktop/yolov5/Video6.mp4'
        self.magni = 1
        self.stats = {'total_instances': 0}

        self.th = VideoThread(self)
        self.th.start()
        self.ui.STOP.clicked.connect(self.StopClicked)
        self.ui.slider.valueChanged.connect(self.setFrame)
        self.ui.PLAY.clicked.connect(self.PlayClicked)
        self.ui.REINITIATE.clicked.connect(self.ReInitiate)
        self.ui.RECORD.clicked.connect(self.RecordClicked)
        # self.ui.pickmodel.clicked.connect(self.setModel)
        self.ui.pickvideo.clicked.connect(self.setInputVideo)
        self.ui.startInference.clicked.connect(self.startInference)
        self.th.changePixmap.connect(self.setImage)

        self.th.frame_signal.connect(self.updateSlideer)
        self.ui.imgLabel_1.clicked.connect(self.instance_clicked)
        self.ui.imgLabel_2.clicked.connect(self.instance_clicked)
        self.ui.imgLabel_3.clicked.connect(self.instance_clicked)
        self.ui.imgLabel_4.clicked.connect(self.instance_clicked)
        self.ui.imgLabel_5.clicked.connect(self.instance_clicked)
        self.ui.offset_input.textChanged.connect(self.offset_input)
        self.ui.threshold_input.textChanged.connect(self.threshold_input)
        # self.ui.imgLabel_0.setText('ddd')

    def startInference(self):
        click_sound.play()
        if self.inference_status == UNINITIALIZED:
            self.dataset = LoadImages(
                self.inputVideo, self.offset, img_size=640, stride=32)
            self.total_frame = self.dataset.nframes - self.offset
            self.th.total_frame = self.total_frame
            self.ui.slider.setMaximum(self.total_frame)
            self.inference = InferenceThread(self)
            self.inference.inference_signal.connect(self.updateInferenceStatus)
            self.inference.frame_info_signal.connect(self.addFrameInfo)
            self.inference.classes_signal.connect(self.updateCls)
            self.inference_status = INFERENCE_INITIALIZED
            self.inference.start()

    def ReInitiate(self):
        self.StopClicked()
        self.re_ini()

    @ pyqtSlot(object)
    def addFrameInfo(self, frame_info):
        # self.th.frame_infos[cur_frame] = frame_info[1]
        if self.inference_status != UNINITIALIZED:
            cur_frame = frame_info['frame']
            progress = f"{cur_frame/self.total_frame:.2%}"
            self.ui.frame_progress.setText(
                f'<b style="font-dize: 20px">{progress}</b>  {cur_frame}/{self.total_frame}')
            num_instances = frame_info['total_instance']
            self.ui.pbar.setValue(int(float(progress[:-1])))
            if num_instances:
                # self.th.frame_infos[cur_frame] = frame_info['instances']
                self.th.instances[cur_frame] = frame_info['instances']
                classes_info = frame_info['classes']
                self.frame_infos[cur_frame] = frame_info['classes']
                self.stats['total_instances'] = self.stats['total_instances'] + \
                    num_instances
                total_instances = self.stats['total_instances']
                s = f'Total_Instances: {total_instances}\n'
                # print(frame_info)
                for i in classes_info:
                    cls_cnt, avg_cnf = classes_info[i]
                    self.stats[i] = [self.stats[i][0] +
                                     cls_cnt, self.stats[i][1] + cls_cnt * avg_cnf]
                    s = s + \
                        f"{i}_cnt : {self.stats[i][0]}, avg_conf: {self.stats[i][1] / self.stats[i][0]:.2%}\n"
                self.ui.total_instances.setText(s)

        # print(self.stats)

    @ pyqtSlot(list)
    def updateCls(self, cls):
        self.clsses = cls
        for i in cls:
            self.stats[i] = [0, 0]

    def setModel(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "Checkpoint Files(*.pt | *.pth)", options=options)
        if fileName:
            self.model = fileName
            print(self.model)
            self.ui.pickmodel.setText(fileName.split('/')[-1])

    def setInputVideo(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "MP4(*.mp4);;AVI(*.avi);;MOV(*.mov)", options=options)
        if fileName:
            self.inputVideo = fileName
            print(self.inputVideo)
            showing_filename = fileName.split('/')[-1]
            self.core_videoname = showing_filename[:-4]
            self.out_dir = os.path.join(self.out_dir, self.core_videoname)
            self.th.out_dir = self.out_dir
            self.ui.pickvideo.setText(fileName.split('/')[-1])

    def instance_clicked(self, value):
        print(value)
        self.magni = value
        if (self.mode == STOPPED and self.inference_status != UNINITIALIZED) and value < len(self.th.cur_images):
            self.ui.imgLabel_6.setPixmap(
                QPixmap.fromImage(self.th.cur_images[value]))

    @ pyqtSlot(int)
    def updateSlideer(self, val):
        self.ui.slider.setValue(val)
        if val in self.frame_infos:
            current_frame_data = self.frame_infos[val]
            for i in current_frame_data:
                index_of_class = kor_cls[i]['index']
                cnt = current_frame_data[i][0]
                avg_cnf = f'{current_frame_data[i][1]:.2%}'
                self.ui.instance_counter.setItem(
                    index_of_class, 0, CenteredTableItem(str(cnt)))
                self.ui.instance_counter.setItem(
                    index_of_class, 1, CenteredTableItem(avg_cnf))

        else:
            text = 'No Instance'

    @ pyqtSlot(int)
    def updateInferenceStatus(self, status):
        print('test')
        if status == INFERENCE_INITIALIZED:
            print('INFERENCE_INITIALIZED')
            classUpdate(self.ui.PLAY, 'warning')
            self.inference_status = INFERENCE_INITIALIZED
            classUpdate(self.ui.inference_progress, 'warning')
            self.ui.inference_progress.setText(
                'Inference Intialized, PLAYABLE')
            # self.ui.inference_progress.setText(
            #     '<b style="color: blue">Inference Intialized, PLAYABLE</b>')
            classUpdate(self.ui.startInference, None)

        elif status == INFERENCE_FINISHED:
            print('INFERENCE_FINISHED')
            classUpdate(self.ui.PLAY, 'success')
            self.inference_status = INFERENCE_FINISHED
            self.th.inference_status = INFERENCE_FINISHED
            classUpdate(self.ui.inference_progress, 'success')
            self.ui.inference_progress.setText(
                f'Inference Finished. Total {self.total_frame}')
            # self.ui.inference_progress.setText(
            #     f'<b style="color: green">Inference Finished. Total {self.total_frame} f</b>')
            classUpdate(self.ui.RECORD, 'warning')
        elif status == RECORD_FINISHED:
            classUpdate(self.ui.RECORD, 'success')

    def setFrame(self):
        if self.mode == STOPPED and self.inference_status != UNINITIALIZED:
            self.th.set_frame(self.ui.slider.value())

    @ pyqtSlot(list)
    def setImage(self, images):
        for i in range(6):
            if i < len(images):
                eval(
                    f"self.ui.imgLabel_{i}.setPixmap(QPixmap.fromImage(images[{i}]))")
            else:
                eval(f"self.ui.imgLabel_{i}.setText('Instance {i}')")
        if self.magni < len(images):
            eval(
                f"self.ui.imgLabel_6.setPixmap(QPixmap.fromImage(images[{self.magni}]))")

    def offset_input(self):
        if self.ui.offset_input.toPlainText().isdigit():
            self.offset = int(self.ui.offset_input.toPlainText())
            print(self.offset)
    
    def threshold_input(self):        
        if self.ui.threshold_input.toPlainText().isdigit():
            self.threshold = int(self.ui.threshold_input.toPlainText()) / 100
            print(self.threshold)

    def PlayClicked(self):
        if self.mode != PLAYING and self.inference_status != UNINITIALIZED:
            self.mode = PLAYING
            self.th.mode = PLAYING
        click_sound.play()

    def StopClicked(self):
        self.th.mode = STOPPED
        self.mode = STOPPED
        click_sound.play()

    def RecordClicked(self):
        if self.inference_status == INFERENCE_FINISHED:
            self.incoding_th = IncodingThread(self.out_dir)
            self.incoding_th.start()
            self.incoding_th.record_signal.connect(self.updateInferenceStatus)
            self.incoding_th.record_progress.connect(self.recordStatusUpdate)
        click_sound.play()

    @ pyqtSlot(str)
    def recordStatusUpdate(self, progress_text):
        incoded_frames = int(progress_text[6:12])
        s = f'{incoded_frames/2639:.2%} / '
        self.ui.record_status_label.setText(s + progress_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    click_wav = 'modern_short.wav'
    click_sound = QtMultimedia.QSoundEffect()
    click_sound.setSource(QUrl.fromLocalFile(click_wav))
    click_sound.setVolume(50)
    mainWindow = MainWindow()

    # mainWindow.show()

    sys.exit(app.exec_())
