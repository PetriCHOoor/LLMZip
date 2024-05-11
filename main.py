#import sys
#from PySide6.QtWidgets import QApplication, QMainWindow
#from PySide6.QtCore import QFile
#from zip_ui import Ui_LLMzip
#from model_build import build_llama

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,QThread,
    QSize, QTime, QUrl, Qt, QDir, Signal)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout,
    QLabel, QLayout, QLineEdit, QPlainTextEdit,QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QSlider,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget, QFileDialog,QMessageBox)

from model_build import build_llama, build_GPT2

import sys
import time
import gc
import json
import torch
from tqdm import tqdm
import zlib
import os
class CompressThread(QThread):

    compress_signal = Signal(int)

    def __init__(self, model, file_path, memory, device, save_path):
        super().__init__()
        self.model = model
        self.file_path = file_path
        self.memory = memory
        self.device = device
        self.save_path = save_path

    def run(self):

        self.compress(self.model,
                      self.file_path,
                      self.memory,
                      self.device,
                      self.save_path)

        self.compress_signal.emit(100)

    @staticmethod
    def get_unique_filename(base_path):

        filename, file_extension = os.path.splitext(base_path)
        counter = 1
        new_path = base_path
        while os.path.exists(new_path):
            new_path = f"{filename}({counter}){file_extension}"
            counter += 1

        return new_path

    def compress_zlib(self,
                  ranks,
                  file_name,
                  save_path: str):
        byte_data = bytearray()
        for num in ranks:
            byte_data.extend(num.to_bytes(2, 'little'))  # 每个整数转换为2字节的字节串

        compressed_data = zlib.compress(byte_data)

        base_path = save_path + '/' + os.path.basename(file_name[0:-4]) + '.bin'
        unique_path = self.get_unique_filename(base_path)

        with open(unique_path, 'wb') as file:
                file.write(compressed_data)


    def compress(self, model, file_name, memory, device, save_path):

        window = QWidget()
        temperature: float = 0.6

        version = model.version

        models = ['GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'GPT2-XL', 'LLaMA2-7B']

        if model.__class__.__name__ == 'LLaMA':

            with open(file_name, 'r', encoding='utf-8') as file:
                text = [file.read()]

            tokens = torch.tensor(model.tokenizer.encode(text, out_type=int, add_bos=False, add_eos=False)).to(device)

            if not tokens.shape[1] > memory:
                QMessageBox.warning(window, "Memory error", f"memory must be smaller than the length of tokens!(memory:{memory},length:{tokens.shape[1]}")

            total = len(tokens[0].tolist()) - memory
            sum = 1

            cur_iterator = tqdm(range(0, total),desc=f"Compressing with {models[version]}")

            ranks = [version, memory]

            for i in range(memory):
                    ranks.append(tokens[0][i].item())

            for cur_pos in cur_iterator:

                with torch.no_grad():
                        logits = model.model.forward(tokens[:, cur_pos:cur_pos + memory], 0)

                probs = torch.softmax(logits[-1, :] / temperature, dim=-1)

                prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)

                token_real_value = tokens[:, cur_pos + memory: cur_pos + memory + 1]

                rank = torch.where(prob_idx == token_real_value)[1].item()

                ranks.append(rank)
                if sum % 20 == 0 and (sum / total) != 1:
                    self.compress_signal.emit((sum / total) * 100)
                sum = sum+1

            self.compress_zlib(ranks, file_name, save_path)

        if model.__class__.__name__ == 'GPT2':

            with open(file_name, 'r', encoding='utf-8') as file:
                text = file.read()

            tokens = torch.tensor([model.tokenizer.encode(text)]).to(device)

            if not tokens.shape[1] > memory:
                    QMessageBox.warning(window, "Memory error",f"memory must be smaller than the length of tokens!(memory:{memory},length:{tokens.shape[1]}")

            ranks = [version, memory]

            model.model.eval()

            for i in range(memory):
                    ranks.append(tokens[0][i].item())

            total = len(tokens[0].tolist()) - memory
            sum = 1

            cur_iterator = tqdm(range(0, total),desc=f'Compressing with {models[version]}')

            for cur_pos in cur_iterator:
                with torch.no_grad():
                        logits = model.model(tokens[:, cur_pos:cur_pos + memory])[0][0, -1, :]

                token_real_value = tokens[:, cur_pos + memory:cur_pos + memory + 1]

                probs = torch.softmax(logits[:] / temperature, dim=-1)
                prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)
                rank = torch.where(prob_idx == token_real_value)[1].item()

                ranks.append(rank)
                if sum % 20 == 0 and (sum / total) != 1:
                    self.compress_signal.emit((sum / total) * 100)
                sum = sum+1

            self.compress_zlib(ranks, file_name, save_path)

class DecompressThread(QThread):

    decompress_signal = Signal(int)
    check_signal = Signal(bool)

    def __init__(self, save_path, model, ranks, version, memory):
        super().__init__()
        self.save_path = save_path
        self.model = model
        self.ranks = ranks
        self.version = version
        self.memory = memory

    def run(self):

        self.decompress(self.save_path,
                        self.model,
                        self.ranks,
                        self.version,
                        self.memory)

        self.decompress_signal.emit(100)

    @staticmethod
    def get_unique_filename(base_path):

        filename, file_extension = os.path.splitext(base_path)
        counter = 1
        new_path = base_path
        while os.path.exists(new_path):
            new_path = f"{filename}({counter}){file_extension}"
            counter += 1

        return new_path


    def decompress(self,
                   save_path,
                   model,
                   ranks,
                   version,
                   memory,
                   ):

        window = QWidget()
        temperature: float = 0.6

        models = ['LLaMA2-7B','GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'GPT2-XL']

        tokens = []
        for i in range(memory):
            tokens.append(ranks[i + 2])

        tokens = torch.tensor([tokens])
        total = len(ranks) - memory - 2
        cur_iterator = tqdm(range(0, total), desc=f"Decompressing with {models[version]}")

        # LLaMA model decompress
        if version == 0:
            sum = 1
            for cur_pos in cur_iterator:
                with torch.no_grad():
                    logits = model.model.forward(tokens[:, cur_pos:cur_pos + memory], 0)

                probs = torch.softmax(logits[-1, :] / temperature, dim=-1)

                prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)

                token = prob_idx[ranks[memory + cur_pos + 2]].unsqueeze(0).unsqueeze(0)

                tokens = torch.cat((tokens, token), dim=1)

                if sum % 20 == 0 and (sum / total) != 1:
                    self.decompress_signal.emit((sum / total) * 100)
                sum = sum+1

        # GPT2 models decompress
        if version == 1 or version == 2 or version == 3 or version == 4:
            model.model.eval()
            sum = 1
            for cur_pos in cur_iterator:
                with torch.no_grad():
                    logits = model.model(tokens[:, cur_pos:cur_pos + memory])[0][0, -1, :]

                probs = torch.softmax(logits[:] / temperature, dim=-1)
                prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)

                token = prob_idx[ranks[memory + cur_pos + 2]].unsqueeze(0).unsqueeze(0)

                tokens = torch.cat((tokens, token), dim=1)

                if sum % 20 == 0 and (sum / total) != 1:
                    self.decompress_signal.emit((sum / total) * 100)
                sum = sum+1

        # decode tokens
        decompressed_text = model.tokenizer.decode(tokens[0].tolist())

        base_path = save_path + '/' + 'decompressed.txt'
        unique_path = self.get_unique_filename(base_path)

        with open(unique_path, 'w', encoding='utf-8') as file:
            file.write(decompressed_text)


class Ui_LLMzip(object):
    def __init__(self):
        super().__init__()

        self.return_path = None

        self.build_llama = build_llama
        self.build_GPT2 = build_GPT2

        self.llama_model_path : str = ''
        self.gpts_model_path  : str = ''
        self.gptm_model_path  : str = ''
        self.gptl_model_path  : str = ''
        self.gptxl_model_path : str = ''

        self.device : str = ''
        self.memory : int = -1

        self.save_path_cprs  : str = ''
        self.save_path_dcprs : str = ''

        self.file_path_cprs  : str = ''
        self.file_path_dcprs : str = ''

        self.gpts_model = None
        self.gptm_model = None
        self.gptl_model = None
        self.gptxl_model = None
        self.llama_model = None

        self.model = None
        self.model_version : int = -1

    def setupUi(self, LLMzip):
        if not LLMzip.objectName():
            LLMzip.setObjectName(u"LLMzip")
        LLMzip.setEnabled(True)
        LLMzip.resize(811, 641)
        LLMzip.setFocusPolicy(Qt.NoFocus)
        LLMzip.setContextMenuPolicy(Qt.ActionsContextMenu)
        LLMzip.setAcceptDrops(False)
        self.verticalLayoutWidget = QWidget(LLMzip)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 791, 621))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.filePathLabelCPRS_2 = QTabWidget(self.verticalLayoutWidget)
        self.filePathLabelCPRS_2.setObjectName(u"filePathLabelCPRS_2")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filePathLabelCPRS_2.sizePolicy().hasHeightForWidth())
        self.filePathLabelCPRS_2.setSizePolicy(sizePolicy)
        font = QFont()
        font.setFamilies([u"\u9ed1\u4f53"])
        font.setPointSize(11)
        font.setBold(True)
        font.setUnderline(False)
        font.setKerning(True)
        font.setStyleStrategy(QFont.PreferDefault)
        self.filePathLabelCPRS_2.setFont(font)
        self.filePathLabelCPRS_2.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.filePathLabelCPRS_2.setLayoutDirection(Qt.LeftToRight)
        self.filePathLabelCPRS_2.setAutoFillBackground(False)
        self.filePathLabelCPRS_2.setStyleSheet(u"QTabBar::tab {\n"
"            width: 80px; /* \u8bbe\u7f6e\u6807\u7b7e\u7684\u5bbd\u5ea6 */\n"
"            height: 105px; /* \u8bbe\u7f6e\u6807\u7b7e\u7684\u9ad8\u5ea6 */\n"
"            qproperty-alignment: 'AlignCenter'; /* \u8bbe\u7f6e\u6587\u5b57\u6c34\u5e73\u5c45\u4e2d */\n"
"			transform: rotate(90deg);\n"
"        }\n"
"        QTabBar::tab:selected {\n"
"            background: rgb(206, 206, 206); /* \u9009\u4e2d\u6807\u7b7e\u7684\u80cc\u666f\u8272 */\n"
"        }")
        self.filePathLabelCPRS_2.setTabPosition(QTabWidget.West)
        self.filePathLabelCPRS_2.setTabShape(QTabWidget.Rounded)
        self.filePathLabelCPRS_2.setIconSize(QSize(11, 16))
        self.filePathLabelCPRS_2.setElideMode(Qt.ElideLeft)
        self.filePathLabelCPRS_2.setDocumentMode(False)
        self.filePathLabelCPRS_2.setTabsClosable(False)
        self.filePathLabelCPRS_2.setMovable(False)
        self.filePathLabelCPRS_2.setTabBarAutoHide(False)
        self.compress = QWidget()
        self.compress.setObjectName(u"compress")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.compress.sizePolicy().hasHeightForWidth())
        self.compress.setSizePolicy(sizePolicy1)
        self.gridLayoutWidget = QWidget(self.compress)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(150, 20, 411, 421))
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(10)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setSizeConstraint(QLayout.SetNoConstraint)
        self.memoryLabel = QLabel(self.gridLayoutWidget)
        self.memoryLabel.setObjectName(u"memoryLabel")
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(False)
        self.memoryLabel.setFont(font1)

        self.horizontalLayout_5.addWidget(self.memoryLabel)

        self.memorySlider = QSlider(self.gridLayoutWidget)
        self.memorySlider.setObjectName(u"memorySlider")
        self.memorySlider.setMinimumSize(QSize(275, 20))
        self.memorySlider.setMaximumSize(QSize(50, 20))
        self.memorySlider.setCursor(QCursor(Qt.ArrowCursor))
        self.memorySlider.setFocusPolicy(Qt.TabFocus)
        self.memorySlider.setMinimum(1)
        self.memorySlider.setMaximum(256)
        self.memorySlider.setOrientation(Qt.Horizontal)
        self.memorySlider.setTickPosition(QSlider.TicksBelow)
        self.memorySlider.setTickInterval(10)

        self.horizontalLayout_5.addWidget(self.memorySlider)

        self.memoryInput = QLineEdit(self.gridLayoutWidget)
        self.memoryInput.setObjectName(u"memoryInput")
        self.memoryInput.setMaximumSize(QSize(65, 16777215))
        font2 = QFont()
        font2.setFamilies([u"Consolas"])
        font2.setPointSize(12)
        self.memoryInput.setFont(font2)

        self.horizontalLayout_5.addWidget(self.memoryInput)


        self.gridLayout.addLayout(self.horizontalLayout_5, 4, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.filePathLabelCPRS = QLabel(self.gridLayoutWidget)
        self.filePathLabelCPRS.setObjectName(u"filePathLabelCPRS")
        self.filePathLabelCPRS.setFont(font1)

        self.horizontalLayout_3.addWidget(self.filePathLabelCPRS)

        self.filePathCPRS = QLineEdit(self.gridLayoutWidget)
        self.filePathCPRS.setObjectName(u"filePathCPRS")
        font3 = QFont()
        font3.setFamilies([u"Microsoft YaHei UI"])
        font3.setPointSize(12)
        self.filePathCPRS.setFont(font3)

        self.horizontalLayout_3.addWidget(self.filePathCPRS)

        self.filePathSelectCPRS = QPushButton(self.gridLayoutWidget)
        self.filePathSelectCPRS.setObjectName(u"filePathSelectCPRS")
        self.filePathSelectCPRS.setMaximumSize(QSize(30, 30))

        self.horizontalLayout_3.addWidget(self.filePathSelectCPRS)


        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(5)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.savePathLabelCPRS = QLabel(self.gridLayoutWidget)
        self.savePathLabelCPRS.setObjectName(u"savePathLabelCPRS")
        self.savePathLabelCPRS.setFont(font1)

        self.horizontalLayout_4.addWidget(self.savePathLabelCPRS)

        self.savePathCPRS = QLineEdit(self.gridLayoutWidget)
        self.savePathCPRS.setObjectName(u"savePathCPRS")
        self.savePathCPRS.setFont(font3)

        self.horizontalLayout_4.addWidget(self.savePathCPRS)

        self.savePathSelectCPRS = QPushButton(self.gridLayoutWidget)
        self.savePathSelectCPRS.setObjectName(u"savePathSelectCPRS")
        self.savePathSelectCPRS.setMaximumSize(QSize(30, 30))

        self.horizontalLayout_4.addWidget(self.savePathSelectCPRS)


        self.gridLayout.addLayout(self.horizontalLayout_4, 3, 0, 1, 1)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.startCompressing = QPushButton(self.gridLayoutWidget)
        self.startCompressing.setObjectName(u"startCompressing")
        self.startCompressing.setEnabled(True)
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Ignored)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.startCompressing.sizePolicy().hasHeightForWidth())
        self.startCompressing.setSizePolicy(sizePolicy2)
        self.startCompressing.setMinimumSize(QSize(0, 0))
        self.startCompressing.setMaximumSize(QSize(100, 40))
        self.startCompressing.setBaseSize(QSize(0, 0))
        self.startCompressing.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_6.addWidget(self.startCompressing)


        self.gridLayout.addLayout(self.horizontalLayout_6, 5, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.modelVersionLabel = QLabel(self.gridLayoutWidget)
        self.modelVersionLabel.setObjectName(u"modelVersionLabel")
        self.modelVersionLabel.setFont(font1)

        self.horizontalLayout.addWidget(self.modelVersionLabel)

        self.modelVersion = QComboBox(self.gridLayoutWidget)
        self.modelVersion.addItem("")
        self.modelVersion.addItem("")
        self.modelVersion.addItem("")
        self.modelVersion.addItem("")
        self.modelVersion.addItem("")
        self.modelVersion.addItem("")
        self.modelVersion.setObjectName(u"modelVersion")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.modelVersion.sizePolicy().hasHeightForWidth())
        self.modelVersion.setSizePolicy(sizePolicy3)
        font4 = QFont()
        font4.setPointSize(12)
        self.modelVersion.setFont(font4)
        self.modelVersion.setFocusPolicy(Qt.WheelFocus)
        self.modelVersion.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.modelVersion.setLayoutDirection(Qt.LeftToRight)
        self.modelVersion.setAutoFillBackground(False)

        self.horizontalLayout.addWidget(self.modelVersion)


        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)

        self.titleCompress = QLabel(self.compress)
        self.titleCompress.setObjectName(u"titleCompress")
        self.titleCompress.setGeometry(QRect(10, 0, 141, 61))
        font5 = QFont()
        font5.setFamilies([u"Consolas"])
        font5.setPointSize(24)
        self.titleCompress.setFont(font5)
        self.filePathLabelCPRS_2.addTab(self.compress, "")
        self.decompress = QWidget()
        self.decompress.setObjectName(u"decompress")
        self.gridLayoutWidget_2 = QWidget(self.decompress)
        self.gridLayoutWidget_2.setObjectName(u"gridLayoutWidget_2")
        self.gridLayoutWidget_2.setGeometry(QRect(80, 70, 541, 331))
        self.gridLayout_2 = QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setSpacing(5)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.savePathLabelDCPRS = QLabel(self.gridLayoutWidget_2)
        self.savePathLabelDCPRS.setObjectName(u"savePathLabelDCPRS")
        self.savePathLabelDCPRS.setFont(font1)

        self.horizontalLayout_8.addWidget(self.savePathLabelDCPRS)

        self.savePathDCPRS = QLineEdit(self.gridLayoutWidget_2)
        self.savePathDCPRS.setObjectName(u"savePathDCPRS")
        self.savePathDCPRS.setFont(font4)

        self.horizontalLayout_8.addWidget(self.savePathDCPRS)

        self.savePathSelectDCPRS = QPushButton(self.gridLayoutWidget_2)
        self.savePathSelectDCPRS.setObjectName(u"savePathSelectDCPRS")
        self.savePathSelectDCPRS.setMaximumSize(QSize(30, 30))

        self.horizontalLayout_8.addWidget(self.savePathSelectDCPRS)


        self.gridLayout_2.addLayout(self.horizontalLayout_8, 1, 0, 1, 1)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(5)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.filePathLabelDCPRS = QLabel(self.gridLayoutWidget_2)
        self.filePathLabelDCPRS.setObjectName(u"filePathLabelDCPRS")
        self.filePathLabelDCPRS.setFont(font1)

        self.horizontalLayout_7.addWidget(self.filePathLabelDCPRS)

        self.filePathDCPRS = QLineEdit(self.gridLayoutWidget_2)
        self.filePathDCPRS.setObjectName(u"filePathDCPRS")
        self.filePathDCPRS.setFont(font4)

        self.horizontalLayout_7.addWidget(self.filePathDCPRS)

        self.filePathSelectDCPRS = QPushButton(self.gridLayoutWidget_2)
        self.filePathSelectDCPRS.setObjectName(u"filePathSelectDCPRS")
        self.filePathSelectDCPRS.setMaximumSize(QSize(30, 30))

        self.horizontalLayout_7.addWidget(self.filePathSelectDCPRS)


        self.gridLayout_2.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.startDecompressing = QPushButton(self.gridLayoutWidget_2)
        self.startDecompressing.setObjectName(u"startDecompressing")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.startDecompressing.sizePolicy().hasHeightForWidth())
        self.startDecompressing.setSizePolicy(sizePolicy4)
        self.startDecompressing.setMinimumSize(QSize(100, 40))
        self.startDecompressing.setMaximumSize(QSize(100, 40))
        self.startDecompressing.setSizeIncrement(QSize(0, 0))
        self.startDecompressing.setBaseSize(QSize(0, 0))

        self.horizontalLayout_9.addWidget(self.startDecompressing)


        self.gridLayout_2.addLayout(self.horizontalLayout_9, 3, 0, 1, 1)

        self.titleDecompress = QLabel(self.decompress)
        self.titleDecompress.setObjectName(u"titleDecompress")
        self.titleDecompress.setGeometry(QRect(10, 0, 211, 61))
        self.titleDecompress.setFont(font5)
        self.titleDecompress.setTabletTracking(False)
        self.titleDecompress.setTextFormat(Qt.AutoText)
        self.titleDecompress.setWordWrap(False)
        self.filePathLabelCPRS_2.addTab(self.decompress, "")
        self.settings = QWidget()
        self.settings.setObjectName(u"settings")
        self.gridLayoutWidget_3 = QWidget(self.settings)
        self.gridLayoutWidget_3.setObjectName(u"gridLayoutWidget_3")
        self.gridLayoutWidget_3.setGeometry(QRect(50, 60, 601, 341))
        self.gridLayout_3 = QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setHorizontalSpacing(6)
        self.gridLayout_3.setVerticalSpacing(0)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.llamaLabel = QLabel(self.gridLayoutWidget_3)
        self.llamaLabel.setObjectName(u"llamaLabel")
        font6 = QFont()
        font6.setPointSize(12)
        font6.setBold(False)
        font6.setKerning(True)
        self.llamaLabel.setFont(font6)

        self.gridLayout_3.addWidget(self.llamaLabel, 0, 0, 1, 1)

        self.gptLargeLabel = QLabel(self.gridLayoutWidget_3)
        self.gptLargeLabel.setObjectName(u"gptLargeLabel")
        self.gptLargeLabel.setFont(font4)

        self.gridLayout_3.addWidget(self.gptLargeLabel, 3, 0, 1, 1)

        self.gptMediumLabel = QLabel(self.gridLayoutWidget_3)
        self.gptMediumLabel.setObjectName(u"gptMediumLabel")
        self.gptMediumLabel.setFont(font4)

        self.gridLayout_3.addWidget(self.gptMediumLabel, 2, 0, 1, 1)

        self.gptSmallLabel = QLabel(self.gridLayoutWidget_3)
        self.gptSmallLabel.setObjectName(u"gptSmallLabel")
        font7 = QFont()
        font7.setPointSize(12)
        font7.setKerning(True)
        self.gptSmallLabel.setFont(font7)

        self.gridLayout_3.addWidget(self.gptSmallLabel, 1, 0, 1, 1)

        self.gptlPath = QLineEdit(self.gridLayoutWidget_3)
        self.gptlPath.setObjectName(u"gptlPath")
        self.gptlPath.setFont(font4)

        self.gridLayout_3.addWidget(self.gptlPath, 3, 1, 1, 1)

        self.loadGPTM = QPushButton(self.gridLayoutWidget_3)
        self.loadGPTM.setObjectName(u"loadGPTM")
        self.loadGPTM.setMaximumSize(QSize(50, 30))
        self.loadGPTM.setFont(font4)

        self.gridLayout_3.addWidget(self.loadGPTM, 2, 3, 1, 1)

        self.deviceLabel = QLabel(self.gridLayoutWidget_3)
        self.deviceLabel.setObjectName(u"deviceLabel")
        self.deviceLabel.setMaximumSize(QSize(200, 30))
        self.deviceLabel.setFont(font4)

        self.gridLayout_3.addWidget(self.deviceLabel, 6, 0, 1, 1)

        self.gptmPath = QLineEdit(self.gridLayoutWidget_3)
        self.gptmPath.setObjectName(u"gptmPath")
        self.gptmPath.setFont(font4)

        self.gridLayout_3.addWidget(self.gptmPath, 2, 1, 1, 1)

        self.releaseGPTXL = QPushButton(self.gridLayoutWidget_3)
        self.releaseGPTXL.setObjectName(u"releaseGPTXL")
        self.releaseGPTXL.setMaximumSize(QSize(50, 30))
        self.releaseGPTXL.setFont(font4)

        self.gridLayout_3.addWidget(self.releaseGPTXL, 4, 4, 1, 1)

        self.gptxlPath = QLineEdit(self.gridLayoutWidget_3)
        self.gptxlPath.setObjectName(u"gptxlPath")
        self.gptxlPath.setFont(font4)

        self.gridLayout_3.addWidget(self.gptxlPath, 4, 1, 1, 1)

        self.releaseGPTM = QPushButton(self.gridLayoutWidget_3)
        self.releaseGPTM.setObjectName(u"releaseGPTM")
        self.releaseGPTM.setMaximumSize(QSize(50, 30))
        self.releaseGPTM.setFont(font4)

        self.gridLayout_3.addWidget(self.releaseGPTM, 2, 4, 1, 1)

        self.loadLLAMA = QPushButton(self.gridLayoutWidget_3)
        self.loadLLAMA.setObjectName(u"loadLLAMA")
        self.loadLLAMA.setMinimumSize(QSize(50, 30))
        self.loadLLAMA.setMaximumSize(QSize(50, 30))
        self.loadLLAMA.setFont(font4)

        self.gridLayout_3.addWidget(self.loadLLAMA, 0, 3, 1, 1)

        self.loadGPTL = QPushButton(self.gridLayoutWidget_3)
        self.loadGPTL.setObjectName(u"loadGPTL")
        self.loadGPTL.setMaximumSize(QSize(50, 30))
        self.loadGPTL.setFont(font4)

        self.gridLayout_3.addWidget(self.loadGPTL, 3, 3, 1, 1)

        self.gptsPath = QLineEdit(self.gridLayoutWidget_3)
        self.gptsPath.setObjectName(u"gptsPath")
        self.gptsPath.setFont(font4)

        self.gridLayout_3.addWidget(self.gptsPath, 1, 1, 1, 1)

        self.releaseLLAMA = QPushButton(self.gridLayoutWidget_3)
        self.releaseLLAMA.setObjectName(u"releaseLLAMA")
        self.releaseLLAMA.setMaximumSize(QSize(50, 30))
        self.releaseLLAMA.setFont(font4)

        self.gridLayout_3.addWidget(self.releaseLLAMA, 0, 4, 1, 1)

        self.llamaPath = QLineEdit(self.gridLayoutWidget_3)
        self.llamaPath.setObjectName(u"llamaPath")
        self.llamaPath.setFont(font4)

        self.gridLayout_3.addWidget(self.llamaPath, 0, 1, 1, 1)

        self.releaseGPTS = QPushButton(self.gridLayoutWidget_3)
        self.releaseGPTS.setObjectName(u"releaseGPTS")
        self.releaseGPTS.setMaximumSize(QSize(50, 30))
        self.releaseGPTS.setFont(font4)

        self.gridLayout_3.addWidget(self.releaseGPTS, 1, 4, 1, 1)

        self.loadGPTXL = QPushButton(self.gridLayoutWidget_3)
        self.loadGPTXL.setObjectName(u"loadGPTXL")
        self.loadGPTXL.setMaximumSize(QSize(50, 30))
        self.loadGPTXL.setFont(font4)

        self.gridLayout_3.addWidget(self.loadGPTXL, 4, 3, 1, 1)

        self.releaseGPTL = QPushButton(self.gridLayoutWidget_3)
        self.releaseGPTL.setObjectName(u"releaseGPTL")
        self.releaseGPTL.setMaximumSize(QSize(50, 30))
        self.releaseGPTL.setFont(font4)

        self.gridLayout_3.addWidget(self.releaseGPTL, 3, 4, 1, 1)

        self.gptXlLabel = QLabel(self.gridLayoutWidget_3)
        self.gptXlLabel.setObjectName(u"gptXlLabel")
        self.gptXlLabel.setFont(font4)

        self.gridLayout_3.addWidget(self.gptXlLabel, 4, 0, 1, 1)

        self.loadGPTS = QPushButton(self.gridLayoutWidget_3)
        self.loadGPTS.setObjectName(u"loadGPTS")
        self.loadGPTS.setMaximumSize(QSize(50, 30))
        self.loadGPTS.setFont(font4)

        self.gridLayout_3.addWidget(self.loadGPTS, 1, 3, 1, 1)

        self.releaseAll = QPushButton(self.gridLayoutWidget_3)
        self.releaseAll.setObjectName(u"releaseAll")
        self.releaseAll.setFont(font4)

        self.gridLayout_3.addWidget(self.releaseAll, 5, 3, 1, 2)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.radioButtonGPU = QRadioButton(self.gridLayoutWidget_3)
        self.radioButtonGPU.setObjectName(u"radioButtonGPU")

        self.horizontalLayout_10.addWidget(self.radioButtonGPU)

        self.radioButtonCPU = QRadioButton(self.gridLayoutWidget_3)
        self.radioButtonCPU.setObjectName(u"radioButtonCPU")

        self.horizontalLayout_10.addWidget(self.radioButtonCPU)


        self.gridLayout_3.addLayout(self.horizontalLayout_10, 6, 1, 1, 4)

        self.llamaPathSelect = QPushButton(self.gridLayoutWidget_3)
        self.llamaPathSelect.setObjectName(u"llamaPathSelect")
        self.llamaPathSelect.setMaximumSize(QSize(30, 30))

        self.gridLayout_3.addWidget(self.llamaPathSelect, 0, 2, 1, 1)

        self.gptsPathSelect = QPushButton(self.gridLayoutWidget_3)
        self.gptsPathSelect.setObjectName(u"gptsPathSelect")
        self.gptsPathSelect.setMaximumSize(QSize(30, 30))

        self.gridLayout_3.addWidget(self.gptsPathSelect, 1, 2, 1, 1)

        self.gptmPathSelect = QPushButton(self.gridLayoutWidget_3)
        self.gptmPathSelect.setObjectName(u"gptmPathSelect")
        self.gptmPathSelect.setMaximumSize(QSize(30, 30))

        self.gridLayout_3.addWidget(self.gptmPathSelect, 2, 2, 1, 1)

        self.gptlPathSelect = QPushButton(self.gridLayoutWidget_3)
        self.gptlPathSelect.setObjectName(u"gptlPathSelect")
        self.gptlPathSelect.setMaximumSize(QSize(30, 30))

        self.gridLayout_3.addWidget(self.gptlPathSelect, 3, 2, 1, 1)

        self.gptxlPathSelect = QPushButton(self.gridLayoutWidget_3)
        self.gptxlPathSelect.setObjectName(u"gptxlPathSelect")
        self.gptxlPathSelect.setMaximumSize(QSize(30, 30))

        self.gridLayout_3.addWidget(self.gptxlPathSelect, 4, 2, 1, 1)

        self.titleSettings = QLabel(self.settings)
        self.titleSettings.setObjectName(u"titleSettings")
        self.titleSettings.setGeometry(QRect(10, 0, 141, 61))
        self.titleSettings.setFont(font5)
        self.filePathLabelCPRS_2.addTab(self.settings, "")
        self.help = QWidget()
        self.help.setObjectName(u"help")
        self.helpText = QTextEdit(self.help)
        self.helpText.setObjectName(u"helpText")
        self.helpText.setGeometry(QRect(10, 60, 691, 351))
        self.titleHelp = QLabel(self.help)
        self.titleHelp.setObjectName(u"titleHelp")
        self.titleHelp.setGeometry(QRect(10, 0, 141, 61))
        self.titleHelp.setFont(font5)
        self.filePathLabelCPRS_2.addTab(self.help, "")

        self.verticalLayout.addWidget(self.filePathLabelCPRS_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.exitButton = QPushButton(self.verticalLayoutWidget)
        self.exitButton.setObjectName(u"exitButton")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.exitButton.sizePolicy().hasHeightForWidth())
        self.exitButton.setSizePolicy(sizePolicy5)
        self.exitButton.setMinimumSize(QSize(0, 50))

        self.horizontalLayout_2.addWidget(self.exitButton)

        self.runningText = QPlainTextEdit(self.verticalLayoutWidget)
        self.runningText.setObjectName(u"runningText")
        self.runningText.setReadOnly(True)
        self.runningText.setMaximumSize(QSize(710, 200))

        self.horizontalLayout_2.addWidget(self.runningText)

        self.horizontalLayout_2.setStretch(0, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout.setStretch(0, 1)

        self.retranslateUi(LLMzip)

        self.filePathLabelCPRS_2.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(LLMzip)
    # setupUi

#*******************************************************************************************************#

        self.model_name = ['LLaMA2-7B','GPT2-Small','GPT2-Medium','GPT2-Large','GPT2-XL']
        self.devices = ['cuda','cpu']

# connections
    # EXIT and SAVE
        self.exitButton.clicked.connect(self.exit_and_save)

    # COMPRESS
        # select model
        self.modelVersion.currentIndexChanged.connect(self.select_model)

        # choose paths
        self.filePathSelectCPRS.clicked.connect(self.file_path_select_cprs)
        self.savePathSelectCPRS.clicked.connect(self.save_path_select_cprs)

        # set memory (memory slider and memory input is connected)
        self.memorySlider.valueChanged.connect(self.memory_input_update)
        self.memoryInput.returnPressed.connect(self.memory_slider_update)

        # start compressing
        self.startCompressing.clicked.connect(self.start_compress)

    # DECOMPRESS
        # choose paths
        self.filePathSelectDCPRS.clicked.connect(self.file_path_select_dcprs)
        self.savePathSelectDCPRS.clicked.connect(self.save_path_select_dcprs)

        # start decompressing
        self.startDecompressing.clicked.connect(self.start_decompress)

    # SETTINGS
        # model path select
        self.llamaPathSelect.clicked.connect(lambda :self.PathSelected_clicked(self.model_name[0]))

        self.gptsPathSelect.clicked.connect(lambda :self.PathSelected_clicked(self.model_name[1]))
        self.gptmPathSelect.clicked.connect(lambda :self.PathSelected_clicked(self.model_name[2]))
        self.gptlPathSelect.clicked.connect(lambda :self.PathSelected_clicked(self.model_name[3]))
        self.gptxlPathSelect.clicked.connect(lambda :self.PathSelected_clicked(self.model_name[4]))

        # device select
        self.radioButtonGPU.clicked.connect(lambda :self.device_clicked(self.devices[0]))
        self.radioButtonCPU.clicked.connect(lambda :self.device_clicked(self.devices[1]))

        # load model
        self.loadLLAMA.clicked.connect(self.load_llama_model)

        self.loadGPTS.clicked.connect(lambda :self.load_gpt2_model(self.model_name[1]))
        self.loadGPTM.clicked.connect(lambda :self.load_gpt2_model(self.model_name[2]))
        self.loadGPTL.clicked.connect(lambda :self.load_gpt2_model(self.model_name[3]))
        self.loadGPTXL.clicked.connect(lambda :self.load_gpt2_model(self.model_name[4]))

        # release model
        self.releaseLLAMA.clicked.connect(self.release_llama_model)

        self.releaseGPTS.clicked.connect(self.release_gpt2s_model)
        self.releaseGPTM.clicked.connect(self.release_gpt2m_model)
        self.releaseGPTL.clicked.connect(self.release_gpt2l_model)
        self.releaseGPTXL.clicked.connect(self.release_gpt2xl_model)

        self.releaseAll.clicked.connect(self.release_all_model)

        self.load_variables()

    def retranslateUi(self, LLMzip):
        LLMzip.setWindowTitle(QCoreApplication.translate("LLMzip", u"LLMZip", None))
        self.memoryLabel.setText(QCoreApplication.translate("LLMzip", u"\u8bb0\u5fc6", None))
#if QT_CONFIG(tooltip)
        self.memoryInput.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.filePathLabelCPRS.setText(QCoreApplication.translate("LLMzip", u"\u6587\u4ef6\u8def\u5f84", None))
        self.filePathSelectCPRS.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.savePathLabelCPRS.setText(QCoreApplication.translate("LLMzip", u"\u5b58\u50a8\u8def\u5f84", None))
        self.savePathSelectCPRS.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
#if QT_CONFIG(tooltip)
        self.startCompressing.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(accessibility)
        self.startCompressing.setAccessibleName("")
#endif // QT_CONFIG(accessibility)
#if QT_CONFIG(accessibility)
        self.startCompressing.setAccessibleDescription("")
#endif // QT_CONFIG(accessibility)
        self.startCompressing.setText(QCoreApplication.translate("LLMzip", u"\u5f00\u59cb\u538b\u7f29", None))
        self.modelVersionLabel.setText(QCoreApplication.translate("LLMzip", u"\u6a21\u578b", None))
        self.modelVersion.setItemText(2, QCoreApplication.translate("LLMzip", u"GPT2-Small", None))
        self.modelVersion.setItemText(3, QCoreApplication.translate("LLMzip", u"GPT2-Medium", None))
        self.modelVersion.setItemText(4, QCoreApplication.translate("LLMzip", u"GPT2-Large", None))
        self.modelVersion.setItemText(5, QCoreApplication.translate("LLMzip", u"GPT2-XL", None))
        self.modelVersion.setItemText(1, QCoreApplication.translate("LLMzip", u"LLaMA2-7B", None))
        self.modelVersion.setItemText(0, QCoreApplication.translate("LLMzip", u"请选择一个模型", None))

        self.titleCompress.setText(QCoreApplication.translate("LLMzip", u"Compress", None))
        self.filePathLabelCPRS_2.setTabText(self.filePathLabelCPRS_2.indexOf(self.compress), QCoreApplication.translate("LLMzip", u"\u538b\u7f29", None))
        self.savePathLabelDCPRS.setText(QCoreApplication.translate("LLMzip", u"\u5b58\u50a8\u8def\u5f84", None))
        self.savePathSelectDCPRS.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.filePathLabelDCPRS.setText(QCoreApplication.translate("LLMzip", u"\u6587\u4ef6\u8def\u5f84", None))
        self.filePathSelectDCPRS.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.startDecompressing.setText(QCoreApplication.translate("LLMzip", u"\u5f00\u59cb\u89e3\u538b", None))
        self.titleDecompress.setText(QCoreApplication.translate("LLMzip", u"Decompress", None))
        self.filePathLabelCPRS_2.setTabText(self.filePathLabelCPRS_2.indexOf(self.decompress), QCoreApplication.translate("LLMzip", u"\u89e3\u538b", None))
        self.llamaLabel.setText(QCoreApplication.translate("LLMzip", u"LLaMA2-7B\u6a21\u578b\u8def\u5f84", None))
        self.gptLargeLabel.setText(QCoreApplication.translate("LLMzip", u"GPT2-Large\u6a21\u578b\u8def\u5f84", None))
        self.gptMediumLabel.setText(QCoreApplication.translate("LLMzip", u"GPT2-Medium\u6a21\u578b\u8def\u5f84", None))
        self.gptSmallLabel.setText(QCoreApplication.translate("LLMzip", u"GPT2-Small\u6a21\u578b\u8def\u5f84", None))
        self.loadGPTM.setText(QCoreApplication.translate("LLMzip", u"\u52a0\u8f7d", None))
        self.deviceLabel.setText(QCoreApplication.translate("LLMzip", u"device(\u8bbe\u5907)", None))
        self.releaseGPTXL.setText(QCoreApplication.translate("LLMzip", u"\u91ca\u653e", None))
        self.releaseGPTM.setText(QCoreApplication.translate("LLMzip", u"\u91ca\u653e", None))
        self.loadLLAMA.setText(QCoreApplication.translate("LLMzip", u"\u52a0\u8f7d", None))
        self.loadGPTL.setText(QCoreApplication.translate("LLMzip", u"\u52a0\u8f7d", None))
        self.releaseLLAMA.setText(QCoreApplication.translate("LLMzip", u"\u91ca\u653e", None))
        self.releaseGPTS.setText(QCoreApplication.translate("LLMzip", u"\u91ca\u653e", None))
        self.loadGPTXL.setText(QCoreApplication.translate("LLMzip", u"\u52a0\u8f7d", None))
        self.releaseGPTL.setText(QCoreApplication.translate("LLMzip", u"\u91ca\u653e", None))
        self.gptXlLabel.setText(QCoreApplication.translate("LLMzip", u"GPT2-XL\u6a21\u578b\u8def\u5f84", None))
        self.loadGPTS.setText(QCoreApplication.translate("LLMzip", u"\u52a0\u8f7d", None))
        self.releaseAll.setText(QCoreApplication.translate("LLMzip", u"\u91ca\u653e\u6240\u6709\u6a21\u578b", None))
        self.radioButtonGPU.setText(QCoreApplication.translate("LLMzip", u"cuda(gpu)", None))
        self.radioButtonCPU.setText(QCoreApplication.translate("LLMzip", u"cpu\uff08\u4e0d\u63a8\u8350\uff09", None))
        self.llamaPathSelect.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.gptsPathSelect.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.gptmPathSelect.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.gptlPathSelect.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.gptxlPathSelect.setText(QCoreApplication.translate("LLMzip", u"\u00b7\u00b7\u00b7", None))
        self.titleSettings.setText(QCoreApplication.translate("LLMzip", u"Settings", None))
        self.filePathLabelCPRS_2.setTabText(self.filePathLabelCPRS_2.indexOf(self.settings), QCoreApplication.translate("LLMzip", u"\u8bbe\u7f6e", None))
        #self.helpText.setHtml(QCoreApplication.translate("LLMzip", u"sore the eventual fascist vi</p></body></html>", None))
        self.helpText.setHtml(    "<h1>LLMZip-大语言模型文本压缩与重建</h1>\n"
    "<h2>LLMZip</h2>\n"
    "<p>本程序使用<strong>大语言模型（Large-Language-Model）</strong>对文本进行压缩，使用大预言模型预测结合传统文本压缩的方法，可实现对文本的<strong>高倍率</strong>的<strong>无损</strong>压缩与重建</p>\n"
    "<h2>使用介绍</h2>\n"
    "<h3><strong>1.压缩界面</strong></h3>\n"
    "<p>在压缩界面，从上到下依次可以选择压缩所使用的模型，被压缩文件(.txt)的路径，压缩完成后文件的保存路径以及进行记忆(Memory)的设置。在所有这些都设置好后，点击开始压缩，本程序即可对所选文件进行无损压缩。</p>\n"
    "<h3><strong>2.解压界面</strong></h3>\n"
    "<p>在解压界面，从上到下依次可以选择被压缩文件(.bin)的路径，解压完成后文件的保存路径。在这两项设置好后，点击开始解压，本程序即可对所选的被压缩文件进行自动的无损解压，复原为原先的文本文件。</p>\n"
    "<h3><strong>3.设置界面</strong></h3>\n"
    "<p>在设置界面，可以对所提供的五种模型进行路径的选择，模型运行设备的选择，模型加载与释放操作。</p>\n"
    "<h3><strong>4.模型下载与使用</strong></h3>\n"
    "<h4>模型的下载地址如下：(注意必须下载pytorch版本的模型)</h4>\n"
    "<ul>\n"
    "<li>LLaMA2-7B模型下载教程：<a href='https://www.bilibili.com/video/BV1H14y1X7kD'>https://www.bilibili.com/video/BV1H14y1X7kD</a> (建议使用Gmail进行申请，不建议使用QQ邮箱，可能会申请不上)</li>\n"
    "<li>GPT2-Small模型下载地址：<a href='https://huggingface.co/openai-community/gpt2'>https://huggingface.co/openai-community/gpt2</a></li>"
    "<li>GPT2-Medium模型下载地址：<a href='https://huggingface.co/openai-community/gpt2-medium'>https://huggingface.co/openai-community/gpt2-medium</a></li>"
    "<li>GPT2-Large模型下载地址：<a href='https://huggingface.co/openai-community/gpt2-large'>https://huggingface.co/openai-community/gpt2-large</a></li>"
    "<li>GPT2-XL模型下载地址：<a href='https://huggingface.co/openai-community/gpt2-xl'>https://huggingface.co/openai-community/gpt2-xl</a></li>"
    "</ul>"
    "<h4>模型文件夹的格式如下：</h4>"
    "<p>LLaMA模型按照如下方式放在一个文件夹中：</p>"
    "<ul>\n"
    "<li>checklist.chk</li>\n"
    "<li>consolidated.00.pth</li>\n"
    "<li>params.json</li>\n"
    "<li>tokenizer_checklist.chk</li>\n"
    "<li>tokenizer.model</li>\n"
    "</ul>\n"
    "<p>GPT2模型按照如下方式放在一个文件夹中：</p>"
    "<ul>\n"
    "<li>config.json</li>\n"
    "<li>merges.txt</li>\n"
    "<li>pytorch_model.bin</li>\n"
    "<li>tokenizer_config.json</li>\n"
    "<li>tokenizer.json</li>\n"
    "<li>vocab.json</li>\n"
    "</ul>"
    "<p><em><strong>注：</strong></em>不同版本GPT2模型的config和pytorch_model不同，其余文件通用</p>")
        self.titleHelp.setText(QCoreApplication.translate("LLMzip", u"Help", None))
        self.filePathLabelCPRS_2.setTabText(self.filePathLabelCPRS_2.indexOf(self.help), QCoreApplication.translate("LLMzip", u"\u5e2e\u52a9", None))
        self.exitButton.setText(QCoreApplication.translate("LLMzip", u"\u9000\u51fa", None))
    # retranslateUi

# HELP TEXT (HTML)


# EXIT SAVE and LOAD
    def exit_and_save(self):
        self.save_variables()
        QApplication.instance().quit()

    def save_variables(self):
        variables_to_save = {
            'config':{
                'llama_model_path' : self.llama_model_path,
                'gpts_model_path' : self.gpts_model_path,
                'gptm_model_path' : self.gptm_model_path,
                'gptl_model_path' : self.gptl_model_path,
                'gptxl_model_path' : self.gptxl_model_path,
                'device' : self.device,
                'memory' : self.memory,
                'save_path_cprs' : self.save_path_cprs,
                'save_path_dcprs' : self.save_path_dcprs
            }
        }
        with open('config.json', 'w') as f:
            json.dump(variables_to_save, f, indent=4)

    def load_variables(self):
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                variables = json.load(f)
            self.set_variables(variables)

    def set_variables(self, varibles):
        self.llamaPath.setText(varibles['config']['llama_model_path'])
        self.llama_model_path = varibles['config']['llama_model_path']

        self.gptsPath.setText(varibles['config']['gpts_model_path'])
        self.gpts_model_path = varibles['config']['gpts_model_path']

        self.gptmPath.setText(varibles['config']['gptm_model_path'])
        self.gptm_model_path = varibles['config']['gptm_model_path']

        self.gptlPath.setText(varibles['config']['gptl_model_path'])
        self.gptl_model_path = varibles['config']['gptl_model_path']

        self.gptxlPath.setText(varibles['config']['gptxl_model_path'])
        self.gptxl_model_path = varibles['config']['gptxl_model_path']

        if varibles['config']['device'] == 'cpu':
            self.radioButtonCPU.setChecked(True)
            self.device = 'cpu'
        elif varibles['config']['device'] == 'cuda':
            self.radioButtonGPU.setChecked(True)
            self.device = 'cuda'

        self.memory = varibles['config']['memory']
        self.memoryInput.setText(str(varibles['config']['memory']))
        self.memorySlider.setValue(varibles['config']['memory'])

        self.savePathCPRS.setText(varibles['config']['save_path_cprs'])
        self.save_path_cprs = varibles['config']['save_path_cprs']

        self.savePathDCPRS.setText(varibles['config']['save_path_dcprs'])
        self.save_path_dcprs = varibles['config']['save_path_dcprs']

# COMPRESS
    # combobox select model
    def select_model(self, index):
        if index > 0:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.runningText.appendPlainText(f'{current_time} model {self.model_name[index-1]} will be used to compress')
            self.model_version = index - 1
    # select file path to compress
    def file_path_select_cprs(self):
        window = QWidget()
        current_path = QDir.currentPath()
        file_path, _ = QFileDialog.getOpenFileName(
                window, "Open File", current_path,
                "Text Files (*.txt)")
        if file_path:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.filePathCPRS.setText(file_path)
            self.file_path_cprs = file_path
        self.runningText.appendPlainText(f'{current_time} file-to-be-compressed path selected at: {file_path}')

    # select file path to save compressed file
    def save_path_select_cprs(self):
        window = QWidget()
        current_path = QDir.currentPath()
        folder_path = QFileDialog.getExistingDirectory(
            window, "Open Folder", current_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if folder_path:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.savePathCPRS.setText(folder_path)
            self.save_path_cprs = folder_path
        self.runningText.appendPlainText(f'{current_time} compressed file save path selected at: {folder_path}')

    # set memeory in both slider and value-typing ways
    def memory_input(self, text):
        window = QWidget()
        if text.isdigit():
            value = int(text)
            if 1 <= value <= 256:
                self.memory_slider_update()
            else:
                QMessageBox.warning(window, "Memory range error","Please enter a value between 1 and 256.")
                self.memoryInput.setText("")
                self.memorySlider.setValue(1)
        else:
            QMessageBox.warning(window, "Input error", "Memory must be a integer value!")

    def memory_slider_update(self):
        window = QWidget()
        if self.memoryInput.text().isdigit():
            value = int(self.memoryInput.text())
            self.memorySlider.setValue(value)
            self.memory = value
        else:
            QMessageBox.warning(window, "Input error", "Memory must be a integer value!")

    def memory_input_update(self, value):
        self.memoryInput.setText(str(value))
        self.memory = value

    # start compressing function
    def start_compress(self):
        window = QWidget()
        model_version = self.modelVersion.currentIndex() - 1
        if model_version >= 0:
            if model_version == 0:
                self.model = self.llama_model
            elif model_version == 1:
                self.model = self.gpts_model
            elif model_version == 2:
                self.model = self.gptm_model
            elif model_version == 3:
                self.model = self.gptl_model
            elif model_version == 4:
                self.model = self.gptxl_model

            if self.model == None:
                QMessageBox.warning(window, "Settings error", f"Please load the model you choose : {self.model_name[model_version]} before compressing")
            else:
                if self.model != None and len(self.filePathCPRS.text()) > 0 and len(self.savePathCPRS.text()) > 0 and self.memory > 0:

                    current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                    self.runningText.appendPlainText(f'{current_time} start compressing')

                    self.compress_worker = CompressThread(self.model,
                                                 self.file_path_cprs,
                                                 self.memory,
                                                 self.device,
                                                 self.save_path_cprs)
                    self.compress_worker.compress_signal.connect(self.complete_compress)
                    self.compress_worker.start()
                    self.startCompressing.setEnabled(False)

                else:
                    QMessageBox.warning(window, "Settings error", "Please check your settings before compressing")
        else:
            QMessageBox.warning(window, "Settings error", "Please choose a model before compressing")

    def complete_compress(self, progress):
        if progress == 100:
            self.startCompressing.setEnabled(True)
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.runningText.appendPlainText(f'{current_time} compressing... 100%')
            self.runningText.appendPlainText(f'{current_time} finished compressing and compressed file is saved to {self.save_path_cprs}')

        else:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.runningText.appendPlainText(f'{current_time} compressing... {progress}%')

# DECOMPRESS
    # select compressed file path
    def file_path_select_dcprs(self):
        window = QWidget()
        current_path = QDir.currentPath()
        file_path, _ = QFileDialog.getOpenFileName(
            window, "Open File", current_path,
            "Binary Files (*.bin)")
        if file_path:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.filePathDCPRS.setText(file_path)
            self.file_path_dcprs = file_path
        self.runningText.appendPlainText(f'{current_time} file-to-be-decompressed path selected at: {file_path}')

    # select file path to save decompressed file
    def save_path_select_dcprs(self):
        window = QWidget()
        current_path = QDir.currentPath()
        folder_path = QFileDialog.getExistingDirectory(
            window, "Open Folder", current_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if folder_path:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.savePathDCPRS.setText(folder_path)
            self.save_path_dcprs = folder_path
        self.runningText.appendPlainText(f'{current_time} decompressed file save path selected at: {folder_path}')

    def get_rvm(self, file_name):

        window = QWidget()
        if os.path.exists(file_name) and file_name[-4:] == '.bin':

            ranks = []

            # read the compressed data
            with open(file_name, 'rb') as file:
                compressed_data = file.read()
            decompressed_data = zlib.decompress(compressed_data)

            # transfer the binary data to numbers
            for i in range(0, len(decompressed_data), 2):
                rank = int.from_bytes(decompressed_data[i:i + 2], 'little')
                ranks.append(rank)

            # recognize the version of the model and the value of the memory automatically
            version = ranks[0]
            memory = ranks[1]

            return ranks, version, memory

        else:
            QMessageBox.warning(window, "Settings error", f"{os.path.basename(file_name)} is not a binary file")


    # start decompressing function
    def start_decompress(self):

        window = QWidget()

        ranks, version, memory = self.get_rvm(self.file_path_dcprs)

        # print("version:",version,"memory:",memory)

        if (version == 0 and self.llama_model) or (version == 1 and self.gpts_model) or (version == 2 and self.gptm_model) or (version == 3 and self.gptl_model) or (version == 4 and self.gptxl_model):

            if version == 0:
                model = self.llama_model
            elif version == 1:
                model = self.gpts_model
            elif version == 2:
                model = self.gptm_model
            elif version == 3:
                model = self.gptl_model
            elif version == 4:
                model = self.gptxl_model

            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")

            self.runningText.appendPlainText(f'{current_time} self-check complete,using {self.model_name[version]} and memory is {memory}')
            self.runningText.appendPlainText(f'{current_time} start decompressing')

            self.decompress_worker = DecompressThread(self.save_path_dcprs,
                                                      model,
                                                      ranks,
                                                      version,
                                                      memory
                                                      )

            self.decompress_worker.decompress_signal.connect(self.complete_decompress)
            self.decompress_worker.start()
            self.startDecompressing.setEnabled(False)

        else:
            QMessageBox.warning(window, "Settings error", f"Currently using model:{self.model_name[version]} is not loaded yet!")

    '''
        if self.complete_check:

            if len(self.filePathDCPRS.text()) > 0 and len(self.savePathDCPRS.text()) > 0:

                current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                self.runningText.appendPlainText(f'{current_time} self-check complete,using {self.model_name[version]} and memory is {memory}')
                self.runningText.appendPlainText(f'{current_time} start decompressing')

                self.decompress_worker = DecompressThread(self.file_path_dcprs,
                                                          self.save_path_dcprs,
                                                          self.llama_model,
                                                          self.gpts_model,
                                                          self.gptm_model,
                                                          self.gptl_model,
                                                          self.gptxl_model)

                self.decompress_worker.decompress_signal.connect(self.complete_decompress)
                self.decompress_worker.start()
                self.startDecompressing.setEnabled(False)

            else:
                QMessageBox.warning(window, "Settings error", "Please check your settings before decompressing")

        else:
            QMessageBox.warning(window, "Settings error", f"Model now using is not loaded yet!({self.model_name[version]})")
        '''
    def complete_decompress(self, progress):
        if progress == 100:
            self.startDecompressing.setEnabled(True)
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.runningText.appendPlainText(f'{current_time} decompressing... 100%')
            self.runningText.appendPlainText(f'{current_time} finished decompressing and decompressed file is saved to {self.save_path_dcprs}')

        else:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.runningText.appendPlainText(f'{current_time} decompressing... {progress}%')


# SETTINGS
    # load functions
    def load_llama_model(self):
        window = QWidget()
        if len(self.llama_model_path) <= 0:
            QMessageBox.warning(window, "Settings error", "Please choose the right path")
            if len(self.device) <= 0:
                QMessageBox.warning(window, "Settings error", "Please choose a device before loading the model")
            else:
                prev_time = time.time()
                self.llama_model = self.build_llama(self.llama_model_path,self.device)
                current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                self.runningText.appendPlainText(f'{current_time} successfully loaded LLaMA2-7B in {time.time()-prev_time:.2f}s')

    def load_gpt2_model(self, model_name):
        window = QWidget()
        if len(self.device) <= 0:
            QMessageBox.warning(window, "Settings error", "Please choose a device before loading the model")
        else:
            if model_name == self.model_name[1]:
                if len(self.gpts_model_path) <= 0:
                    QMessageBox.warning(window, "Settings error", "Please choose the right path")
                else:
                    prev_time = time.time()
                    self.gpts_model = self.build_GPT2(model_name.lower().replace("-", "_"), self.gpts_model_path, self.device)
                    current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                    self.runningText.appendPlainText(f'{current_time} successfully loaded {model_name} in {time.time() - prev_time:.2f}s')
            elif model_name == self.model_name[2]:
                if len(self.gptm_model_path) <= 0:
                    QMessageBox.warning(window, "Settings error", "Please choose the right path")
                else:
                    prev_time = time.time()
                    self.gptm_model = self.build_GPT2(model_name.lower().replace("-", "_"), self.gptm_model_path, self.device)
                    current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                    self.runningText.appendPlainText(f'{current_time} successfully loaded {model_name} in {time.time() - prev_time:.2f}s')
            elif model_name == self.model_name[3]:
                if len(self.gptl_model_path) <= 0:
                    QMessageBox.warning(window, "Settings error", "Please choose the right path")
                else:
                    prev_time = time.time()
                    self.gptl_model = self.build_GPT2(model_name.lower().replace("-", "_"), self.gptl_model_path, self.device)
                    current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                    self.runningText.appendPlainText(f'{current_time} successfully loaded {model_name} in {time.time() - prev_time:.2f}s')
            elif model_name == self.model_name[4]:
                if len(self.gptxl_model_path) <= 0:
                    QMessageBox.warning(window, "Settings error", "Please choose the right path")
                else:
                    prev_time = time.time()
                    self.gptxl_model = self.build_GPT2(model_name.lower().replace("-", "_"), self.gptxl_model_path, self.device)
                    current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
                    self.runningText.appendPlainText(f'{current_time} successfully loaded {model_name} in {time.time() - prev_time:.2f}s')

    # release functions
    def release_llama_model(self):
        self.llama_model = None
        gc.collect()
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.runningText.appendPlainText(f'{current_time} successfully released LLaMA2 model')

    def release_gpt2s_model(self):
        self.gpts_model = None
        gc.collect()
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.runningText.appendPlainText(f'{current_time} successfully released GPT2-Small model')
    def release_gpt2m_model(self):
        self.gptm_model = None
        gc.collect()
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.runningText.appendPlainText(f'{current_time} successfully released GPT2-Medium model')
    def release_gpt2l_model(self):
        self.gptl_model = None
        gc.collect()
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.runningText.appendPlainText(f'{current_time} successfully released GPT2-Large model')
    def release_gpt2xl_model(self):
        self.gptxl_model = None
        gc.collect()
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.runningText.appendPlainText(f'{current_time} successfully released GPT2-XL model')

    def release_all_model(self):
        self.llama_model = None
        self.gpts_model = None
        self.gptm_model = None
        self.gptl_model = None
        self.gptxl_model = None
        gc.collect()
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.runningText.appendPlainText(f'{current_time} successfully released ALL models')

    # device select function
    def device_clicked(self, device):
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.runningText.appendPlainText(f'{current_time} using device: {device}')
        self.device = device

    # model path select function
    def PathSelected_clicked(self, model_name):
        window = QWidget()
        current_path = QDir.currentPath()
        folder_path = QFileDialog.getExistingDirectory(
            window, "Open Folder", current_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if folder_path:
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            if model_name == self.model_name[0]:
                self.llamaPath.setText(folder_path)
                self.llama_model_path = folder_path
                # print(self.llama_model_path)
            elif model_name == self.model_name[1]:
                self.gptsPath.setText(folder_path)
                self.gpts_model_path = folder_path
                # print(self.gpt_model_path)
            elif model_name == self.model_name[2]:
                self.gptmPath.setText(folder_path)
                self.gptm_model_path = folder_path
                # print(self.gpt_model_path)
            elif model_name == self.model_name[3]:
                self.gptlPath.setText(folder_path)
                self.gptl_model_path = folder_path
                # print(self.gpt_model_path)
            elif model_name == self.model_name[4]:
                self.gptxlPath.setText(folder_path)
                self.gptxl_model_path = folder_path
                # print(self.gpt_model_path)
            self.runningText.appendPlainText(f'{current_time} {model_name} model path selected at: {folder_path}')


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_LLMzip()
        self.ui.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())