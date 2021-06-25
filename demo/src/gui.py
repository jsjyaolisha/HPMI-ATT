import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from recognition import *
from PyQt5 import QtWidgets

sys.path.append(os.path.dirname(__file__) + '/ui')
from ui import UI


def load_cnn_model():
    model = VGG16_HPMI()
    model.load_weights('./models/HPMI_best_weights.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
