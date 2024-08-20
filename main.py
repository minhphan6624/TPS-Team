# Turn off bytecode generation
import sys
sys.dont_write_bytecode = True

from app import MainWindow
from PyQt5.QtWidgets import *


def main():
    w = MainWindow()
    w.run()

if __name__ == '__main__':
    main()