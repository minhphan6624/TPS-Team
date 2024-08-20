import sys

sys.dont_write_bytecode = True

from app import MainWindow
from PyQt5.QtWidgets import *


def main():
    linux = False
    if "-l" in sys.argv or "--linux" in sys.argv:
        linux = True

    w = MainWindow(linux)
    w.run()
    
if __name__ == '__main__':
    main()
