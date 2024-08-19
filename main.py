import sys

sys.dont_write_bytecode = True

from app import App

def main():
    linux = False
    if "-l" in sys.argv or "--linux" in sys.argv:
        linux = True

    app = App(linux)

    app.run()

if __name__ == '__main__':
    main()
