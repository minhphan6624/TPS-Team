# Turn off bytecode generation
import sys
sys.dont_write_bytecode = True

from app import App

def main():
    app = App()
    app.run()

if __name__ == '__main__':
    main()