from Drawing import GUI
from DrawingFunctions import DrawingHelperFunction
if __name__ == "__main__":
    gui=GUI()
    Manager=DrawingHelperFunction(gui)
    Manager.Bind()
    gui.Run()
