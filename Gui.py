# Creates GUI for the project

import tkinter
from matplotlib.figure import Figure

import MainFile
from tkinter import filedialog, StringVar


# noinspection PyArgumentList
class Window:
    # Browse button click method : opens browse file dialog box
    def __browse(self):
        self.entry.config(state='normal')
        self.file_name = filedialog.askopenfilename(filetypes=(("CSV  files", "*.csv"), ("All Files", "*.*")))
        self.input_text.set(self.file_name)
        self.entry.config(state='readonly')

    # Check result click method : outputs the result of the selected method
    def __checkResult(self):
        self.selected_method = self.variable.get()
        self.data_text.set("")
        self.rmse_text.set("")
        if self.file_name == "":
            self.data_text.set("No File Selected. Please select a file !!!")
        else:
            try:
                data, rmse, figure = MainFile.Main().start(self.file_name, self.selected_method)
                self.rmse_text.set(rmse)
                self.data_text.set(data)
                figure.show()
            except:
                self.data_text.set("File is not appropriate. Please select another file !!!")

    # Constructor
    def __init__(self):

        self.file_name = ""
        self.input_text = StringVar()
        self.variable = StringVar()
        self.selected_method = StringVar()
        self.rmse_text = StringVar()
        self.data_text = StringVar()
        self.graph = Figure()

    # Create the window with elements
    def drawWindow(self, master):

        master.title("Stock Prize Prediction")
        tkinter.Label(master).grid(rowspan=5)
        tkinter.Label(master, text="Select Data File").grid(row=6)
        self.entry = tkinter.Entry(master, textvariable=self.input_text)
        self.entry.config(state='readonly')
        self.entry.grid(row=6, column=1)
        tkinter.Button(master, text="Browse File", fg="red", command=lambda: self.__browse()).grid(row=6, column=2)
        tkinter.Label(master).grid(row=6, column=3)
        tkinter.Label(master).grid(rowspan=3)
        tkinter.Label(master, text="Select Method").grid(row=10)
        self.variable.set("Moving Average")
        tkinter.OptionMenu(master, self.variable, "Moving Average", "Linear Regression", "K-Nearest Neighbours",
                           "Auto ARIMA", "LSTM").grid(row=10, column=1)
        tkinter.Label(master).grid(rowspan=3)
        tkinter.Button(master, text="Check Result", command=lambda: self.__checkResult()).grid()
        tkinter.Label(master).grid(rowspan=5)
        tkinter.Label(master, text="Data used for the method").grid()
        tkinter.Label(master).grid(rowspan=3)
        tkinter.Label(master, textvariable=self.data_text).grid()
        tkinter.Label(master).grid(rowspan=3)
        tkinter.Label(master, text="Root Mean Square Error RMSE for the method").grid()
        tkinter.Label(master).grid(rowspan=3)
        tkinter.Label(master, textvariable=self.rmse_text).grid()
        master.mainloop()


if __name__ == '__main__':
    root = tkinter.Tk()
    window = Window()
    window.drawWindow(root)
    root.mainloop()
