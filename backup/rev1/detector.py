from tkinter import *
import sqlite3

root=Tk()
root.geometry('410x450')
root.title("Drowsiness Detector")
root.configure(background="powder blue")

menu=Menu(root)
root.config(menu=menu)

menu = Menu(root)
root.config(menu=menu)

def helpp():
   help(sqlite3)

subm = Menu(menu)
menu.add_cascade(label="Help",menu=subm)
subm.add_command(label="Sqlite3 Docs",command=helpp)

root.mainloop()