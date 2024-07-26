from tkinter import filedialog
from tkinter import messagebox

"""
Basic functions used in multiple scripts
"""

def yes_no_inp(q):
    while True:
        a = input(q)
        if a.lower() == 'y':
            a = True
            break
        elif a.lower() == 'n':
            a = False
            break
        else:
            print('Invalid input.')
    return a


def int_inp(q, minval=0, maxval=10):
    while True:
        a = input(q)
        try:
            a = int(a)
        except ValueError:
            print('Input is not an integer.')
            continue
        if minval <= a <= maxval:
            break
        else:
            print(f'Input values between {minval} and {maxval} are accepted.')
    return a


def select_file(message, title, filetypes):
    messagebox.showinfo('Information', message)
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(),
                                           title=title,
                                           filetypes=filetypes)
    if file_path is None or file_path == '':
        raise Exception('No file selected. Aborting.')
    return file_path


def select_folder(message, title):
    messagebox.showinfo('Information', message)
    folder_path = filedialog.askdirectory(initialdir=os.getcwd(),
                                          title=title)
    if folder_path is None or folder_path == '':
        raise Exception('No folder selected. Aborting')
    return folder_path
