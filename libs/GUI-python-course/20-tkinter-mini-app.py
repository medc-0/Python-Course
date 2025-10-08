"""
20-mini-app.py

Build a mini app: To-Do List.

Overview:
---------
Combine widgets and logic for a useful GUI app.

Examples:
---------
"""

import tkinter as tk

def add_task():
    task = entry.get()
    if task:
        listbox.insert(tk.END, task)
        entry.delete(0, tk.END)

def remove_task():
    selection = listbox.curselection()
    if selection:
        listbox.delete(selection[0])

root = tk.Tk()
root.title("To-Do List App")

entry = tk.Entry(root, width=30)
entry.pack(pady=5)

add_btn = tk.Button(root, text="Add Task", command=add_task)
add_btn.pack(pady=5)

listbox = tk.Listbox(root, width=40)
listbox.pack(pady=5)

remove_btn = tk.Button(root, text="Remove Task", command=remove_task)
remove_btn.pack(pady=5)

root.mainloop()

"""
Tips:
-----
- Combine widgets and logic for real apps.
- Practice by building calculators, note apps, games, etc.
"""
