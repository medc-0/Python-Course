"""
08-listbox.py

Listboxes let users select from a list of items.

Overview:
---------
Use tk.Listbox for single or multiple selection.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Listbox Example")

listbox = tk.Listbox(root, selectmode=tk.SINGLE)
for item in ["Apple", "Banana", "Cherry"]:
    listbox.insert(tk.END, item)
listbox.pack(pady=10)

def show_selection():
    selection = listbox.curselection()
    if selection:
        print("Selected:", listbox.get(selection[0]))

button = tk.Button(root, text="Show Selection", command=show_selection)
button.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Use selectmode=tk.MULTIPLE for multi-select.
- Use listbox.curselection() to get selected indices.
"""
