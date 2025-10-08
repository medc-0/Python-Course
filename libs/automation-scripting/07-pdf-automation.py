"""
07-pdf-automation.py

Automate PDF merging and splitting.

Overview:
---------
PyPDF2 is a Python library for working with PDF files. You can merge, split, and extract pages from PDFs.
PDF automation is useful for document management, combining reports, or extracting specific pages.

Troubleshooting:
----------------
If you get a "file not found" error, make sure:
- The PDF files (file1.pdf, file2.pdf) are in the same directory as your script.
- The filenames and extensions are correct (case-sensitive).
- Use os.getcwd() to check your current working directory.

How to use PyPDF2:
------------------
- PdfMerger: Merge multiple PDFs into one.
- PdfReader: Read and extract pages from a PDF.
- PdfWriter: Write pages to a new PDF.

Examples:
---------
"""

import os
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
# debug
# print("Current directory:", os.getcwd())
# print("Files in directory:", os.listdir("."))

# Merge PDFs
merger = PdfMerger()
if os.path.exists("file1.pdf") and os.path.exists("file2.pdf"):
    merger.append("file1.pdf")
    merger.append("file2.pdf")
    merger.write("merged.pdf")
    merger.close()
    print("PDFs merged into merged.pdf")
else:
    print("file1.pdf or file2.pdf not found. Please check the directory and filenames.")

# Split PDF (extract first page)
if os.path.exists("file1.pdf"):
    reader = PdfReader("file1.pdf")
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    with open("page1.pdf", "wb") as f:
        writer.write(f)
    print("First page extracted to page1.pdf")
else:
    print("file1.pdf not found for splitting.")

print("PDF automation example (requires actual PDF files).")

"""
Tips:
-----
- Install PyPDF2: pip install pypdf2
- Use os.listdir() and os.getcwd() to debug file locations.
- Always check if files exist before processing.
- PyPDF2 works only with non-encrypted PDFs.
- For more features, see: https://pypdf2.readthedocs.io/en/latest/
"""
