from PyPDF2 import PdfMerger

merger = PdfMerger()
merger.append("figures/im2col_boom.pdf")
merger.append("figures/unfold_boom.pdf")
merger.write("combined.pdf")
merger.close()