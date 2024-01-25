from pdf_parser import process_pdf
import json

pdf_processor  = process_pdf()

object = pdf_processor.consume_pdf('data/input/BNSS2023.pdf')

with open("output.json", "w") as outfile:
    json.dump(object, outfile)
    outfile.close()