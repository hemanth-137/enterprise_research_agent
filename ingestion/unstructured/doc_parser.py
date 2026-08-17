import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  #use this if you face c++ compiler errors

import json
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path


def doc_parser(folder_path:str):

    # this effects only for .pdf files
    pdf_options = PdfPipelineOptions(
        do_ocr=False,   # Most of my pdfs are digital, so i turned off ocr to reduce memory usage 
        force_backend_text=True,
        do_table_structure=True,

        layout_batch_size=1,    # These 3 are for my persoanl case to save memory
        queue_max_size=1,
        images_scale=0.5,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_options
            )
        }
    )

    files = [
        file for file in folder_path.iterdir()
        if file.is_file()
    ]



    results = converter.convert_all(files,raises_on_error=False)
    documents = [
        result.document
        for result in results
        if result.document is not None
    ]

    return documents

if __name__ == "__main__":

    file_path = "./data/pdfs/0097-pdf.pdf"

    doc = doc_parser(file_path)
    markdown_output = doc.export_to_markdown()
    json_output = doc.export_to_dict()

    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(markdown_output)

    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(json_output, file, indent=2, ensure_ascii=False)

    print("\nConversion finished.")
    print("Markdown saved to output.txt")