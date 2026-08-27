import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # to avoid any cpp compiling errors
#import json
from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from tqdm import tqdm


def doc_parser(folder_path, single_file = False):

    target_path = Path(folder_path)

    pdf_options = PdfPipelineOptions(
        do_ocr=False,   # as am running it local and have moslty digital pdfs
        force_backend_text=True,
        do_table_structure=True,

        layout_batch_size=1,
        queue_max_size=1,
        images_scale=0.5, # to avoid ram issues
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_options
            )
        }
    )

    if not single_file:

        files = [
            file for file in target_path.iterdir()
            if file.is_file() and file.suffix.lower() == ".pdf"
        ]

        if not files:
            print("No PDF files found in the given folder.")
            return
        
        results_generator = converter.convert_all(files, raises_on_error=False)

        for result in tqdm(results_generator,total=len(files), desc="Parsing PDFs", unit="doc"):
            if result.document is not None:
                yield result.document
            else:
                print(f"unable to covert {result.input.file}")
    else:
        result = converter.convert(target_path, raises_on_error=False)
        if result.document is not None:
            yield result.document

if __name__ == "__main__":

    file_path = "./data/pdfs/resume.pdf"

    doc = doc_parser(file_path,single_file=True)
    # #markdown_output = doc.export_to_markdown()
    # json_output = doc.export_to_dict()

    # # with open("output_res.txt", "w", encoding="utf-8") as file:
    # #     file.write(markdown_output)

    # with open("output_res.json", "w", encoding="utf-8") as file:
    #     json.dump(json_output, file, indent=2, ensure_ascii=False)

    # print("\nConversion finished.")
    # print("Markdown saved to output.txt")

    #print(doc.export_to_text())