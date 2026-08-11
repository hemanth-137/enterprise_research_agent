import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  #use this if you do not have C++ visual studio downloaded

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


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

pdf_path = "data/docx/cap.pptx"
result = converter.convert(pdf_path)

document = result.document

markdown_output = document.export_to_markdown()

with open("output.txt", "w", encoding="utf-8") as file:
    file.write(markdown_output)

print("\nConversion finished.")
print("Markdown saved to output.txt")