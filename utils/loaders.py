import docx
import pandas as pd
import PyPDF2
import io

def load_file(uploaded_file):
    """
    Load text from various file types: PDF, DOCX, XLSX.
    Returns a list containing a single large string of the document's content.
    """
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_type == "pdf":
            # Create a BytesIO object from the uploaded file
            pdf_bytes = io.BytesIO(uploaded_file.getvalue())
            reader = PyPDF2.PdfReader(pdf_bytes)
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elif file_type == "docx":
            # Create a BytesIO object from the uploaded file
            docx_bytes = io.BytesIO(uploaded_file.getvalue())
            doc = docx.Document(docx_bytes)
            
            for para in doc.paragraphs:
                if para.text:
                    text += para.text + "\n"

        elif file_type == "xlsx":
            # Create a BytesIO object from the uploaded file
            excel_bytes = io.BytesIO(uploaded_file.getvalue())
            df = pd.read_excel(excel_bytes)
            
            # Convert each row to string and join with spaces
            text = df.astype(str).apply(lambda row: " ".join(row), axis=1).str.cat(sep="\n")

        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return [""]

    return [text] if text.strip() else [""]