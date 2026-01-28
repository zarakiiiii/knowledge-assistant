from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text=[]

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)

if __name__ == "__main__":
    text = extract_text_from_pdf("data/uploads/ML Unit 2 - part 1.pdf")
    print(text[:500])