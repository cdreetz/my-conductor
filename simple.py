import re
import PyPDF2


def extract_largest_number_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        largest_number = float('-inf')
        largest_number_page = -1
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            potential_numbers = re.findall(r'\b-?\d+\.\d+\b|\b-?\d+\b', text)

            numbers = []
            for num_str in potential_numbers:
                index = text.find(num_str)

                # if num has characters, its likely an ID and not a true num value
                if (index == 0 or not text[index-1].isalnum()) and (index + len(num_str) == len(text) or not text[index + len(num_str)].isalnum()):
                    numbers.append(num_str)
            
            for num_str in numbers:
                num = float(num_str)
                if num > largest_number:
                    largest_number = num
                    largest_number_page = page_num + 1
        
        return largest_number, largest_number_page

if __name__ == "__main__":
    pdf_path = "FY25 Air Force Working Capital Fund.pdf"
    largest_value, page_number = extract_largest_number_from_pdf(pdf_path)
    print(f"The largest numerical value in the document is: {largest_value}")
    print(f"Found on page: {page_number}")

