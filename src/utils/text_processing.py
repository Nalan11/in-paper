import re
from bs4 import BeautifulSoup

def html_table_to_markdown(html_content):
    """Converts HTML <table> to Markdown table using BeautifulSoup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    
    markdown_tables = []
    for table in tables:
        rows = table.find_all('tr')
        if not rows: continue
        
        md_rows = []
        for i, row in enumerate(rows):
            cols = row.find_all(['td', 'th'])
            cols_text = [c.get_text(strip=True) for c in cols]
            md_rows.append("| " + " | ".join(cols_text) + " |")
            
            # Add separator after header
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
        
        markdown_tables.append("\n".join(md_rows))
    
    return "\n\n".join(markdown_tables) if markdown_tables else html_content

def clean_ocr_text(text):
    """Cleans OCR text: removes img tags and converts tables."""
    # 1. Remove <img ...> tags
    text = re.sub(r'<img[^>]*>', '', text)
    
    # 2. Extract <table> contents and convert to markdown
    def table_replacer(match):
        return html_table_to_markdown(match.group(0))
    
    cleaned_text = re.sub(r'<table>.*?</table>', table_replacer, text, flags=re.DOTALL)
    
    # 3. Clean up excessive whitespace
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def extract_and_combine_content(data):
    """Helper to extract content from PaddleOCRVL results."""
    combined_content = []
    if isinstance(data, list) and data:
        if 'parsing_res_list' in data[0] and isinstance(data[0]['parsing_res_list'], list):
            for item in data[0]['parsing_res_list']:
                content = None
                if hasattr(item, 'content'):
                    content = item.content
                elif isinstance(item, dict):
                    content = item.get('content')
                
                if content is not None:
                    combined_content.append(content)
    return '\n'.join(combined_content)
