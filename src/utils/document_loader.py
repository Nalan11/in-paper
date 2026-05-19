import os
from pdf2image import convert_from_path

def split_pdf_to_images(pdf_path, output_dir):
    """
    Splits a PDF into a list of image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path)
    image_paths = []
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.jpg")
        image.save(image_path, "JPEG")
        image_paths.append(image_path)
        
    return image_paths
