#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_smarter.py - Module
A module for processing PDF files and converting them to Markdown.
This module contains all the core functions for image processing,
content extraction, and communicating with the Large Language Model.
"""

# --- Library Imports ---
import os
import re
import shutil
import sys
import json
import base64
import time
from typing import List, Dict, Optional

import requests
import fitz  # PyMuPDF
import cv2
import numpy as np
from pdf2image import convert_from_path

# Import rich for optional, styled logging
try:
    from rich.console import Console
except ImportError:
    # Create a dummy Console class if rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

# --- API Configuration ---
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-pro"

# --- Helper and Pre-processing Functions ---
def sanitize_filename(name: str) -> str:
    """Removes invalid characters from filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_").replace("\n", "")
    return name[:100]

# --- Image Processing Functions ---
def preprocess_page_image(page_image: np.ndarray) -> np.ndarray:
    """Applies a full pre-processing pipeline to a single page image."""
    img = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray = deskew(gray)
    gray = apply_gamma(gray)
    gray = denoise(gray)
    bw_image = binarize(gray)
    cropped_image = crop_borders(bw_image)
    
    return cropped_image

def deskew(gray: np.ndarray) -> np.ndarray:
    """Corrects image skew."""
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0: return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = 90 + angle
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def apply_gamma(gray: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Adjusts gamma to improve contrast."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)], dtype="uint8")
    return cv2.LUT(gray, table)

def denoise(gray: np.ndarray, h: float = 20) -> np.ndarray:
    """Removes noise from the image."""
    return cv2.fastNlMeansDenoising(gray, h=h)

def binarize(denoised: np.ndarray) -> np.ndarray:
    """Converts the image to black and white."""
    _, bw = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def crop_borders(img: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crops white borders around the content."""
    inverted_img = cv2.bitwise_not(img)
    coords = cv2.findNonZero(inverted_img)
    if coords is None: return img
    x, y, w, h = cv2.boundingRect(coords)
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    return img[y1:y2, x1:x2]

# --- Content Extraction and LLM Functions ---
def extract_illustrative_images(doc: fitz.Document, page_num: int, page_images_output_dir: str, console: Console) -> List[Dict]:
    """Extracts illustrative images from a single PDF page."""
    os.makedirs(page_images_output_dir, exist_ok=True)
    page = doc.load_page(page_num)
    image_list = page.get_images(full=True)
    extracted_images = []
    min_width, min_height = 50, 50

    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            if not image_bytes or base_image["width"] < min_width or base_image["height"] < min_height:
                continue

            img_ext = base_image["ext"]
            img_filename = f"img_{img_index}.{img_ext}"
            img_path = os.path.join(page_images_output_dir, img_filename)

            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
            relative_path = os.path.join(os.path.basename(page_images_output_dir), img_filename).replace("\\", "/")
            extracted_images.append({"path": relative_path, "base64": img_b64})
        except Exception as e:
            console.print(f"       [yellow]Warning: Could not extract image xref={xref} on page {page_num + 1}. Error: {e}[/yellow]")
            continue
            
    return extracted_images

def generate_analysis_from_llm(full_page_b64: str, sub_images: List[Dict], api_key: str, console: Console) -> Optional[Dict]:
    """Sends page image and sub-images to the LLM for JSON analysis."""
    if not api_key:
        console.print("[bold red]Error: API key is not available.[/bold red]")
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt_text = (
        "You are an expert AI assistant for converting educational materials into structured Arabic Markdown. "
        "You will receive a main image of a textbook page and a list of smaller images extracted from it.\n\n"
        "Your task is to create a single JSON object with the keys: 'unit_name', 'lesson_number', 'lesson_title', 'markdown_content'.\n\n"
        "Follow these instructions precisely:\n"
        "1.  **Transcribe Text**: Analyze the main page image and transcribe all text into 'markdown_content'.\n"
        "2.  **Handle Blanks**: If you see dotted lines (`...........`) or other fill-in-the-blank areas, replace them with `...`.\n"
        "3.  **MOST IMPORTANT - Evaluate Image Importance**: For each of the smaller images, you must classify it into one of three categories based on this priority:\n"
        "    a. **Top Priority (Must Include)**: **Annotated diagrams and infographics.**\n"
        "    b. **Useful Content (Include if Relevant)**: Photos that have significant visual information.\n"
        "    c. **Decorative (Must Ignore)**: Purely decorative elements.\n"
        "4.  **Integrate and Describe Images**: For important images, create a clear Arabic description and insert it into the markdown using the format: `![وصف عربي للصورة](image_path)`.\n"
        "5.  **Extract Metadata**: Identify 'unit_name', 'lesson_number', 'lesson_title'. Use an empty string \"\" if not found.\n\n"
        "Image paths to evaluate:\n"
    )

    if sub_images:
        for i, img_info in enumerate(sub_images):
            prompt_text += f"- Image path {i+1}: {img_info['path']}\n"
    else:
        prompt_text += "- No sub-images were extracted from this page.\n"
        
    message_content = [{"type": "text", "text": prompt_text}]
    message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{full_page_b64}"}})
    for img_info in sub_images:
        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_info['base64']}"}})
        
    data = { "model": MODEL_NAME, "messages": [{"role": "user", "content": message_content}], "temperature": 0.0, "response_format": {"type": "json_object"} }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=180)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            console.print(f"\n   [yellow]API request failed (Attempt {attempt + 1}/{max_retries}): {e}[/yellow]")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                console.print("[bold red]   Exceeded max retries.[/bold red]")
                if 'resp' in locals() and hasattr(resp, 'text'):
                    console.print(f"[red]Last response content: {resp.text}[/red]")
                return None

# --- Main Processing Functions ---
def process_single_pdf(pdf_path: str, settings: dict, api_key: str, console: Console):
    """Coordinates the processing of a single PDF file from start to finish."""
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    console.print(f"\n[bold magenta]--- Starting processing for PDF: {pdf_base_name} ---[/bold magenta]")

    pdf_output_root = os.path.join(settings["output_dir"], pdf_base_name)
    md_output_dir = os.path.join(pdf_output_root, "markdown_content")
    illustrative_images_dir = os.path.join(pdf_output_root, "illustrative_images")
    processed_scans_dir = os.path.join(pdf_output_root, "processed_page_scans")

    os.makedirs(md_output_dir, exist_ok=True)
    os.makedirs(illustrative_images_dir, exist_ok=True)
    if settings["save_processed_scans"]:
        os.makedirs(processed_scans_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        pil_page_images = convert_from_path(pdf_path, dpi=settings["dpi"], thread_count=4)
    except Exception as e:
        console.print(f"[bold red]Fatal Error: Could not open or convert PDF '{pdf_base_name}'. File may be corrupt or Poppler is not installed.[/bold red]")
        console.print(f"Error details: {e}")
        return

    for i, page_image in enumerate(pil_page_images):
        page_num = i
        console.print(f"  - Processing page {page_num + 1}/{len(doc)}...")
        
        processed_image_np = preprocess_page_image(page_image)
        
        if settings["save_processed_scans"]:
            save_path = os.path.join(processed_scans_dir, f"scan_page_{page_num + 1}.png")
            cv2.imwrite(save_path, processed_image_np)

        output_dir_for_page_images = os.path.join(illustrative_images_dir, f"page_{page_num + 1}")
        sub_images = extract_illustrative_images(doc, page_num, output_dir_for_page_images, console)
        if sub_images:
            console.print(f"    Found {len(sub_images)} illustrative images.")
            
        _, buffered = cv2.imencode('.png', processed_image_np)
        full_page_b64 = base64.b64encode(buffered).decode('utf-8')

        result = generate_analysis_from_llm(full_page_b64, sub_images, api_key, console)
        
        if result and 'choices' in result and result['choices']:
            try:
                message_content = result['choices'][0]['message']['content']
                json_obj = json.loads(message_content)
                markdown_content = json_obj.get('markdown_content')

                if not markdown_content or not markdown_content.strip():
                    console.print(f"      [yellow]Warning: Markdown content is empty for page {page_num + 1}. Skipping.[/yellow]")
                    continue
                
                referenced_images = set(re.findall(r'!\[.*?\]\((.*?)\)', markdown_content))
                for img_info in sub_images:
                    if img_info['path'] not in referenced_images:
                        full_img_path = os.path.join(illustrative_images_dir, img_info['path'])
                        if os.path.exists(full_img_path):
                            os.remove(full_img_path)
                
                final_markdown = re.sub(r'(!\[.*?\]\()', r'\1../illustrative_images/', markdown_content)
                
                unit = json_obj.get('unit_name') or 'UnknownUnit'
                lesson_num = json_obj.get('lesson_number') or f'Page_{page_num + 1}'
                title = json_obj.get('lesson_title') or 'Untitled'
                filename = f"Unit_{sanitize_filename(unit)}_Lesson_{sanitize_filename(str(lesson_num))}_{sanitize_filename(title)}.md"
                output_md_path = os.path.join(md_output_dir, filename)

                with open(output_md_path, 'w', encoding='utf-8') as f:
                    f.write(final_markdown)
                console.print(f"    [green]Success: Page content saved to '[/green][cyan]{os.path.basename(output_md_path)}[/cyan]'")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                console.print(f"      [bold red]Error: Could not parse LLM response for page {page_num + 1}. Error: {e}[/bold red]")
                console.print(f"      [red]Raw LLM Response: {result['choices'][0]['message']['content']}[/red]")
        else:
            console.print(f"      [bold red]Error: Did not get a valid response from LLM for page {page_num + 1}.[/bold red]")
    doc.close()
    console.print(f"[bold magenta]--- Finished processing file: {pdf_base_name} ---[/bold magenta]")

def run_conversion_pipeline(settings: dict, api_key: str, console: Optional[Console] = None):
    """
    This is the main entry point function called by the UI (main.py)
    to start the processing pipeline for all files.
    """
    # If a console object isn't passed, create a default one.
    if console is None:
        console = Console()
        
    if settings["clear_output"]:
        console.print(f"[yellow]Clearing previous output directory: {settings['output_dir']}...[/yellow]")
        if os.path.exists(settings['output_dir']):
            shutil.rmtree(settings['output_dir'])
    
    os.makedirs(settings['output_dir'], exist_ok=True)
    os.makedirs(settings['pdf_dir'], exist_ok=True)
    
    pdf_files = [f for f in os.listdir(settings['pdf_dir']) if f.lower().endswith('.pdf')]
    if not pdf_files:
        console.print(f"\n[bold red]No PDF files found in the directory '{settings['pdf_dir']}'.[/bold red]")
        console.print("Please add some files and try again.")
        return
    
    console.print(f"\n[bold]Found {len(pdf_files)} PDF file(s). Starting conversion...[/bold]")
    for pdf_file in pdf_files:
        process_single_pdf(
            pdf_path=os.path.join(settings['pdf_dir'], pdf_file),
            settings=settings,
            api_key=api_key,
            console=console
        )
    console.print('\n[bold green]Successfully completed the process for all files.[/bold green]')
