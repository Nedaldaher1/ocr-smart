#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_smart_arabic_v9.py - نسخة محسنة لمعالجة ملفات PDF العربية.

تحسينات رئيسية في هذه النسخة:
1.  **بنية مجلدات منظمة:** يتم إنشاء مجلد مخصص لكل ملف PDF مع مجلدات فرعية واضحة:
    - `markdown_content/`: لحفظ ملفات الماركدوان النهائية.
    - `illustrative_images/`: لحفظ الصور التوضيحية المستخرجة من صفحات PDF.
    - `processed_page_scans/`: (اختياري) لحفظ صور الصفحات بعد معالجتها (أبيض وأسود، إزالة التشويش، إلخ).
2.  **معالجة صور الصفحات:** تتم معالجة كل صفحة من ملف PDF لتحسين جودة النصوص قبل إرسالها إلى النموذج اللغوي (LLM).
3.  **تحسين مسارات الصور:** يتم تعديل مسارات الصور في ملفات الماركدوان النهائية لتكون نسبية وصحيحة ضمن الهيكل الجديد.
4.  **كود أنظف:** إعادة تسمية المتغيرات والدوال لجعل الكود أكثر وضوحًا وسهولة في الصيانة.
"""
import os
import re
import shutil
import sys
import json
import base64
import argparse
import requests
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import cv2
import numpy as np
from typing import List, Dict, Optional

# --- إعدادات البرنامج ---
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-pro"
API_KEY_ENV_NAME = "OPENROUTER_API_KEY"

# --- دوال مساعدة ومعالجة أولية ---
def sanitize_filename(name: str) -> str:
    """إزالة الأحرف غير الصالحة من أسماء الملفات."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_").replace("\n", "")
    return name[:100]

def clear_directory(path: str):
    """حذف وإعادة إنشاء مجلد لضمان نظافته."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# --- دوال معالجة الصور ---
def preprocess_page_image(page_image: np.ndarray) -> np.ndarray:
    """تطبيق خط أنابيب معالجة كامل على صورة صفحة واحدة لتحسين جودة النص."""
    # تحويل من PIL Image إلى تنسيق OpenCV
    img = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # تطبيق سلسلة من التحسينات
    gray = deskew(gray)
    gray = apply_gamma(gray)
    gray = denoise(gray)
    bw_image = binarize(gray)
    cropped_image = crop_borders(bw_image)
    
    return cropped_image

def deskew(gray: np.ndarray) -> np.ndarray:
    """تصحيح ميلان الصورة."""
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0: return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = 90 + angle
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def apply_gamma(gray: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """تعديل GAMA لتحسين التباين."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)], dtype="uint8")
    return cv2.LUT(gray, table)

def denoise(gray: np.ndarray, h: float = 20) -> np.ndarray:
    """إزالة التشويش من الصورة."""
    return cv2.fastNlMeansDenoising(gray, h=h)

def binarize(denoised: np.ndarray) -> np.ndarray:
    """تحويل الصورة إلى أبيض وأسود فقط."""
    _, bw = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def crop_borders(img: np.ndarray, pad: int = 10) -> np.ndarray:
    """قص الحواف البيضاء الزائدة حول المحتوى."""
    inverted_img = cv2.bitwise_not(img)
    coords = cv2.findNonZero(inverted_img)
    if coords is None: return img
    x, y, w, h = cv2.boundingRect(coords)
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    return img[y1:y2, x1:x2]

# --- دوال استخراج المحتوى والاتصال بالنموذج اللغوي ---
def extract_illustrative_images(doc: fitz.Document, page_num: int, page_images_output_dir: str) -> List[Dict]:
    """استخراج الصور التوضيحية (غير الزخرفية) من صفحة PDF واحدة."""
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
            # إنشاء مسار نسبي بسيط لاستخدامه في الماركدوان
            relative_path = os.path.join(os.path.basename(page_images_output_dir), img_filename).replace("\\", "/")
            extracted_images.append({"path": relative_path, "base64": img_b64})
        except Exception as e:
            print(f"      تحذير: لم يتمكن من استخراج الصورة xref={xref} في الصفحة {page_num + 1}. خطأ: {e}")
            continue
            
    return extracted_images

def generate_analysis_from_llm(full_page_b64: str, sub_images: List[Dict], api_key: str) -> Optional[Dict]:
    """إرسال صورة الصفحة والصور الفرعية إلى النموذج اللغوي للحصول على تحليل بصيغة JSON."""
    if not api_key:
        print(f"خطأ: مفتاح API غير موجود. يرجى تعيين متغير البيئة '{API_KEY_ENV_NAME}'.")
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt_text = (
        "أنت مساعد AI خبير في تحويل المواد التعليمية إلى تنسيق Markdown منظم باللغة العربية. "
        "ستستلم صورة رئيسية لصفحة من كتاب مدرسي، وقائمة بالصور الأصغر المستخرجة منها.\n\n"
        "مهمتك هي إنشاء كائن JSON واحد يحتوي على المفاتيح التالية: 'unit_name', 'lesson_number', 'lesson_title', 'markdown_content'.\n\n"
        "اتبع هذه التعليمات بدقة:\n"
        "1.  **نسخ النص**: قم بتحليل صورة الصفحة الرئيسية ونسخ كل النصوص الموجودة فيها إلى 'markdown_content'.\n"
        "2.  **التعامل مع الفراغات**: إذا رأيت خطوطًا منقطة (`...........`) أو مناطق أخرى لملء الفراغ، استبدلها بـ `...`.\n"
        "3.  **الأهم - تقييم أهمية الصور**: لكل صورة من الصور الأصغر، يجب عليك تصنيفها إلى إحدى الفئات الثلاث بناءً على هذه الأولوية:\n"
        "    أ. **أولوية قصوى (يجب تضمينها)**: **الرسوم البيانية الموضحة والإنفوجرافيك.** هذه هي أهم الصور. تحتوي على رسومات وتسميات نصية (مثل الرسوم التشريحية، الخرائط، المخططات الفنية). يجب تضمينها.\n"
        "    ب. **محتوى مفيد (تضمين إذا كان ذا صلة)**: الصور التي تحتوي على معلومات بصرية هامة ولكن بدون تسميات نصية (مثل صور الأحداث التاريخية). قم بتضمينها إذا كانت تضيف قيمة للنص.\n"
        "    ج. **زخرفية (يجب تجاهلها)**: العناصر الزخرفية البحتة. **تجاهل هذه تمامًا.** وهذا يشمل الأيقونات الصغيرة (أقلام رصاص، مصابيح)، الحدود، الشعارات، أو الأشكال المجردة البسيطة.\n"
        "4.  **دمج ووصف الصور**: بالنسبة للصور في الفئتين (أ) و (ب)، قم بإنشاء وصف عربي واضح وأدخلها في 'markdown_content' في الموضع المنطقي الصحيح باستخدام التنسيق التالي: `![وصف عربي للصورة](مسار_الصورة)`.\n"
        "5.  **استخراج البيانات الوصفية**: حدد 'unit_name', 'lesson_number', 'lesson_title'. استخدم سلسلة فارغة \"\" إذا لم يتم العثور عليها.\n\n"
        "مسارات الصور التي تحتاج إلى تقييمها:\n"
    )

    if sub_images:
        for i, img_info in enumerate(sub_images):
            prompt_text += f"- مسار الصورة {i+1}: {img_info['path']}\n"
    else:
        prompt_text += "- لم يتم استخراج أي صور فرعية من هذه الصفحة.\n"
        
    message_content = [{"type": "text", "text": prompt_text}]
    message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{full_page_b64}"}})
    for img_info in sub_images:
        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_info['base64']}"}})
        
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"فشل طلب API: {e}")
        if 'resp' in locals() and hasattr(resp, 'text'):
            print(f"محتوى الاستجابة: {resp.text}")
        return None

# --- الدالة الرئيسية لتنظيم عملية المعالجة ---
def process_pdf(pdf_path: str, main_output_dir: str, dpi: int, save_scans: bool, api_key: str):
    """
    الدالة الرئيسية التي تنسق عملية معالجة ملف PDF واحد من البداية إلى النهاية.
    """
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n--- بدء معالجة ملف PDF: {pdf_base_name} ---")

    # 1. إعداد بنية المجلدات المنظمة للمخرجات
    pdf_output_root = os.path.join(main_output_dir, pdf_base_name)
    md_output_dir = os.path.join(pdf_output_root, "markdown_content")
    illustrative_images_dir = os.path.join(pdf_output_root, "illustrative_images")
    processed_scans_dir = os.path.join(pdf_output_root, "processed_page_scans")

    os.makedirs(md_output_dir, exist_ok=True)
    os.makedirs(illustrative_images_dir, exist_ok=True)
    if save_scans:
        os.makedirs(processed_scans_dir, exist_ok=True)

    try:
        # 2. تحويل PDF إلى صور وفتحه باستخدام fitz
        doc = fitz.open(pdf_path)
        pil_page_images = convert_from_path(pdf_path, dpi=dpi, thread_count=4)
    except Exception as e:
        print(f"خطأ فادح: لا يمكن فتح أو تحويل PDF '{pdf_base_name}'. قد يكون الملف تالفًا. الخطأ: {e}")
        return

    # 3. معالجة كل صفحة على حدة
    for i, page_image in enumerate(pil_page_images):
        page_num = i
        print(f"  - جاري معالجة الصفحة {page_num + 1}/{len(doc)}...")
        
        # 4. معالجة صورة الصفحة لتحسينها (B&W، تنقية، إلخ)
        processed_image_np = preprocess_page_image(page_image)
        
        # 5. (اختياري) حفظ النسخة المعالجة من صورة الصفحة
        if save_scans:
            save_path = os.path.join(processed_scans_dir, f"scan_page_{page_num + 1}.png")
            cv2.imwrite(save_path, processed_image_np)
            print(f"    تم حفظ صورة الصفحة المعالجة في: '{save_path}'")

        # 6. استخراج الصور التوضيحية من الصفحة الأصلية
        output_dir_for_page_images = os.path.join(illustrative_images_dir, f"page_{page_num + 1}")
        sub_images = extract_illustrative_images(doc, page_num, output_dir_for_page_images)
        if sub_images:
            print(f"    تم العثور على {len(sub_images)} صورة توضيحية. جاري تقييمها...")
            
        # 7. تحويل الصورة المعالجة إلى base64 لإرسالها
        _, buffered = cv2.imencode('.png', processed_image_np)
        full_page_b64 = base64.b64encode(buffered).decode('utf-8')

        # 8. إرسال البيانات إلى النموذج اللغوي
        result = generate_analysis_from_llm(full_page_b64, sub_images, api_key)
        
        # 9. معالجة النتائج وحفظها
        if result and 'choices' in result and result['choices']:
            try:
                message_content = result['choices'][0]['message']['content']
                json_obj = json.loads(message_content)
                markdown_content = json_obj.get('markdown_content')

                if not markdown_content or not markdown_content.strip():
                    print(f"      تحذير: محتوى الماركدوان فارغ للصفحة {page_num + 1}. تم التخطي."); continue
                
                # 10. تنظيف الصور غير المستخدمة وتصحيح المسارات
                referenced_images = set(re.findall(r'!\[.*?\]\((.*?)\)', markdown_content))
                for img_info in sub_images:
                    if img_info['path'] not in referenced_images:
                        full_img_path = os.path.join(illustrative_images_dir, img_info['path'])
                        if os.path.exists(full_img_path):
                            print(f"      تنظيف صورة زخرفية/غير مرجعية: {os.path.basename(full_img_path)}")
                            os.remove(full_img_path)
                
                # تصحيح مسارات الصور في الماركدوان لتكون نسبية وصحيحة
                final_markdown = re.sub(r'(!\[.*?\]\()', r'\1../illustrative_images/', markdown_content)
                
                # 11. إنشاء اسم ملف وصفي وحفظ المحتوى
                unit = json_obj.get('unit_name') or 'UnknownUnit'
                lesson_num = json_obj.get('lesson_number') or f'Page_{page_num + 1}'
                title = json_obj.get('lesson_title') or 'Untitled'
                filename = f"Unit_{sanitize_filename(unit)}_Lesson_{sanitize_filename(str(lesson_num))}_{sanitize_filename(title)}.md"
                output_md_path = os.path.join(md_output_dir, filename)

                with open(output_md_path, 'w', encoding='utf-8') as f:
                    f.write(final_markdown)
                print(f"    نجاح: تم حفظ محتوى الصفحة في '{output_md_path}'")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"      خطأ: لا يمكن تحليل استجابة LLM للصفحة {page_num + 1}. خطأ: {e}")
                print(f"      استجابة LLM الخام: {result['choices'][0]['message']['content']}")
        else:
            print(f"      خطأ: لم يتم الحصول على استجابة صالحة من LLM للصفحة {page_num + 1}.")
    doc.close()

def main():
    """الدالة الرئيسية لتشغيل البرنامج من سطر الأوامر."""
    # محاولة تحميل متغيرات البيئة من ملف .env
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            print("تم تحميل متغيرات البيئة من ملف .env.")
    except ImportError:
        print("تحذير: 'python-dotenv' غير مثبت. قم بتشغيله: pip install python-dotenv")

    parser = argparse.ArgumentParser(
        description='خط أنابيب OCR متقدم للمستندات العربية مع اختيار الصور ذي الأولوية.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pdf-dir', default='pdfs', help="المجلد الذي يحتوي على ملفات PDF للمعالجة.")
    parser.add_argument('--output-dir', default='output', help="المجلد الرئيسي لحفظ جميع المخرجات (ماركداون وصور).")
    parser.add_argument('--dpi', type=int, default=300, help="الدقة (DPI) لتحويل PDF إلى صور.")
    parser.add_argument('--api-key', help=f"مفتاح OpenRouter API. يتجاوز متغير البيئة {API_KEY_ENV_NAME}.")
    parser.add_argument('--clear-output', action='store_true', help="حذف مجلد المخرجات الرئيسي قبل البدء.")
    parser.add_argument('--save-processed-scans', action='store_true', help="تمكين حفظ صور الصفحات المعالجة (B&W) للمراجعة.")
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv(API_KEY_ENV_NAME)
    if not api_key:
        print(f"خطأ فادح: مفتاح API غير موجود. يرجى توفيره عبر سطر الأوامر أو متغيرات البيئة.")
        sys.exit(1)
    
    if args.clear_output:
        print("جاري حذف مجلد المخرجات السابق...")
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(args.pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"لم يتم العثور على ملفات PDF في المجلد '{args.pdf_dir}'.")
        return
    
    print(f"بدء عملية OCR المتقدمة...")
    for pdf_file in pdf_files:
        process_pdf(
            pdf_path=os.path.join(args.pdf_dir, pdf_file),
            main_output_dir=args.output_dir,
            dpi=args.dpi,
            save_scans=args.save_processed_scans,
            api_key=api_key
        )
    print('\nاكتملت العملية بنجاح.')

if __name__ == '__main__':
    main()
