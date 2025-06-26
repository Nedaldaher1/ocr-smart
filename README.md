# محوّل PDF عربي ذكي إلى ماركداون

هذا المشروع هو خط أنابيب متقدم لتحويل ملفات PDF التعليمية باللغة العربية إلى ملفات Markdown (MD) منظمة. يستخدم السكريبت نماذج لغوية كبيرة (LLM) متعددة الوسائط لتحليل صور صفحات PDF، استخراج النصوص، تحديد الصور الهامة، وإنشاء محتوى نصي وصفي لها، مع الحفاظ على بنية منطقية للمحتوى.

---

<div dir="rtl">

## 📜 نظرة عامة

الهدف الأساسي لهذا المشروع هو أتمتة عملية تحويل المواد التعليمية المطبوعة (PDF) إلى صيغة رقمية مرنة وقابلة للتحرير (Markdown). هذا يسهل أرشفة المحتوى، البحث فيه، وإعادة استخدامه في منصات التعليم الإلكتروني. السكريبت مصمم خصيصاً للتعامل مع تحديات المحتوى العربي، مثل اتجاه النص والصور المتداخلة.

## ✨ الميزات الرئيسية

- **معالجة صور الصفحات**: يقوم بتنقية كل صفحة (إزالة الميلان، تقليل التشويش، تحسين التباين) لزيادة دقة التعرف على النصوص (OCR).
- **استخراج ذكي للصور**: يميز بين الصور التوضيحية الهامة (مثل الرسوم البيانية والخرائط) والعناصر الزخرفية (مثل الأيقونات والإطارات)، ويتجاهل الأخيرة.
- **تحليل بواسطة الذكاء الاصطناعي**: يرسل صورة الصفحة والصور المستخرجة إلى نموذج `Gemini 2.5 Pro` لتحليل شامل.
- **توليد محتوى وصفي**: يقوم النموذج اللغوي بتوليد أوصاف عربية للصور الهامة ويدمجها في ملف الماركداون.
- **استخراج البيانات الوصفية**: يتعرف على اسم الوحدة، عنوان الدرس، ورقم الدرس من الصفحة.
- **هيكل مخرجات منظم**: ينشئ مجلداً خاصاً لكل ملف PDF مع مجلدات فرعية للمحتوى والصور، مما يضمن تنظيم المشروع.
- **واجهة سطر الأوامر (CLI)**: يوفر واجهة مرنة للتحكم في المعالجة، مثل تحديد مجلدات المدخلات والمخرجات.

## ⚙️ آلية العمل

يعمل السكريبت عبر سلسلة من الخطوات المنظمة لكل صفحة في ملف PDF:

1.  **تحويل PDF إلى صور**: يتم تحويل كل صفحة من ملف PDF إلى صورة عالية الدقة (PNG).
2.  **معالجة الصورة الرئيسية**: تخضع صورة الصفحة لعمليات تحسين لضمان أقصى وضوح للنص.
3.  **استخراج الصور الفرعية**: يتم استخراج جميع الصور المضمنة من الصفحة الأصلية باستخدام PyMuPDF.
4.  **التحليل عبر LLM**: تُرسل صورة الصفحة المُحسَّنة والصور الفرعية المستخرجة إلى Gemini API.
5.  **توليد الماركداون**: يقوم النموذج بتحليل المحتوى، كتابة النص، تقييم الصور، كتابة أوصاف للصور الهامة، وتنسيق كل ذلك في كائن JSON.
6.  **حفظ وتنظيف**: يتم حفظ محتوى الماركداون في ملف `.md`، ويتم حذف الصور التي اعتبرها النموذج زخرفية أو غير ضرورية للحفاظ على نظافة المخرجات.

## 🚀 الإعداد والتشغيل

اتبع هذه الخطوات لإعداد المشروع وتشغيله على جهازك.

### 1. المتطلبات الأساسية

- **Python 3.8+**
- **مفتاح API**: تحتاج إلى مفتاح API من [OpenRouter.ai](https://openrouter.ai/) للوصول إلى نموذج Gemini.
- **Poppler**: هذه مكتبة خارجية ضرورية لعمل `pdf2image`.

    - **على Windows**:
        1.  حمّل أحدث نسخة من [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
        2.  فك ضغط الملف وأضف مسار مجلد `bin` إلى متغيرات البيئة `Path`.

    - **على macOS** (باستخدام Homebrew):
        ```bash
        brew install poppler
        ```
    - **على Linux** (Debian/Ubuntu):
        ```bash
        sudo apt-get update && sudo apt-get install poppler-utils
        ```

### 2. خطوات التثبيت

1.  **استنسخ المستودع (Clone the repository):**
    ```bash
    git clone [your-repository-url]
    cd [repository-folder]
    ```

2.  **ثبّت المكتبات المطلوبة:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **قم بإعداد متغيرات البيئة:**
    -   أنشئ ملفاً جديداً باسم `.env` في المجلد الرئيسي للمشروع.
    -   أضف مفتاح API الخاص بك داخل الملف بهذا الشكل:
        ```
        OPENROUTER_API_KEY="sk-or-..."
        ```

### 3. طريقة الاستخدام

1.  ضع جميع ملفات PDF التي تريد معالجتها داخل مجلد `pdfs`. إذا لم يكن موجوداً، قم بإنشائه.
2.  قم بتشغيل السكريبت من سطر الأوامر.

    **أبسط أمر للتشغيل:**
    ```bash
    python ocr_smart_arabic_v9.py
    ```

    **أمر مع خيارات مخصصة:**
    ```bash
    python ocr_smart_arabic_v9.py --pdf-dir "my_books" --output-dir "results" --dpi 300 --save-processed-scans
    ```

### 4. خيارات سطر الأوامر

-   `--pdf-dir`: المجلد الذي يحتوي على ملفات PDF (الافتراضي: `pdfs`).
-   `--output-dir`: المجلد الرئيسي لحفظ جميع المخرجات (الافتراضي: `output`).
-   `--dpi`: الدقة (DPI) المستخدمة لتحويل PDF إلى صور (الافتراضي: `300`).
-   `--api-key`: لتمرير مفتاح API مباشرة (يتجاوز متغير البيئة).
-   `--clear-output`: لحذف مجلد المخرجات الرئيسي قبل البدء من جديد.
-   `--save-processed-scans`: لحفظ صور الصفحات بعد معالجتها (لأغراض المراجعة والتصحيح).

## 📁 هيكل المخرجات

لكل ملف PDF تتم معالجته، سيتم إنشاء الهيكل التالي داخل مجلد `output`:

```
output/
└── اسم_ملف_PDF/
    ├── illustrative_images/
    │   ├── page_1/
    │   │   └── img_0.png
    │   └── page_2/
    │       ├── img_0.png
    │       └── img_1.png
    ├── markdown_content/
    │   └── Unit_الوحدة_الأولى_Lesson_الدرس_الأول_عنوان_الدرس.md
    └── processed_page_scans/  (إذا تم تفعيله)
        ├── scan_page_1.png
        └── scan_page_2.png
```

## 📄 الترخيص

هذا المشروع مرخص تحت رخصة MIT.

</div>

---

# Smart Arabic PDF to Markdown Converter

This project is an advanced pipeline for converting Arabic educational PDF files into structured Markdown (MD) files. The script utilizes a multimodal Large Language Model (LLM) to analyze PDF page images, extract text, identify important images, and generate descriptive content for them, all while maintaining a logical content structure.

---

<div dir="ltr">

## 📜 Overview

The primary goal of this project is to automate the conversion of printed educational materials (PDF) into a flexible and editable digital format (Markdown). This facilitates content archiving, searching, and reuse in e-learning platforms. The script is specifically designed to handle the challenges of Arabic content, such as right-to-left text direction and nested images.

## ✨ Key Features

- **Page Image Pre-processing**: Cleans each page (deskew, denoise, enhance contrast) to improve Optical Character Recognition (OCR) accuracy.
- **Intelligent Image Extraction**: Differentiates between important illustrative images (like diagrams and maps) and decorative elements (like icons and borders), ignoring the latter.
- **AI-Powered Analysis**: Sends the page image and extracted sub-images to the `Gemini 2.5 Pro` model for a comprehensive analysis.
- **Descriptive Content Generation**: The LLM generates Arabic descriptions for important images and integrates them into the Markdown file.
- **Metadata Extraction**: Identifies the unit name, lesson title, and lesson number from the page.
- **Organized Output Structure**: Creates a dedicated folder for each PDF with clear subdirectories for content and images, ensuring a tidy project.
- **Command-Line Interface (CLI)**: Provides a flexible interface to control the processing, such as specifying input and output directories.

## ⚙️ How It Works

The script operates through a structured pipeline for each page in the PDF file:

1.  **PDF to Image Conversion**: Each PDF page is converted into a high-resolution image (PNG).
2.  **Main Image Pre-processing**: The page image undergoes enhancement operations to ensure maximum text clarity.
3.  **Sub-Image Extraction**: All embedded images are extracted from the original page using PyMuPDF.
4.  **LLM Analysis**: The processed page image and the extracted sub-images are sent to the Gemini API.
5.  **Markdown Generation**: The model analyzes the content, transcribes the text, evaluates the images, writes captions for the important ones, and formats everything into a JSON object.
6.  **Saving and Cleanup**: The Markdown content is saved to a `.md` file, and images deemed decorative or unnecessary by the model are deleted to keep the output clean.

## 🚀 Setup and Usage

Follow these steps to set up and run the project on your machine.

### 1. Prerequisites

- **Python 3.8+**
- **API Key**: You will need an API key from [OpenRouter.ai](https://openrouter.ai/) to access the Gemini model.
- **Poppler**: This is an external dependency required by `pdf2image`.

    - **On Windows**:
        1.  Download the latest release from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
        2.  Unzip the file and add the path to the `bin` folder to your system's `Path` environment variable.

    - **On macOS** (using Homebrew):
        ```bash
        brew install poppler
        ```
    - **On Linux** (Debian/Ubuntu):
        ```bash
        sudo apt-get update && sudo apt-get install poppler-utils
        ```

### 2. Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [repository-folder]
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    -   Create a new file named `.env` in the project's root directory.
    -   Add your API key to the file in this format:
        ```
        OPENROUTER_API_KEY="sk-or-..."
        ```

### 3. How to Use

1.  Place all the PDF files you want to process inside the `pdfs` directory. If it doesn't exist, create it.
2.  Run the script from your command line.

    **Simplest command to run:**
    ```bash
    python ocr_smart_arabic_v9.py
    ```

    **Command with custom options:**
    ```bash
    python ocr_smart_arabic_v9.py --pdf-dir "my_books" --output-dir "results" --dpi 300 --save-processed-scans
    ```

### 4. Command-Line Arguments

-   `--pdf-dir`: The directory containing the PDF files (Default: `pdfs`).
-   `--output-dir`: The main directory to save all outputs (Default: `output`).
-   `--dpi`: The resolution (DPI) to use for converting PDF to images (Default: `300`).
-   `--api-key`: Pass the API key directly (overrides the environment variable).
-   `--clear-output`: Delete the main output directory before starting.
-   `--save-processed-scans`: Enable saving the processed (B&W) page images for review.

## 📁 Output Structure

For each processed PDF, the following structure will be created inside the `output` directory:

```
output/
└── PDF_FILENAME/
    ├── illustrative_images/
    │   ├── page_1/
    │   │   └── img_0.png
    │   └── page_2/
    │       ├── img_0.png
    │       └── img_1.png
    ├── markdown_content/
    │   └── Unit_UnitOne_Lesson_LessonOne_LessonTitle.md
    └── processed_page_scans/  (if enabled)
        ├── scan_page_1.png
        └── scan_page_2.png
```

## 📄 License

This project is licensed under the MIT License.

</div>