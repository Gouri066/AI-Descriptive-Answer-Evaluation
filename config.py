import os

# Base folder of the project (the folder this file lives in)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Used by Flask to protect login sessions. Change to any long random text.
    SECRET_KEY = "fcec2df044b671a34a452ef941e9fe26b6ce68ac98f48e78fe1d77df117666bf"
    
    # Teachers must enter this code to register. Give it only to real teachers.
    SIGNUP_CODE = "SDMCET2026"

    # Where the SQLite database file is stored
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "answer_eval.db")

    # Where uploaded answer sheet images will be saved (Phase 4)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    # Only these image types are allowed
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    # Maximum upload size: 10 MB
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"