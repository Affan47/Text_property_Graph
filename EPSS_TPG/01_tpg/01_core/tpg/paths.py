"""Canonical paths for the editable research checkout; never create files here."""
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".tpg-project-root").is_file()
)
TPG_ROOT = PROJECT_ROOT / "01_tpg"
EXAMPLES = TPG_ROOT / "02_examples"
TEXT_INPUTS = EXAMPLES / "04_inputs/01_text"
PDF_INPUTS = EXAMPLES / "04_inputs/02_pdf"
GENERATED = EXAMPLES / "03_generated"
WORKSPACE = TPG_ROOT / "05_workspace"
DEFAULT_DATABASE = str(WORKSPACE / "01_database/tpg_workspace.db")
DEFAULT_UPLOADS = WORKSPACE / "02_uploads"
DEFAULT_CHATBOT_STORE = str(WORKSPACE / "03_chatbot_stores/store.json")
