from pathlib import Path
path = Path('app.py')
text = path.read_text(encoding='utf-8')
old = "st.set_page_config(page_title=\"���ʲ�������Ϣ���\", layout=\"wide\")\n\nSTART_DATE = \"2015-10-01\"\nDATA_DIR = Path(\"d:/pyproject/�ƽ�������Ϣ����\")\nif not DATA_DIR.exists():\n    DATA_DIR = (Path.cwd() / \"�ƽ�������Ϣ����\").resolve()\n\nASSET_CONFIG = {"
if old not in text:
    raise SystemExit('old block not found')
new = "st.set_page_config(page_title=\"多资产隐含降息次数与资产价格对照\", layout=\"wide\")\n\nSTART_DATE = \"2015-10-01\"\nBASE_DIR = Path(__file__).resolve().parent\nDATA_DIR = BASE_DIR / \"data\"\n\nASSET_CONFIG = {"
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
