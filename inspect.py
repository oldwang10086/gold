from pathlib import Path
lines = Path('app.py').read_text(encoding='utf-8').splitlines()
for i in range(10,17):
    print(i, repr(lines[i]))
