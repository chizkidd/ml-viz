"""Build index.html from template.html + rows.py (the 21 dataset rows)."""
import json, pathlib, importlib.util
here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("rows", here / "rows.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
keys = ("id", "cat", "sub", "lang", "gt", "out", "cer", "e", "p")
rows = [dict(zip(keys, r)) for r in m.ROWS]
html = (here / "template.html").read_text().replace("/*__ROWS__*/", json.dumps(rows, ensure_ascii=False))
(here / "index.html").write_text(html)
print("wrote index.html", len(html), "bytes,", len(rows), "rows")
