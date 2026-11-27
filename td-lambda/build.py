#!/usr/bin/env python3
"""Inline src/sim.js into template.html and write index.html.

python3 build.py            -> index.html (links ../shared/theme.css and ../shared/ui.js)
python3 build.py --inline   -> index.inline.html (theme + ui.js inlined, standalone file)
"""
import pathlib
import sys

here = pathlib.Path(__file__).parent
tpl = (here / "template.html").read_text(encoding="utf-8")
sim = (here / "src" / "sim.js").read_text(encoding="utf-8")
assert "/*SIM*/" in tpl
inline = "--inline" in sys.argv
if inline:
    theme = (here.parent / "shared/theme.css").read_text(encoding="utf-8")
    ui = (here.parent / "shared/ui.js").read_text(encoding="utf-8")
    theme_tag = f"<style>\n{theme}\n</style>"
    ui_tag = f"<script>\n{ui}\n</script>"
else:
    theme_tag = '<link rel="stylesheet" href="../shared/theme.css">'
    ui_tag = '<script src="../shared/ui.js"></script>'

out = (
    tpl.replace("<!--__THEME__-->", theme_tag)
    .replace("<!--__UI__-->", ui_tag)
    .replace("/*SIM*/", sim)
)
target = "index.inline.html" if inline else "index.html"
(here / target).write_text(out, encoding="utf-8")
print("wrote", here / target)
