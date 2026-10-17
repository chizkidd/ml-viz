#!/usr/bin/env python3
"""Inline src/sim.js into template.html and write index.html (single self-contained page + shared theme link)."""
import pathlib
here = pathlib.Path(__file__).parent
tpl = (here / "template.html").read_text(encoding="utf-8")
sim = (here / "src" / "sim.js").read_text(encoding="utf-8")
assert "/*SIM*/" in tpl
(here / "index.html").write_text(tpl.replace("/*SIM*/", sim), encoding="utf-8")
print("wrote", here / "index.html")
