"""Assemble index.html from src/template.html + data.json + src/sim.js.

python3 build.py            -> index.html (links ../shared/theme.css, for GitHub Pages)
python3 build.py --inline   -> index.inline.html (theme inlined, a single standalone file)
"""
import sys, pathlib
here = pathlib.Path(__file__).parent
tpl = (here / "src/template.html").read_text()
data = (here / "data.json").read_text().strip()
sim = (here / "src/sim.js").read_text()
sim = sim[: sim.index("if (typeof module")]  # drop the node-only export
inline = "--inline" in sys.argv
theme = (here.parent / "shared/theme.css").read_text()
tag = f"<style>\n{theme}\n</style>" if inline else '<link rel="stylesheet" href="../shared/theme.css">'
out = tpl.replace("<!--__THEME__-->", tag).replace("/*__DATA__*/", data).replace("/*__SIM__*/", sim)
(here / ("index.inline.html" if inline else "index.html")).write_text(out)
print("wrote", "index.inline.html" if inline else "index.html", len(out), "bytes")
