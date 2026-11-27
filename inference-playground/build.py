"""Assemble index.html from src/template.html + data.json + src/sim.js.

python3 build.py            -> index.html (links ../shared/theme.css and ../shared/ui.js)
python3 build.py --inline   -> index.inline.html (theme + ui.js inlined, a single standalone file)
"""
import sys, pathlib
here = pathlib.Path(__file__).parent
tpl = (here / "src/template.html").read_text(encoding="utf-8")
data = (here / "data.json").read_text(encoding="utf-8").strip()
sim = (here / "src/sim.js").read_text(encoding="utf-8")
sim = sim[: sim.index("if (typeof module")]  # drop the node-only export
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
    .replace("/*__DATA__*/", data)
    .replace("/*__SIM__*/", sim)
)
target = "index.inline.html" if inline else "index.html"
(here / target).write_text(out, encoding="utf-8")
print("wrote", target, len(out), "bytes")
