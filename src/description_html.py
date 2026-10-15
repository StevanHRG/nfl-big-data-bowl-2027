"""Renders the writeup body to formatted HTML so it can be copied into Kaggle's rich-text editor."""
from pathlib import Path

import markdown

W = Path(__file__).resolve().parents[1] / "writeup"
md = (W / "DESCRIPTION_PASTE.md").read_text(encoding="utf8")
md = md.replace(">>>>> INSERT IMAGE HERE:", "[[ INSERT IMAGE HERE:").replace("<<<<<", "]]")
body = markdown.markdown(md, extensions=["tables"])
html = f"""<!doctype html>
<meta charset="utf-8">
<title>Copy into Kaggle</title>
<style>
  body {{ max-width: 820px; margin: 32px auto; padding: 0 16px; font-family: "Segoe UI", sans-serif; line-height: 1.5; }}
  .note {{ background: #fff3cd; padding: 12px; border-radius: 6px; }}
  table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
</style>
<p class="note"><b>Instruções:</b> clique em qualquer lugar ABAIXO da linha cinza, aperte Ctrl+A e Ctrl+C,
e cole (Ctrl+V) no campo Project Description do Kaggle, depois de apagar o texto que está lá.
A caixa amarela também será copiada; apague-a no Kaggle depois de colar.</p>
<hr>
{body}
"""
(W / "DESCRIPTION_COPY.html").write_text(html, encoding="utf8")
print("ok")
