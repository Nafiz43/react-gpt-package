"""Self-contained offline graph reports; no external JavaScript, fonts, or model calls."""
import json
from pathlib import Path
from .storage import write_text


def render_graph(path, rows, articles, taxonomy, domain, processed_at, sources_html, fallback_html):
    assets = Path(__file__).with_name('assets')
    fields = ('rank', 'quality_tier', 'cluster_id', 'article_id', 'article_title', 'actionable',
              'impact', 'evidence', 'support', 'avg_confidence', 'SOUND', 'PRECISE', 'CATEGORY',
              'categories', 'models_present', 'judge')
    payload = {'rows': [{k: r[k] for k in fields if k in r} for r in rows],
               'articles': articles, 'taxonomy': taxonomy, 'domain': domain, 'processed_at': processed_at}
    # JSON is inside a raw-text script element; escape HTML delimiters, including </script>.
    encoded = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    replacements = {'__STYLE__': (assets / 'report.css').read_text(),
                    '__SCRIPT__': (assets / 'report.js').read_text(), '__DATA__': encoded,
                    '__SOURCES__': sources_html, '__FALLBACK__': fallback_html}
    import re
    template = (assets / 'report.html').read_text()
    # Single substitution pass prevents corpus text from being interpreted as template markers.
    write_text(path, re.sub(r'__(?:STYLE|SCRIPT|DATA|SOURCES|FALLBACK)__', lambda m: replacements[m.group()], template))
