"""Write docs/presentation/tokens.css from docs/presentation/tokens.json (light and dark)."""
from pathlib import Path
import json

DIR = Path(__file__).resolve().parents[1] / "docs" / "presentation"


def main():
    t = json.loads((DIR / "tokens.json").read_text(encoding="utf-8"))
    colors = t["color"]["tokens"]
    out = [f"/* {t['name']} - generated from tokens.json by scripts/presentation_tokens_css.py */"]
    for i, theme in enumerate(th["id"] for th in t["color"]["themes"]):
        selector = f':root, [data-theme="{theme}"]' if i == 0 else f'[data-theme="{theme}"]'
        out.append(selector + " {")
        out += [f"  --{c['name']}: {c['value'][theme]};" for c in colors]
        out.append("}")
    out.append(":root {")
    for family in ("spacing", "radius"):
        out += [f"  --{c['name']}: {c['value']};" for c in t[family]["tokens"]]
    out += [f"  --font-{k}: {v};" for k, v in t["type"]["families"].items()]
    out.append("}")
    for group in t["type"]["groups"]:
        for s in group["styles"]:
            spacing = f" letter-spacing: {s['letterSpacing']};" if "letterSpacing" in s else ""
            out.append(f".{s['name']} {{ font-family: var(--font-{group['family']}); "
                       f"font-size: {s['fontSize']}; line-height: {s['lineHeight']}; "
                       f"font-weight: {s['fontWeight']};{spacing} }}")
    (DIR / "tokens.css").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {DIR / 'tokens.css'}")


if __name__ == "__main__":
    main()
