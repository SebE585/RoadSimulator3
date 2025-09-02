import yaml

def merge_schema(base_schema_path: str, fragments: list[dict]) -> dict:
    with open(base_schema_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base_cols = base.get("columns", [])
    names = {c.get("name") for c in base_cols if isinstance(c, dict)}

    for frag in fragments or []:
        for c in frag.get("columns", []):
            if isinstance(c, dict) and c.get("name") and c["name"] not in names:
                base_cols.append(c)
                names.add(c["name"])
    base["columns"] = base_cols
    return base