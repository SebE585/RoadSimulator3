#!/usr/bin/env python3
import os, re, sys, ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if __file__ else Path.cwd()
IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", "data", "out", "outputs", "logs", "node_modules", "dist", "build"}
PY = [p for p in ROOT.rglob("*.py") if not any(part in IGNORE_DIRS for part in p.parts)]

def find_deprecated_functions():
    deprecated = {}  # name -> {"file": Path, "lineno": int}
    for path in PY:
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_depr = any(
                    (isinstance(d, ast.Name) and d.id == "deprecated") or
                    (isinstance(d, ast.Attribute) and d.attr == "deprecated")
                    for d in (node.decorator_list or [])
                )
                if has_depr:
                    deprecated[node.name] = {"file": path, "lineno": node.lineno}
    return deprecated

def find_calls(name, defining_file):
    calls = []
    for path in PY:
        if path == defining_file:
            # on ignore le fichier qui définit la fonction pour éviter le faux-positif sur "def name"
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(path))
        except Exception:
            continue
        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                callee = node.func
                # cas simples: name(...) ou module.name(...)
                if isinstance(callee, ast.Name) and callee.id == name:
                    calls.append((path, node.lineno))
                elif isinstance(callee, ast.Attribute) and callee.attr == name:
                    calls.append((path, node.lineno))
                self.generic_visit(node)
        CallVisitor().visit(tree)
    return calls

def main():
    deprecated = find_deprecated_functions()
    if not deprecated:
        print("Aucune fonction @deprecated trouvée.")
        return
    print("# Carte des appels aux fonctions @deprecated\n")
    for name, meta in sorted(deprecated.items()):
        print(f"- {name}  (déclarée dans {meta['file']}:{meta['lineno']})")
        calls = find_calls(name, meta["file"])
        if not calls:
            print("    ↳ aucun appel restant ✅")
        else:
            for f, ln in calls:
                print(f"    ↳ {f}:{ln}")
        print()
if __name__ == "__main__":
    sys.exit(main())