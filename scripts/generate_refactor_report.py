#!/usr/bin/env python3
import os

# Configurations
THRESHOLD_LINES = 400
EXCLUDE_DIRS = {"venv", "__pycache__", "build", "data", "tests", "static", "templates", "out", ".git"}
PROJECT_ROOT = "."
EXCLUDE_FILES = set()

def scan_python_files(root="."):
    py_files = []
    for dirpath, _, filenames in os.walk(root):
        if any(excluded in dirpath for excluded in EXCLUDE_DIRS):
            continue
        for filename in filenames:
            if filename.endswith(".py") and filename not in EXCLUDE_FILES:
                full_path = os.path.join(dirpath, filename)
                py_files.append(full_path)
    return py_files

def analyze_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return {"path": path, "error": str(e)}

    n_lines = len(lines)
    has_docstring = lines and (
        lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''")
    )
    n_defs = sum(1 for l in lines if l.strip().startswith("def "))
    n_classes = sum(1 for l in lines if l.strip().startswith("class "))
    needs_refactor = n_lines > THRESHOLD_LINES or not has_docstring

    return {
        "path": path,
        "lines": n_lines,
        "functions": n_defs,
        "classes": n_classes,
        "docstring": has_docstring,
        "needs_refactor": needs_refactor,
        "error": None
    }

def generate_report():
    print("# 🔧 Rapport automatique de refactoring")
    print("\n> Généré automatiquement par `make refactor-plan`\n")
    print("| Fichier | Lignes | Fonctions | Classes | Docstring | 🔥 À revoir |")
    print("|---------|--------|-----------|---------|-----------|-------------|")

    files = scan_python_files(PROJECT_ROOT)
    files.sort()

    for filepath in files:
        stat = analyze_file(filepath)
        if stat["error"]:
            print(f"| `{stat['path']}` | ❌ Erreur lecture : {stat['error']} |")
            continue
        print(f"| `{stat['path']}` | {stat['lines']} | {stat['functions']} | {stat['classes']} | {'✅' if stat['docstring'] else '❌'} | {'⚠️' if stat['needs_refactor'] else ''} |")

if __name__ == "__main__":
    generate_report()
