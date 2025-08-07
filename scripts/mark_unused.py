

import os
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mark_unused")

VULTURE_REPORT = "logs/vulture_unused.txt"
DECORATOR = "@deprecated"
WARNING_LINE = '    logger.warning("⚠️ Appel d\'une fonction marquée @deprecated.")'

def parse_vulture_report(report_path):
    unused = []
    with open(report_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) >= 3:
                file_path, line_num, description = parts[0], parts[1], parts[2]
                match = re.search(r"unused (function|method) '(\w+)'", description)
                if match:
                    unused.append((file_path, int(line_num), match.group(2)))
    return unused

def insert_deprecation_warning(file_path, line_num, func_name):
    logger.info(f"📝 Traitement de {file_path}:{line_num} pour {func_name}")
    if not os.path.exists(file_path):
        logger.warning(f"⛔ Fichier non trouvé : {file_path}")
        return

    with open(file_path, "r") as f:
        lines = f.readlines()

    insert_idx = line_num - 1
    while insert_idx < len(lines):
        if lines[insert_idx].strip().startswith("def ") and func_name in lines[insert_idx]:
            break
        insert_idx += 1

    if insert_idx >= len(lines):
        logger.warning(f"❌ Fonction {func_name} non trouvée dans {file_path}")
        return

    indent_match = re.match(r"(\s*)def ", lines[insert_idx])
    indent = indent_match.group(1) if indent_match else ""

    # Insérer le décorateur + warning
    lines.insert(insert_idx, f"{indent}{DECORATOR}\n")
    lines.insert(insert_idx + 2, f"{indent}{WARNING_LINE}\n")

    with open(file_path, "w") as f:
        f.writelines(lines)

    logger.info(f"✅ Dépréciation insérée dans {file_path} à la ligne {insert_idx + 1}")

if __name__ == "__main__":
    targets = parse_vulture_report(VULTURE_REPORT)
    logger.info(f"🔍 {len(targets)} fonctions à marquer comme @deprecated")
    for path, line, name in targets:
        insert_deprecation_warning(path, line, name)