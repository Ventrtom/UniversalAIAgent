import os

# Nastavení – uprav dle potřeby
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Hlavní složka projektu
OUTPUT_FILE = os.path.join(ROOT_DIR, "project_snapshot.md")

CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "data")
EXCLUDE_FOLDERS = {'.git', '__pycache__', '.venv', 'venv', 'input', 'output', CHROMA_DIR, 'rag_chroma_db'}
EXCLUDE_FILES = {'project_snapshot.md', 'merge_project.py', '.env', '.png'}

def should_exclude(path: str, exclude_set: set) -> bool:
    return any(part in exclude_set for part in path.split(os.sep))

def format_file_header(path: str, root_dir: str) -> str:
    rel_path = os.path.relpath(path, root_dir)
    return f"\n\n---\n\n### `{rel_path}`\n\n```python\n"

def generate_tree_structure(root_dir: str) -> str:
    tree_lines = []
    prefix_stack = []

    def walk(directory: str, prefix: str = ''):
        entries = sorted(os.listdir(directory))
        entries = [e for e in entries if not should_exclude(os.path.join(directory, e), EXCLUDE_FOLDERS | EXCLUDE_FILES)]

        for i, entry in enumerate(entries):
            path = os.path.join(directory, entry)
            is_last = (i == len(entries) - 1)
            branch = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")

            tree_lines.append(f"{prefix}{branch}{entry}")

            if os.path.isdir(path):
                walk(path, new_prefix)

    tree_lines.append(f"{os.path.basename(root_dir)}/")
    walk(root_dir)
    return "\n".join(tree_lines)

def collect_project_snapshot(root_dir: str, output_file: str):
    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.write("# 🧠 Project Snapshot\n\n")
        out_file.write("Tento soubor obsahuje strukturu projektu a obsah jednotlivých souborů pro použití s AI asistenty.\n\n")
        out_file.write("## 📂 Struktura projektu\n\n")
        out_file.write("```\n")
        out_file.write(generate_tree_structure(root_dir))
        out_file.write("\n```\n")

        out_file.write("\n## 📄 Obsahy souborů\n")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if not should_exclude(os.path.join(dirpath, d), EXCLUDE_FOLDERS)]

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if should_exclude(filepath, EXCLUDE_FILES):
                    continue

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        out_file.write(format_file_header(filepath, root_dir))
                        out_file.write(f.read())
                        out_file.write("\n```\n")
                except Exception as e:
                    out_file.write(format_file_header(filepath, root_dir))
                    out_file.write(f"# Nelze načíst soubor: {e}\n```\n")

    print(f"✅ Snapshot vytvořen: {output_file}")

if __name__ == "__main__":
    collect_project_snapshot(ROOT_DIR, OUTPUT_FILE)
