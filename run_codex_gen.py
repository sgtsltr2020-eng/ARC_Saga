from pathlib import Path

from saga.core.codex_index import CodexIndexGenerator


def main():
    base_dir = Path("c:/Users/sgtsl/PROJECTS/ARC_Saga")
    docs_path = base_dir / "docs/sagacodex_python_fastapi.md"
    index_path = base_dir / ".saga/sagacodex_index.json"

    print(f"Generating index from {docs_path} to {index_path}...")
    generator = CodexIndexGenerator(docs_path, index_path)
    generator.write_index()
    print("Done.")

if __name__ == "__main__":
    main()
