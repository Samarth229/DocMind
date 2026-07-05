"""
AST-based code chunker for Python source files.

Why AST chunking instead of recursive character splitting:
A function is a semantically complete unit — splitting it mid-body (as character
splitting would do on a long function) destroys its meaning and produces chunks
that are not coherent retrieval units. Parsing the syntax tree lets us chunk at
exact function/class boundaries and attach structural metadata (function name,
class name, docstring) that text-based chunking can't produce.

Design decision — classes vs methods:
We chunk each *method* individually (with class_name attached as metadata) rather
than the whole class as one unit. Methods are more focused retrieval units; a
class body can be hundreds of lines and would overwhelm the LLM context. Attaching
class_name preserves enough context so retrieved method chunks aren't orphaned.

Module-level code (imports, constants, if __name__ == "__main__" blocks) is
deliberately captured as a "module_level" chunk rather than silently dropped —
imports define dependencies; top-level constants define config; all are meaningful
retrieval targets for code Q&A.
"""
import logging
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

_PY_LANGUAGE = Language(tspython.language())


def _make_parser() -> Parser:
    return Parser(_PY_LANGUAGE)


def _extract_docstring(node, source_bytes: bytes) -> str | None:
    """Return the docstring of a function/class node if the first statement is a string."""
    body = next((c for c in node.children if c.type == "block"), None)
    if body is None:
        return None
    first = next((c for c in body.children if c.type not in (":", "\n")), None)
    if first is None or first.type != "expression_statement":
        return None
    expr = next((c for c in first.children if c.type == "string"), None)
    if expr is None:
        return None
    raw = source_bytes[expr.start_byte:expr.end_byte].decode("utf-8", errors="replace")
    return raw.strip('"\' \t\n').strip('"""').strip("'''").strip()


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _chunk_id(file_path: str, kind: str, name: str) -> str:
    stem = file_path.replace("\\", "/").replace("/", "_").replace(".", "_")
    return f"{stem}_{kind}_{name}"


class ASTCodeSplitter:
    def __init__(self):
        self._parser = _make_parser()

    def split_code(self, source: str, file_path: str) -> list[dict]:
        """
        Parse Python source and return one chunk per function/class/method/module-level block.

        Args:
            source:    Raw Python source text.
            file_path: Relative path string used for chunk_id and source metadata.

        Returns:
            List of chunk dicts — same shape as document pipeline chunks plus code-specific fields.
        """
        source_bytes = source.encode("utf-8")
        try:
            tree = self._parser.parse(source_bytes)
        except Exception as e:
            logger.warning("tree-sitter failed to parse '%s': %s — skipping.", file_path, e)
            return []

        if tree.root_node.has_error:
            logger.warning("Syntax errors in '%s' — parsing best-effort.", file_path)

        chunks: list[dict] = []
        top_level_covered: set[int] = set()  # byte offsets of nodes already chunked

        for node in tree.root_node.children:
            if node.type == "function_definition":
                name = _get_name(node)
                chunks.append(_make_chunk(
                    chunk_id=_chunk_id(file_path, "func", name),
                    text=_node_text(node, source_bytes),
                    source=file_path,
                    chunk_type="function",
                    function_name=name,
                    class_name=None,
                    docstring=_extract_docstring(node, source_bytes),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))
                top_level_covered.add(node.start_byte)

            elif node.type == "class_definition":
                class_name = _get_name(node)
                top_level_covered.add(node.start_byte)
                body = next((c for c in node.children if c.type == "block"), None)
                if body is None:
                    continue
                has_methods = False
                for item in body.children:
                    if item.type == "function_definition":
                        has_methods = True
                        method_name = _get_name(item)
                        chunks.append(_make_chunk(
                            chunk_id=_chunk_id(file_path, f"class_{class_name}", method_name),
                            text=_node_text(item, source_bytes),
                            source=file_path,
                            chunk_type="method",
                            function_name=method_name,
                            class_name=class_name,
                            docstring=_extract_docstring(item, source_bytes),
                            start_line=item.start_point[0] + 1,
                            end_line=item.end_point[0] + 1,
                        ))
                    elif item.type == "decorated_definition":
                        # Handle @decorator def method():
                        inner = next((c for c in item.children if c.type == "function_definition"), None)
                        if inner:
                            has_methods = True
                            method_name = _get_name(inner)
                            chunks.append(_make_chunk(
                                chunk_id=_chunk_id(file_path, f"class_{class_name}", method_name),
                                text=_node_text(item, source_bytes),
                                source=file_path,
                                chunk_type="method",
                                function_name=method_name,
                                class_name=class_name,
                                docstring=_extract_docstring(inner, source_bytes),
                                start_line=item.start_point[0] + 1,
                                end_line=item.end_point[0] + 1,
                            ))
                if not has_methods:
                    # Data class or class with only class-level statements — chunk whole class.
                    chunks.append(_make_chunk(
                        chunk_id=_chunk_id(file_path, "class", class_name),
                        text=_node_text(node, source_bytes),
                        source=file_path,
                        chunk_type="class",
                        function_name=None,
                        class_name=class_name,
                        docstring=_extract_docstring(node, source_bytes),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))

        # Collect module-level code not covered by a function/class node.
        module_lines: list[str] = []
        module_start: int | None = None
        for node in tree.root_node.children:
            if node.start_byte in top_level_covered:
                continue
            if node.type in ("comment", "newline", "\n"):
                continue
            if module_start is None:
                module_start = node.start_point[0] + 1
            module_lines.append(_node_text(node, source_bytes))

        if module_lines:
            text = "\n".join(module_lines)
            chunks.append(_make_chunk(
                chunk_id=_chunk_id(file_path, "module", "level"),
                text=text,
                source=file_path,
                chunk_type="module_level",
                function_name=None,
                class_name=None,
                docstring=None,
                start_line=module_start or 1,
                end_line=tree.root_node.end_point[0] + 1,
            ))

        return chunks


def _get_name(node) -> str:
    name_node = next((c for c in node.children if c.type == "identifier"), None)
    return name_node.text.decode("utf-8") if name_node else "unknown"


def _make_chunk(*, chunk_id, text, source, chunk_type, function_name,
                class_name, docstring, start_line, end_line) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "page": None,
        # chunk_index is required by VectorStore.add_chunks metadata schema;
        # for code chunks start_line is a more meaningful positional marker,
        # but we still include chunk_index=0 as a no-op placeholder.
        "chunk_index": 0,
        "chunk_type": chunk_type,
        "function_name": function_name,
        "class_name": class_name,
        "docstring": docstring,
        "start_line": start_line,
        "end_line": end_line,
    }
