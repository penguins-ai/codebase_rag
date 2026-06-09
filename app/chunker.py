# chunker.py
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
from pathlib import Path
from dataclasses import dataclass, field

CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

@dataclass
class CodeChunk:
    type: str          # "class", "method", "function", "struct"
    name: str
    class_name: str | None
    file: str
    line_start: int
    line_end: int
    content: str       # raw source text of this chunk
    namespace: str | None = None

def get_node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def get_name(node, source: bytes) -> str | None:
    if node is None:
        return None

    if node.type in {
        "identifier",
        "field_identifier",
        "type_identifier",
        "qualified_identifier",
        "destructor_name",
        "operator_name",
    }:
        return get_node_text(node, source)

    for child in node.children:
        name = get_name(child, source)
        if name:
            return name

    return None

def extract_chunks(filepath: str) -> list[CodeChunk]:
    source = Path(filepath).read_bytes()
    tree = parser.parse(source)
    chunks = []
    
    def walk(node, current_class=None):
        if node.type == "class_specifier":
            name_node = node.child_by_field_name("name")
            class_name = get_node_text(name_node, source) if name_node else "anonymous"
            
            chunks.append(CodeChunk(
                type="class",
                name=class_name,
                class_name=None,
                file=filepath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                content=get_node_text(node, source)[:3000],  # cap very large classes
            ))
            
            # recurse into class body with class context
            for child in node.children:
                walk(child, current_class=class_name)
            return  # don't double-recurse below

        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            name = get_name(declarator, source) if declarator else "unknown"

            inferred_class = current_class
            if not inferred_class and name and "::" in name:
                inferred_class = name.split("::")[0]

            chunk_type = "method" if inferred_class else "function"

            chunks.append(CodeChunk(
                type=chunk_type,
                name=name or "unknown",
                class_name=inferred_class,
                file=filepath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                content=get_node_text(node, source)[:2000],
            ))

        if node.type == "struct_specifier":
            name_node = node.child_by_field_name("name")
            struct_name = get_node_text(name_node, source) if name_node else "anonymous"
            chunks.append(CodeChunk(
                type="struct",
                name=struct_name,
                class_name=current_class,
                file=filepath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                content=get_node_text(node, source)[:1500],
            ))

        for child in node.children:
            walk(child, current_class=current_class)
    
    walk(tree.root_node)
    return chunks