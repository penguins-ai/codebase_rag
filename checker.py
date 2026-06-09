import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

def node_to_sexp(node, indent=0):
    result = " " * indent + f"({node.type}"
    if node.child_count == 0:
        result += f' "{node.text.decode()}"'
    result += ")"
    for child in node.children:
        result = result[:-1] + "\n" + node_to_sexp(child, indent + 2) + ")"
    return result


CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

code = b"class Foo { void bar() {} };"
tree = parser.parse(code)
print(node_to_sexp(tree.root_node))