import nbformat


def md(src: str):
    return nbformat.v4.new_markdown_cell(src.strip("\n"))


def code(src: str):
    return nbformat.v4.new_code_cell(src.strip("\n"))
