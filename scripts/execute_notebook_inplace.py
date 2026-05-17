"""Execute a simple project notebook and save visible outputs in-place.

This avoids depending on nbconvert/nbclient, which may not be installed in
lightweight local environments.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import traceback
from pathlib import Path
from types import CodeType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Jomana_BM25_Retrieval.ipynb"


def rich_output(value: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"text/plain": repr(value)}
    if hasattr(value, "_repr_html_"):
        html = value._repr_html_()
        if html:
            data["text/html"] = html
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": data,
    }


def split_last_expression(source: str) -> tuple[CodeType, ast.expr | None]:
    parsed = ast.parse(source)
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        last_expression = parsed.body.pop()
        setup_code = compile(parsed, "<notebook-cell>", "exec")
        return setup_code, last_expression.value
    return compile(parsed, "<notebook-cell>", "exec"), None


def main() -> None:
    os.chdir(PROJECT_ROOT)
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
    }

    execution_count = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        execution_count += 1
        cell["execution_count"] = execution_count
        outputs: list[dict[str, Any]] = []

        def display(*values: Any) -> None:
            for value in values:
                outputs.append(rich_output(value))

        namespace["display"] = display
        source = "".join(cell.get("source", []))
        stdout = io.StringIO()

        try:
            setup_code, last_expression = split_last_expression(source)
            with contextlib.redirect_stdout(stdout):
                exec(setup_code, namespace)
                if last_expression is not None:
                    value = eval(compile(ast.Expression(last_expression), "<notebook-cell>", "eval"), namespace)
                    if value is not None:
                        outputs.append(
                            {
                                "output_type": "execute_result",
                                "execution_count": execution_count,
                                "metadata": {},
                                "data": rich_output(value)["data"],
                            }
                        )
        except Exception as exc:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exception(exc),
                }
            )

        text = stdout.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text})
        cell["outputs"] = outputs

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Executed notebook and saved outputs: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
