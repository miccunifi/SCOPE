"""Conservative static audit of intra-project imports and direct function calls."""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Finding:
    file: str
    line: int
    kind: str
    detail: str


def iter_sources(root: Path):
    for directory in ("scope", "training", "evaluation", "analysis", "tools", "tests"):
        if (root / directory).is_dir():
            yield from sorted((root / directory).rglob("*.py"))


def module_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_import(source: str, module: str | None, level: int, is_package: bool) -> str:
    if level == 0:
        return module or ""
    parent = source.split(".") if is_package else source.split(".")[:-1]
    if level - 1 > len(parent):
        return ""
    return ".".join([*parent[:len(parent) - level + 1], *(module.split(".") if module else [])])


def module_index(root: Path):
    index = {}
    for file in iter_sources(root):
        name = module_name(file, root)
        try:
            index[name] = (file, ast.parse(file.read_text(encoding="utf-8"), filename=str(file)))
        except (SyntaxError, UnicodeError) as exc:
            index[name] = (file, exc)
    return index


def exported_names(tree: ast.Module) -> set[str]:
    exported = set()
    def visit(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exported.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        exported.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        exported.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        exported.update(el.id for el in target.elts if isinstance(el, ast.Name))
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                for attr in ("body", "orelse", "finalbody"):
                    visit(getattr(node, attr, []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)
    visit(tree.body)
    return exported


def signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef):
    args = node.args
    positional = args.posonlyargs + args.args
    required = len(positional) - len(args.defaults)
    positional_names = [a.arg for a in positional]
    required_names = set(positional_names[:required])
    required_names.update(a.arg for a, default in zip(args.kwonlyargs, args.kw_defaults) if default is None)
    kw_names = set(positional_names[len(args.posonlyargs):]) | {a.arg for a in args.kwonlyargs}
    return positional_names, kw_names, required_names, args.vararg is not None, args.kwarg is not None


def bound_in_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    bound = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
    if node.args.vararg:
        bound.add(node.args.vararg.arg)
    if node.args.kwarg:
        bound.add(node.args.kwarg.arg)
    for item in ast.walk(node):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in item.names if alias.name != "*")
        elif isinstance(item, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        bound.add(sub.id)
    return bound


def validate_direct_call(node: ast.Call, definition: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
        return None
    positions, kw_names, required, varargs, varkw = signature_of(definition)
    provided = {kw.arg for kw in node.keywords}
    n = len(node.args)
    if n > len(positions) and not varargs:
        return f"{definition.name} accepts <= {len(positions)} positional args, got {n}"
    filled = set(positions[:n])
    conflict = filled & provided
    if conflict:
        return f"{definition.name} has duplicate positional/keyword args: {sorted(conflict)}"
    unexpected = provided - kw_names
    if unexpected and not varkw:
        return f"{definition.name} unexpected keyword(s): {sorted(unexpected)}"
    missing = required - filled - provided
    if missing:
        return f"{definition.name} missing required args: {sorted(missing)}"
    return None


def audit(root: Path = ROOT) -> list[Finding]:
    index = module_index(root)
    findings: list[Finding] = []
    functions = {}
    symbols = {}
    for name, (path, tree) in index.items():
        if isinstance(tree, Exception):
            findings.append(Finding(str(path.relative_to(root)), 0, "syntax", str(tree)))
            continue
        functions[name] = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        symbols[name] = exported_names(tree)
    for name, (path, tree) in index.items():
        if not isinstance(tree, ast.Module):
            continue
        file = str(path.relative_to(root))
        is_package = path.name == "__init__.py"
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = resolve_import(name, node.module, node.level, is_package)
                if not (target == "scope" or target.startswith("scope.") or target == "tools" or target.startswith("tools.")):
                    continue
                if target not in index:
                    findings.append(Finding(file, node.lineno, "missing-module", target))
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    submodule = f"{target}.{alias.name}"
                    if alias.name not in symbols.get(target, set()) and submodule not in index:
                        findings.append(Finding(file, node.lineno, "missing-symbol", f"{target}.{alias.name}"))
                    imports[alias.asname or alias.name] = (target, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if not (target == "scope" or target.startswith("scope.") or target == "tools" or target.startswith("tools.")):
                        continue
                    if target not in index:
                        findings.append(Finding(file, node.lineno, "missing-module", target))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                pass
        local_functions = functions.get(name, {})
        for top_level in tree.body:
            if not isinstance(top_level, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            bound = bound_in_function(top_level)
            for call in ast.walk(top_level):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                    continue
                called = call.func.id
                if called in bound:
                    continue
                target_fn = local_functions.get(called)
                if target_fn is None and called in imports:
                    source, symbol = imports[called]
                    target_fn = functions.get(source, {}).get(symbol)
                if target_fn is None:
                    continue
                error = validate_direct_call(call, target_fn)
                if error:
                    findings.append(Finding(file, call.lineno, "signature", error))
    return sorted(findings, key=lambda x: (x.file, x.line, x.kind))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check local imports and statically resolvable function calls.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    findings = audit(args.root.resolve())
    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2))
    else:
        for item in findings:
            print(f"{item.file}:{item.line}: {item.kind}: {item.detail}")
        print(f"Static audit: {len(findings)} issue(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
