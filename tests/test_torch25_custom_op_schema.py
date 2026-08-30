'''Regression guard for PyTorch 2.5 torch.library schema inference.'''

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'h3_optimizations'


def _is_custom_op_decorator(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return isinstance(target, ast.Attribute) and target.attr == 'custom_op'


def _uses_builtin_list_generic(annotation):
    if annotation is None:
        return False
    return any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'list'
        for node in ast.walk(annotation)
    )


class Torch25CustomOpSchemaTests(unittest.TestCase):
    def test_custom_op_inputs_avoid_pep585_list_annotations(self):
        '''PyTorch 2.5 infer_schema accepts typing.List[T], not list[T].'''
        offenders = []
        for path in PACKAGE.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not any(_is_custom_op_decorator(d) for d in node.decorator_list):
                    continue
                arguments = (
                    list(node.args.posonlyargs)
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                )
                for argument in arguments:
                    if _uses_builtin_list_generic(argument.annotation):
                        offenders.append(
                            '%s:%d %s.%s'
                            % (
                                path.relative_to(ROOT),
                                argument.lineno,
                                node.name,
                                argument.arg,
                            )
                        )

        self.assertEqual(
            offenders,
            [],
            'PyTorch 2.5 torch.library.infer_schema rejects PEP 585 list[T] '
            'annotations on custom-op inputs; use typing.List[T] instead',
        )


if __name__ == '__main__':
    unittest.main()
