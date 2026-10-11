"""h3_sm75_server 回归自检（从 tests/ 目录直接跑）。

跑法（需要有 torch 的解释器；本地可用 ~/comfyui/.venv）：
  cd h3_sm75_server/tests && python _check_call_sites.py

背景：2026-10-11 的两次线上启动失败（self._install_attention
找不到 / bundle 缺 keep_ratio）都是"静态 import+compileall
挡不住、真执行才炸"的类型，所以这些脚本必须真实执行到关键
路径，不能只 import。
"""

import ast
import io
import os
import sys

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dit_svc", "runtime.py")


def check(path=PATH) -> int:
    with io.open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    runtime = next((n for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == "Runtime"), None)
    if runtime is None:
        print("FATAL: class Runtime 未找到")
        return 2

    defined = set()
    for node in ast.walk(runtime):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and isinstance(getattr(t, "ctx", None), ast.Store):
                    defined.add(t.id)
                elif isinstance(t, ast.Attribute) and isinstance(t.ctx, ast.Store) \
                        and isinstance(t.value, ast.Name) and t.value.id == "self":
                    defined.add(t.attr)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute) \
                and isinstance(node.target.ctx, ast.Store) \
                and isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
            defined.add(node.target.attr)

    bad = []
    for node in ast.walk(runtime):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self" and node.attr.startswith("_") \
                and isinstance(node.ctx, ast.Load):
            if node.attr not in defined:
                bad.append((node.lineno, node.attr))
    if bad:
        print("FATAL: self._X 但 X 不在 Runtime 内定义: %s" % bad)
        return 3

    # 反向：模块级 def 里不应出现 self._X（模块函数没有 self）
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "self":
                print("FATAL: 模块级函数 %s() 第 %d 行用了 self.%s（没有 self）"
                      % (fn.name, node.lineno, node.attr))
                return 4

    # ---- 第三次线上事故（2026-10-11）：接口签名不一致 ----
    # `veda_star7.install` 返回 2 元组，runtime 里却解包 3 个 -> 线上
    # ValueError: too many values to unpack（DiT 加载 124s 之后）。
    # 静态检查：两个文件的元组长度必须一致。
    veda_path = os.path.join(os.path.dirname(PATH), "veda_star7.py")
    with io.open(veda_path, encoding="utf-8") as f:
        veda_tree = ast.parse(f.read())
    install_fn = next(n for n in veda_tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "install")
    returns = [n for n in ast.walk(install_fn) if isinstance(n, ast.Return)
               and n.value is not None]
    arity = set()
    for r in returns:
        arity.add(len(r.value.elts) if isinstance(r.value, ast.Tuple) else 1)
    # 注意：调用点在**模块级函数** `_install_attention` 里（不是 Runtime 方法），
    # 所以这里必须 walk 整个模块树而不是只 walk class Runtime。
    call_arities = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr == "install" \
                and isinstance(node.value.func.value, ast.Name) \
                and node.value.func.value.id == "veda_star7":
            t = node.targets[0]
            call_arities.add(len(t.elts) if isinstance(t, ast.Tuple) else 1)
    if not call_arities:
        print("FATAL: runtime.py 里找不到 veda_star7.install(...) 调用点")
        return 7
    if arity and call_arities and arity != call_arities:
        print("FATAL: veda_star7.install 返回 %s 元组，runtime 解包 %s 个"
              % (sorted(arity), sorted(call_arities)))
        return 6
    print("ok: veda_star7.install 签名一致（return %s / 解包 %s）"
          % (sorted(arity), sorted(call_arities)))

    methods = sorted(n.name for n in ast.walk(runtime)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    print("OK runtime.py 调用点一致（Runtime methods: %s）" % methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
