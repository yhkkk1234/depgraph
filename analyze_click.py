import click, os, re

proj = os.path.dirname(click.__file__)

# Get functions from utils.py
utils_code = open(os.path.join(proj, "utils.py"), encoding="utf-8").read()
funcs = re.findall(r"^def (\w+)", utils_code, re.M)
print("Functions in utils.py:")
for f in funcs:
    print(f"  {f}")

# Find which files call these functions
print("\nCross-file usage of utils.py functions:")
for pyfile in sorted(os.listdir(proj)):
    if not pyfile.endswith(".py") or pyfile in ("utils.py", "__init__.py"):
        continue
    code = open(os.path.join(proj, pyfile), encoding="utf-8").read()
    used = []
    for func in funcs:
        if func in code:
            used.append(func)
    if used:
        print(f"  {pyfile}: {', '.join(used)}")
