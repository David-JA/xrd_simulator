# custom bash script to build the docs and put them in a place that is
# visible to github pages, which unfortunately cannot see into the
# build folder.
import os
import shutil

with open(os.path.join("source", "raw_README.rst"), "r") as f:
    data = f.readlines()

new_data =[]
for i,line in enumerate(data):
    if line.strip().startswith(".. literalinclude::"):
        py_file = line.strip().split("::")[-1].strip()
        py_file = os.path.join("source",py_file)
        py_file = os.path.abspath(py_file)
        new_data.append("   .. code:: python")
        new_data.append("\n\n")
        with open(py_file, "r") as f:
            for line in f.readlines():
                if len(line.strip())>0:
                    new_data.append("      "+line)
                else:
                    new_data.append("\n")
    else:
        new_data.append(line)

with open(os.path.join("..","README.rst"), 'w') as f:
    f.writelines(new_data)

out = os.system("make html")
if out != 0:
    print("")
    print("")
    raise ValueError("Failed to build docs")
html = os.path.join("build","html")
for file in os.listdir(html):
    if not file.startswith('_'):
        shutil.copy2(os.path.join(html, file), ".")

for file in os.listdir(os.path.join(html, '_static')):
    if file.endswith('.png'):
        shutil.copy2( os.path.join(os.path.join(html, '_static') , file), "_static")
print("Copied all docs to path visible by github pages")
