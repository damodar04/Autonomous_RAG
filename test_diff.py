import difflib

# Simulate Session State
original_protocol = """
# Protocol Title
## Objectives
To evaluate drug X.
## Design
Randomized, Double-Blind.
"""

modified_protocol = """
# Protocol Title
## Objectives
To evaluate drug X in adults.
## Design
Randomized, Double-Blind, Placebo-Controlled.
"""

print("Original Protocol:")
print(original_protocol)
print("\nModified Protocol:")
print(modified_protocol)

# Diff Logic from App
original = original_protocol.splitlines()
modified = modified_protocol.splitlines()

diff = difflib.ndiff(original, modified)

diff_text = []
for line in diff:
    if line.startswith('- '):
        diff_text.append(f":red[- {line[2:]}]")
    elif line.startswith('+ '):
        diff_text.append(f":green[+ {line[2:]}]")
    elif line.startswith('? '):
        continue
    else:
        diff_text.append(f"  {line[2:]}")

print("\nGenerated Diff Output:")
print("\n".join(diff_text))
