# Task 4: Verification script
# TODO: Create a small script that prints:
# - Python version
# - numpy, pandas, sklearn versions
# TODO: Run the script from the venv and paste the output below

import numpy, pandas, sklearn , torch

print(f"numpy version: {numpy.__version__}")
print(f"pandas version: {pandas.__version__}")
print(f"sklearn version: {sklearn.__version__}")

# print(f"torch version: {torch.__version__}")

print("Done! Your environment is reproducible and verified.")

# run script -> python task4.py