# Task 3: Lockfile and reproducibility
# TODO: Generate uv.lock
# TODO: Explain in 3-5 comments why lockfiles matter for teams


# CMD write : uv lock = generate uv.lock


# Explain why lockfiles matter (3–5 comments)

# Lockfiles ensure everyone on the team installs the exact same dependency versions
# This prevents "works on my machine" issues caused by version drift
# They make builds and experiments reproducible across time and environments
# CI/CD pipelines rely on lockfiles to guarantee consistent installs
# Debugging is easier because dependency changes are explicit and tracked

