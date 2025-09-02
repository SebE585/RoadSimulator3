.PHONY: install dev test simulate clean

install:
\tpython -m pip install -r requirements.txt

dev:
\tpython -m pip install -r requirements.txt -r requirements-dev.txt

test:
\tpytest

simulate:
\tpython -m runner.run_simulation --config config/example.yaml

clean:
\trm -rf logs outputs .pytest_cache __pycache__
