test:
	PYTHONPATH=. pytest
deps:
	pip install setuptools wheel twine --upgrade

package:
	poetry build
	poetry publish
