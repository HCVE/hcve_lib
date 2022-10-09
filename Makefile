test:
	PYTHONPATH=. pytest
deps:
	pip install setuptools wheel twine --upgrade
package:
	pipenv-setup sync
	python setup.py sdist bdist_wheel
	python -m twine upload dist/*

