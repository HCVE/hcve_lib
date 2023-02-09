test:
	PYTHONPATH=. pytest
deps:
	pip install setuptools wheel twine --upgrade
package:
	rm -r ./dist 2> /dev/null || true
	pipenv-setup sync
	python setup.py sdist bdist_wheel
	python -m twine upload dist/*

