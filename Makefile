test:
	PYTHONPATH=. pytest --color=yes | less -R
package:
	pipenv-setup sync
mypy:
	mypy .

