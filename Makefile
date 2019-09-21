
install:
	pip install ./

tests:

	pytest --cov od tests/

.PHONY: tests
