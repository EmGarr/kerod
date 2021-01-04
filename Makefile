
install:
	pip install ./

tests:
	pytest --cov kerod tests/

.PHONY: tests
