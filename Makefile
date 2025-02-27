.PHONY: docs clean

docs:
	cd docs && make html

clean:
	cd docs && make clean 