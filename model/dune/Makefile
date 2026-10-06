export YAMLFIX_INDENT_MAPPING := 4
export YAMLFIX_INDENT_OFFSET := 4
export YAMLFIX_INDENT_SEQUENCE := 4


format:
	find . -type f -name "*.py" | xargs black

	find . -type f -name "*.py" | xargs isort \
		--profile black \
		--atomic \
		--star-first \
		--only-sections \
		--order-by-type \
		--use-parentheses \
		--lines-after-imports=2 \
		--known-local-folder=model,data,teachers,utils


clean:
	# Remove __pycache__ folders
	find . -type d -name "__pycache__" -print0 -exec rm -rf {} \;


test:
	python -m unittest discover -s tests -p "test_*.py"


conda:
	conda env export > environment.yaml