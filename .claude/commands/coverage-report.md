# coverage-report

Run the full test suite with coverage and open the HTML report.

## Usage

```
/coverage-report
```

## What this runs

```bash
python -m pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  -v
```

Then summarise: overall %, any module under 80%, and which lines are uncovered.
Suggest one test per uncovered branch if coverage < 80% on any file.
