PYTHON ?= python3
TRACE ?= input/trace-round1.jsonl
METRICS ?= results/replay.jsonl

.PHONY: analyze score test

analyze:
	$(PYTHON) scripts/analyze_trace.py --trace $(TRACE)

score:
	$(PYTHON) scripts/score_ers.py --metrics $(METRICS)

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

