prediction:
	pip freeze | grep -v timeseries > requirements.txt
	pip install -e .
	pip install -r requirements.txt
	fetch_raw_data --ticker JPM
	format_timeseries
	predict_tomorrow
