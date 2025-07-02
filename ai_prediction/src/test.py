import datetime

# Calculate number of days from 2017-8-17 to today
today = datetime.date.today()
start_date = datetime.date(2017, 8, 17)
delta = today - start_date
print(f"Number of days from {start_date} to {today}: {delta.days}")
