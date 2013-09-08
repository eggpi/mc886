from datetime import datetime
from dateutil.relativedelta import relativedelta

USEC_PER_SEC = 1e6
EPOCH = datetime(1970, 1, 1)
SHIFTED_EPOCH = datetime(2013, 1, 1)
END_OF_THE_WORLD = SHIFTED_EPOCH + relativedelta(years = 1)

SHIFTED_EPOCH_US = USEC_PER_SEC * (SHIFTED_EPOCH - EPOCH).total_seconds()
END_OF_THE_WORLD_US = USEC_PER_SEC * (END_OF_THE_WORLD - EPOCH).total_seconds()

ONE_HOUR_US = 3600 * USEC_PER_SEC
ONE_DAY_US = 24 * ONE_HOUR_US
ONE_WEEK_US = 7 * ONE_DAY_US

now_us = 1378599212143689.0

timestamp_us = 1378580457713224.0
timestamp_shifted_us = timestamp_us - SHIFTED_EPOCH_US
timestamp_dt = datetime.fromtimestamp(timestamp_us / USEC_PER_SEC)

some_hours_later_dt = timestamp_dt + relativedelta(hours = 6)
one_year_later_dt = timestamp_dt + relativedelta(years = 1)
one_month_later_dt = timestamp_dt + relativedelta(months = 1)
one_week_later_dt = timestamp_dt + relativedelta(weeks = 1)

some_hours_later_us = USEC_PER_SEC * (some_hours_later_dt - EPOCH).total_seconds()

one_year_later_us = USEC_PER_SEC * (one_year_later_dt - EPOCH).total_seconds()
one_year_later_shifted_us = one_year_later_us - SHIFTED_EPOCH_US

one_month_later_us = USEC_PER_SEC * (one_month_later_dt - EPOCH).total_seconds()
one_month_later_shifted_us = one_month_later_us - SHIFTED_EPOCH_US

one_week_later_us = USEC_PER_SEC * (one_week_later_dt - EPOCH).total_seconds()
one_week_later_shifted_us = one_week_later_us - SHIFTED_EPOCH_US

original_year_ratio = timestamp_us / one_year_later_us
week_year_ratio = one_week_later_us / one_year_later_us

print 'Original timestamp: {}'.format(timestamp_us)
print 'One week later: {}'.format(one_week_later_us)
print 'One year later: {}'.format(one_year_later_us)
print
print 'now - original: {}'.format(now_us - timestamp_us)
print 'original / one year later: {}'.format(original_year_ratio)
print 'one week later / one year later: {}'.format(week_year_ratio)
print
print 'original in days: {}'.format(timestamp_us / ONE_DAY_US)
print 'one week later in days: {}'.format(one_week_later_us / ONE_DAY_US)
print
print 'original in days, shifted epoch {}'.format(
        timestamp_shifted_us / ONE_DAY_US)
print '(original - shifted) / (end - shifted): {}'.format(
        timestamp_shifted_us / (END_OF_THE_WORLD_US - SHIFTED_EPOCH_US))
print '(week - shifted) / (end - shifted): {}'.format(
        one_week_later_shifted_us / (END_OF_THE_WORLD_US - SHIFTED_EPOCH_US))
print '(month - shifted) / (end - shifted): {}'.format(
        one_month_later_shifted_us / (END_OF_THE_WORLD_US - SHIFTED_EPOCH_US))
print '(year - shifted) / (end - shifted): {}'.format(
        one_year_later_shifted_us / (END_OF_THE_WORLD_US - SHIFTED_EPOCH_US))
print
print 'original, days after beginning of the year: {}'.format(
        (USEC_PER_SEC * (timestamp_dt - datetime(2013, 1, 1)).total_seconds()) / ONE_DAY_US)
print 'week, days after beginning of the year: {}'.format(
        (USEC_PER_SEC * (one_week_later_dt - datetime(2013, 1, 1)).total_seconds()) / ONE_DAY_US)
print 'hours, days after beginning of the year: {}'.format(
        (USEC_PER_SEC * (some_hours_later_dt - datetime(2013, 1, 1)).total_seconds()) / ONE_DAY_US)

"""
Conclusion: makes sense to use days within the year as normalized timestamp, as
it (1) does not depend on current time, and (2) might not dominate the number of
hits too much
"""
