import pandas as pd


def get_holidays_dictionary():
    return set(['2018-01-01', '2018-01-02', '2018-01-12', '2018-01-13', '2018-01-14', '2018-01-15', '2018-02-16', '2018-02-17', '2018-02-18', '2018-02-19', '2018-05-25', '2018-05-26', '2018-05-27', '2018-05-28', '2018-07-04', '2018-08-31', '2018-09-01', '2018-09-02', '2018-09-03', '2018-11-09', '2018-11-10', '2018-11-11', '2018-11-12', '2018-11-21', '2018-11-22', '2018-11-23', '2018-11-24', '2018-11-25', '2018-12-23', '2018-12-24', '2018-12-25', '2018-12-26', '2018-12-27', '2018-12-28', '2018-12-29', '2018-12-30', '2018-12-31'])


def is_a_holiday(s):
    holiday_dict = get_holidays_dictionary()
    if s in holiday_dict:
        return True
    else:
        return False


if __name__ == '__main__':
    d = get_holidays_dictionary()
    path = './1_Weather included.csv'
    df = pd.read_csv(path, index_col=False)
    holiday = []
    for i in range(len(df)):
        if df['FL_DATE'][i] in d:
            holiday.append(1)
        else:
            holiday.append(0)
    df['HOLIDAY'] = holiday
    df.to_csv('./Jan-data-updated.csv', index=False)