import pandas as pd


def delay_percentage():
    count_total_flights = 0
    count_delayed_departure = 0
    count_delayed_arrival = 0
    count_cancelled = 0

    for i in range(1, 13):
        cur_file = pd.read_csv('./Jan-Dec/' + str(i) + '.csv')
        columns = cur_file[["DEP_DELAY_NEW", "ARR_DELAY_NEW", "CANCELLED"]]
        cur_list = columns.values.tolist()
        for [departure_delayed, arrival_delayed, cancelled] in cur_list:
            if cancelled == 1:
                count_cancelled += 1
                count_total_flights += 1
                continue
            if departure_delayed >= 15:
                count_delayed_departure += 1
            if arrival_delayed >= 15:
                count_delayed_arrival += 1
            count_total_flights += 1
        break

    return [count_total_flights, count_cancelled, count_delayed_departure, count_delayed_arrival]


# def get_count_total_flights():
#     return delay_percentage()[0]
#
#
# def get_count_cancelled():
#     return delay_percentage()[1]
#
#
# def get_count_delayed_departure():
#     return delay_percentage()[2]
#
#
# def get_count_delayed_arrival():
#     return delay_percentage()[3]


print(delay_percentage())

