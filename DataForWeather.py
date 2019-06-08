import pandas as pd


def get_departure_airports():
    airports_dict = set([])
    for i in range(1, 13):
        cur_file = pd.read_csv('./Jan-Dec/' + str(i) + '.csv')
        columns = cur_file[["ORIGIN", "ORIGIN_STATE_ABR"]]
        cur_list = columns.values.tolist()
        for [airport, state] in cur_list:
            airports_dict.add((airport, state))
    IATA_codes = list(airports_dict)
    IATA_codes.sort()
    return IATA_codes


def get_index(code):
    IATA_codes = get_departure_airports()
    for i in range(len(IATA_codes)):
        if IATA_codes[i][0] == code:
            return i
    return -1


# print(get_departure_airports())
# print(get_index('TLH'))
