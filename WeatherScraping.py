from datetime import datetime, timedelta
import DataForWeather
import urllib.request
import json


def weather_crawler(IATA_code, state, f):
    cur_date = datetime(year=2018, month=3, day=1)
    end_date = datetime(year=2018, month=3, day=31)
    while cur_date <= end_date:
        request = urllib.request.urlopen('http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=990e2a68aa0f46fdbeb211046190705&q=' + IATA_code + '&format=json&date=' + str(cur_date.year) + "-" + str(cur_date.month) + "-" + str(cur_date.day))
        response = request.read()
        json_file = json.loads(response)
        extraction_from_json(json_file, f, IATA_code, state)
        cur_date += timedelta(days=1)
    return


def extraction_from_json(json_file, f, IATA_code, state):
    date = json_file['data']['weather'][0]['date']
    aver_temp = (float(json_file['data']['weather'][0]['maxtempC']) + float(json_file['data']['weather'][0]['mintempC'])) / 2
    total_snow = float(json_file['data']['weather'][0]['totalSnow_cm'])
    hourly_dict = json_file['data']['weather'][0]['hourly']
    wind_speed = 0
    precipitation = 0
    visibility = 0
    length = 0

    for each_hour in hourly_dict:
        wind_speed += float(each_hour['windspeedKmph'])
        precipitation += float(each_hour['precipMM'])
        visibility += float(each_hour['visibility'])
        length += 1
    aver_wind_speed = wind_speed / length
    aver_visibility = visibility / length

    f.write(IATA_code + ',' + state + ',' + date + ',' + two_decimal_places(aver_temp) + ',' + two_decimal_places(total_snow) + ',' + two_decimal_places(aver_wind_speed) + ',' + two_decimal_places(precipitation) + ',' + two_decimal_places(aver_visibility) + '\n')


def two_decimal_places(num):
    num = str(num)
    index = num.find('.')
    if index == -1 or index == len(num) - 2 or index == len(num) - 3:    # Integer or 1-digit or 2-digit
        return num
    else:       # More than 3 digits after '.'
        if '5' <= num[index + 3] <= '9':
            return num[: index + 2] + str(int(num[index + 2]) + 1)
        else:
            return num[: index + 3]


f = open('weather_data.csv', 'w')
# f.write('IATA_code, state, date, aver_temp, total_snow, aver_wind_speed, precipitation, aver_visibility' + '\n')
airports_list = DataForWeather.get_departure_airports()
airports_list = airports_list[335:]

for (IATA_code, state) in airports_list:
    weather_crawler(IATA_code, state, f)
    # break
