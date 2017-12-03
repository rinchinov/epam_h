import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

paths_to_data = {"crime": 'data/MVD_News/news_with_location.csv',
                 "problems": "data/GorodGovSpb/GorodGovSpb.csv",
                 "art": 'data/OpenStreetMap/objects.csv',
                 "amenities": "data/OpenStreetMap/amenities.csv"}
                 #"buildings": 'data/OpenStreetMap/buildings_data.csv',}
                 #"wifi": 'wifi_data.csv'}


object_rate = {'fountain': 10,
               'bench': 9,
               'place_of_worship': 8,
               'toilets': 7,
               'bus_station': 6,
               'theatre': 7,
               'bicycle_parking': 6,
               'payment_terminal': 6,
               'atm': 5,
               'wifi': 0}
amenities_rate = {'restaurant': 6,
                  'waste_disposal': 5,
                  'bicycle_parking': 6,
                  'ice_cream': 7,
                  'vehicle_ramp': 5,
                  'drinking_water': 6,
                  'bicycle_rental': 6}
category_rate = {'Фасад': -6,
                 'Благоустройство': -5,
                 'Повреждения или неисправность элементов уличной инфраструктуры': -3,
                 'Незаконная информационная и (или) рекламная конструкция': -4,
                 'Санитарное состояние': -5}


def crime_type(name):
    if "краж" in name.lower():
        return -5
    elif "граб" in name.lower():
        return -7
    elif "убийство" in name.lower():
        return -10
    elif "простит" in name.lower():
        return -2
    elif "уголовный" in name.lower():
        return -1
    elif "уголовно-процессуальный" in name.lower():
        return -6
    elif "Кодекс Об Административных Правонарушениях".lower() in name.lower():
        return -3
    else:
        return 0


def problem_type(type_):
    try:
        return category_rate[type_]
    except KeyError:
        return -1


def art_type(type_):
    return object_rate[type_]


def amenities_type(type_):
    try:
        return amenities_rate[type_]
    except KeyError:
        return 0


def wifi_type():
    return 0


def count_rate(data_, radius_, lat_, lng_):
    tr = data_[(data_.lat <= lat_ + radius_) & (lat_ - radius_ <= data_.lat) & (data_.lng <= lng_ + radius_) & (
        lng_ - radius_ <= data_.lng)]
    if tr.rate.count() == 0:
        return 0
    else:
        x = tr.lat.values - np.ones(tr.lat.count())*lat_
        y = tr.lng.values - np.ones(tr.lng.count())*lng_

        return np.dot(tr.rate.values, np.exp(x**2 + y**2))


def calculate_tables(left_top_corner_lat=59.90, right_bottom_corner_lat=59.63, left_top_corner_lng=30.42,
                     right_bottom_corner_lng=30.72):
    y_step = (left_top_corner_lat - right_bottom_corner_lat)/100
    x_step = -(left_top_corner_lng - right_bottom_corner_lng)/100
    radius = 5 * (y_step + x_step)/2
    heat_map_image = {type_: np.ndarray(shape=(100, 100), dtype=float) for type_ in paths_to_data.keys()}
    data = {type_: pd.read_csv(paths_to_data[type_]) for type_ in paths_to_data.keys()}

    data["art"]['rate'] = data["art"].type.apply(art_type)
    data["crime"]['rate'] = data["crime"].decreepart.apply(crime_type)
    data["amenities"]['rate'] = data["amenities"].amenity.apply(amenities_type)
    data["problems"]['rate'] = data["problems"].category.apply(problem_type)

    heat_map_image["json"] = {
                              "type": "FeatureCollection",
                              "features": []
    }

    x_line = np.linspace(right_bottom_corner_lat, left_top_corner_lat, 101)
    y_line = np.linspace(left_top_corner_lng, right_bottom_corner_lng, 101)
    x, y = 0, 0
    for i in range(10000):
        pol_coord = [[x_line[x], y_line[y]], [x_line[x + 1], y_line[y]],
                     [x_line[x + 1], y_line[y + 1]], [x_line[x], y_line[y + 1]], [x_line[x], y_line[y]]]
        for retr in pol_coord:
            retr.reverse()

        heat_map_image["json"]["features"].append({"id": i,
                                                   "type": "Feature",
                                                   "geometry": {
                                                       "type": "Polygon",
                                                       "coordinates": [pol_coord]
                                                    },
                                                   "properties": {'highlight': {}, 'name': i, 'style': {}}
                                                   })
        if x < 99:
            x += 1
        else:
            x = 0
            y += 1

    for i in range(100):
        for j in range(100):
            lat = right_bottom_corner_lat + y_step*i + y_step/2
            lng = left_top_corner_lng + x_step*j + x_step/2
            for type_ in heat_map_image.keys():
                if type_ != "json":
                    heat_map_image[type_][i][j] = count_rate(data[type_], radius, lat, lng)
    prefix = "[({},{})-({},{})]".format(int(left_top_corner_lat*1000), int(left_top_corner_lng*1000),
                                        int(right_bottom_corner_lat*1000), int(right_bottom_corner_lng*1000))
    pickle.dump(heat_map_image, open("dump/heat_map_image{}".format(prefix), "wb"))

    for type_ in heat_map_image.keys():
        if type_ != 'json':
            plt.imshow(heat_map_image[type_])
            plt.savefig("images/{}{}".format(type_, prefix))


if __name__ == "__main__":
    try:
        calculate_tables(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except IndexError:
        calculate_tables(left_top_corner_lat=59.90, right_bottom_corner_lat=59.83, left_top_corner_lng=30.12,
                         right_bottom_corner_lng=30.22)

