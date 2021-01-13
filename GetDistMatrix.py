from collections import OrderedDict
import json
import requests
from tasks import mapquest_key # TODO: необходимо сделать файл с конфигом
import pandas
from openpyxl import load_workbook
import sys

path = u"{}".format(sys.argv[1])

places = pandas.read_excel(path)

in_data = list()

# Удобнее из датасета

for building in places.index:
    place = OrderedDict([("adminArea1Type", "Country"),
    ("adminArea1", u"{}".format(places.iloc[building,]['Страна (по ISO-3166-1) Alpha 2'])),
    ("adminArea2", u"{}".format(places.iloc[building,]['Федеральный округ'])),
    ("adminArea3", u"{}".format(places.iloc[building,]['Субъект'])),
    ("adminArea4", u"{}".format(places.iloc[building,]['Территориальная единица (район)'])),
    ("adminArea5", u"{}".format(places.iloc[building,]['Населённый пункт'])),
    ("street", u"{} {} {}".format(
        places.iloc[building,]['Номер строения'],
        places.iloc[building,]['Название улицы/шоссе/проезда'],
        places.iloc[building,]['Тип (улица/шоссе/проезд)']
        )
    )])
    in_data.append(place)

response = requests.post(f"http://www.mapquestapi.com/directions/v2/routematrix?key={mapquest_key}",
json= {"locations" : in_data,
"options": {
    "allToAll": "true",
    "manyToOne": "true"
}})

to_use = json.loads(response.text)

ds = pandas.DataFrame(
    to_use["distance"], 
    columns = [f"{places.iloc[building,]['Тип (улица/шоссе/проезд)']} {places.iloc[building,]['Название улицы/шоссе/проезда']} {places.iloc[building,]['Номер строения']}" for building in places.index], 
    index = [f"{places.iloc[building,]['Тип (улица/шоссе/проезд)']} {places.iloc[building,]['Название улицы/шоссе/проезда']} {places.iloc[building,]['Номер строения']}" for building in places.index]
    )

book = load_workbook(path)
writer = pandas.ExcelWriter(path, engine = 'openpyxl')
writer.book = book

ds.to_excel(writer, sheet_name = 'Матрица Расстояний')
writer.save()
writer.close()

# python "D:\py fi\TSP_try\GetDistMatrix.py" "D:\new_file.xlsx"