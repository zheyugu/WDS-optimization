import xlrd
import os
import openpyxl
import pandas as pd
from openpyxl import load_workbook

os.chdir("C:/Users/12757/Desktop/Columbia/M.S. Thesis/Optimization")

wb = load_workbook(filename='24hr quality.xlsx')
ws = wb.active

quality = {}
for column in list(ws.columns)[1:]:
    quality [column[0].value] = [(c.value) for c in column[1:]][0]

print(quality)
# print(quality["59146264"])
# df = pd.DataFrame(quality,index=[0]).T  # transpose to look just like the sheet above
# df.to_excel('file.xls')

wb = load_workbook(filename='24hr flowrate.xlsx')
ws = wb.active

flowrate = {}
for column in list(ws.columns)[1:]:
    flowrate [column[1].value,column[2].value] = [(c.value) for c in column[3:]][0]

# df = pd.DataFrame(flowrate,index=[0]).T  # transpose to look just like the sheet above
# df.to_excel('file.xls')
# print(flowrate)
# print(flowrate[(59104431, 'reservoir')])

# combine positive and newnegative dictionaries
def Merge(positive, newnagative): 
    return(newnagative.update(positive)) 

# extract negative flowrate into a new dic
negative = dict((k, v) for k, v in flowrate.items() if v < 0)

positive = dict((k, v) for k, v in flowrate.items() if v > 0)

x = []
y = []
v1 = []

for k,v in flowrate.items():
    if v < 0:
        x.append(k[1])
        y.append(k[0])
        # v1.append(v[0])
        # new = dict((merge(x,y), -v)
        v1.append(-v)

new_negative = dict(zip(zip(x, y),v1))

Merge(positive, new_negative)
new_flowrate = new_negative 
print(new_flowrate)

