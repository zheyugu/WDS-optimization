import xlrd
import os
import openpyxl
import pandas as pd
from openpyxl import load_workbook
from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
from itertools import product

os.chdir("C:/Users/12757/Desktop/Columbia/M.S. Thesis/Optimization")

wb = load_workbook(filename='24hr quality.xlsx')
ws = wb.active

quality = {}
for column in list(ws.columns)[1:]:
    quality [column[0].value] = [(c.value) for c in column[1:]][0]

# print(quality)
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
# print(new_flowrate)

wb = load_workbook(filename='elevation.xlsx')
ws = wb.active

elevation = {}
for column in list(ws.columns)[1:]:
    elevation [column[0].value] = [(c.value) for c in column[1:]][0]


# create a list with node
node = pd.read_csv('node.csv', header=None)
node.values.T[0].tolist()

node = list(node[0])

node[269] = 1

for i in range(0, len(node)):
    node[i] = int(node[i])

node[269] ='reservoir'

# create a list with facility
facility = [59101483, 59104418, 59097687, 59082686, 59140961]

# link = list(flowrate.keys())
# facility = []
# node = []
# for i in range(len(link)):
#     facility.append(link[i][1])
#     node.append(link[i][0])

# create a model
m = gp.Model('Facility Placement')

link,cl = gp.multidict(flowrate)
# print(link)

# create variables
x = m.addVars(facility,node,vtype=GRB.BINARY, name="x")
y = m.addVars(facility,vtype=GRB.BINARY, name="y")
cap = m.addVars(facility,vtype=GRB.CONTINUOUS, name="cap")

# set parameter value
# gpd for D, CAP_MAX, CAP_MIN,    $ for FC  $/gpd for TC
D = []
CAP_MAX = []
CAP_MIN = []
FC = [100000, 110000, 120000, 130000, 140000]
TC = [100, 90, 80, 70, 60]
for i in range(290):
    D.append(22800)
    
for i in range(5):
    CAP_MAX.append(10000000)
    CAP_MIN.append(0)
    # FC.append(100000)
    # TC.append(100)
    
D = dict(zip(node,D))
CAP_MAX = dict(zip(facility,CAP_MAX))
CAP_MIN = dict(zip(facility,CAP_MIN))
FC = dict(zip(facility,FC))
TC = dict(zip(facility,TC))
cl =  {key: value * 1000 for key, value in quality.items()}

# create constraints
m.addConstrs((gp.quicksum(x[i,j] for i in facility) == 1 for j in node), name='Demand')
m.addConstrs((gp.quicksum(D[j] * x[i,j] for j in node) <= cap[i] for i in facility), 
               name='Facility Capacity') 
m.addConstr((gp.quicksum(D[j] for j in node) == gp.quicksum(cap[i] for i in facility)), 
               name='Equal Total Capacity and Demand')
m.addConstrs((CAP_MAX[i] * y[i] >= cap[i] for i in facility), name='MAX Facility Capacity')
m.addConstrs((cap[i] >= CAP_MIN[i] * y[i] for i in facility), name='MIN Facility Capacity')

# create objective
# link = list(flowrate.keys())
# facility = []
# node = []
# for i in range(len(link)):
#     facility.append(link[i][1])
#     node.append(link[i][0])
# m.setObjective((gp.quicksum(x[i,j] * cl[i,j] for i,j in link)),GRB.MAXIMIZE) 
# m.setObjective((gp.quicksum(x[i,j] * 1 for i,j in link)),GRB.MAXIMIZE)   
m.setObjective((gp.quicksum(y[i] * FC[i] for i in facility) 
                + gp.quicksum(cap[i] * TC[i]  for i in facility)),GRB.MINIMIZE)

# Run optimization engine
m.optimize()
       
# #Analysis
# facility_placement = pd.DataFrame(columns=["Node", "Facility"])

# count = 0

# for j in node:
#     for i in facility:
#         if(x[i,j].x > 0.5):
#             count += 1
#             facility_placement = facility_placement.append({"Node": j, "Facility": i }, ignore_index=True )
# facility_placement.index=['']*count
# print(facility_placement)

status = m.status
if status == GRB.Status.OPTIMAL:
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)
       
elif status == GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)
    # do IIS
    m.computeIIS()
    m.write("m.ilp")
    gp.read("m.ilp")
    for c in m.getConstrs():
        if c.IISConstr:
            print('%s' % c.constrName)