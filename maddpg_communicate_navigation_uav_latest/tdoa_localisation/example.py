from turtle import pos
from Locator import Locator

l = Locator(
    [
        {
            'x': 18,
            'y': 40
        },
        {
            'x': 19,
            'y': 70
        },
        {
            'x': 13,
            'y': 16
        },
    ])
#这里传参的话 那么自定义初始化就无了

def getvector(pos_local:list):
    return {
            'x': pos_local[0],
            'y': pos_local[1],
        }

def getlocate():
    return l.locate(getTime())

def getTime():
    #list的话 是根据坐标的位置
    list = [0.01, 0.06, 0.08]
    return list

def getTimediffK0_1():
    list = getTime()
    return abs(list[0] - list[1])

def getTimediffK0_2():
    list = getTime()
    return abs(list[0] - list[2]);

print(l.locate(getTime()))  
#tdoa是大致求出定位点的位置  也就是Xu的坐标 那么传输时间就直接用距离求解吧
#而克拉美罗界才能评估定位性能 那么这里的传输时间差就是传入print的参数
