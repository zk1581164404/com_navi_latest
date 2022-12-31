
def getlist():
    uav_list = []
    for line in open('/root/com_navi_latest/maddpg_communicate_navigation_uav_latest/maddpg-master/experiments/uav_pos.txt',"r"):    #设置文件对象
        xy_pos = []
        xy_pos.append(line[:-1])  #去除换行
        uav_list.append(xy_pos)
    return uav_list