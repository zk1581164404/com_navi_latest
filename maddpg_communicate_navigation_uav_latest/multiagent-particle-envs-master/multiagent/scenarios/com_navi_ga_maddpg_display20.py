from dis import dis
from sqlite3 import connect
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
from multiagent.scenarios.Locator import Locator #奇怪  必须要绝对路径
import math
from numpy.random import default_rng
import json
import copy
import math
import datetime

#不同阶段  不同速度   不同阶段的话 用类似状态机的转移即可
#todo 加上到达success的时间
#先搞定一下输入吧 
#地图是 -1*1 --> 500*500  相当于乘250 + 250
#出结果的时候考虑一下距离  尤其是前几组数据  宁愿迟一点交
#先从最简单的  不更变角色开始部署吧
#开始处理输出问题

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Scenario(BaseScenario): #在reset的时候 修改用户比例
    f = open('/root/com_navi_latest/maddpg_communicate_navigation_uav_latest/multiagent-particle-envs-master/multiagent/scenarios/location20.json')
    location_dict = json.load(f)   #load针对文件  loads针对内存对象

    com_uav_pos = []
    navi_uav_pos = []
    uav_pos = []
    com_uav_origin_pos = []
    navi_uav_origin_pos = []
    uav_origin_pos = []
    cnt_all = 0
    user_com_rate = []
    user_pos_auc = []
    uav_to_user_com = []  #json类型
    uav_to_user_navi = []
    uav_to_user = []
    com_connects = 0
    navi_connnects = 0
    starttime = datetime.datetime.now()
    sum_agent = 8
    com_agent_num = 5
    navi_agent_num = 3
    
    uav_roles = []
    for i in range(0,com_agent_num):
        uav_roles.append(0)
    for i in range(0,navi_agent_num):
        uav_roles.append(1)
    
    agent_pre_pos = []
    for i in range(0,sum_agent):
        agent_pre_pos.append(np.random.uniform(-0.005, +0.005, 2))
    
    # print(location_dict[0]['user_location'][0])
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2  #是定位维度
        num_communications = self.com_agent_num
        num_navigations = self.navi_agent_num
        num_agents = self.sum_agent
        num_communicate_user = 14 #推理 训练皆可变
        num_navigate_user = 6
        num_landmarks = num_communicate_user + num_navigate_user
        self.com_connects = math.floor(num_communicate_user/num_communications) + 1
        self.navi_connnects = math.floor(num_navigate_user/num_navigations) + 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.communication = True if i < num_communications else False
            agent.navigation = True if i >= num_communications else False
            agent.size = 0.01 if agent.communication else 0.01
            agent.accel = 3.0 if agent.communication else 3.0
            agent.max_speed = 0.8 if agent.communication else 0.8 #人员移动速度可控  这里agent.max_speed是从1编程0.1 9.4
        # add landmarks 
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.communication = True if i < num_communicate_user else False
            landmark.navigation = True if i >= num_communicate_user else False
            # if(landmark.communication) :
            #     landmark.movable = True
            # else :
            #     landmark.movable = False
            landmark.movable = False
            landmark.size = 0.005
            landmark.boundary = False
        # rng = default_rng()
        X = np.random.normal(0, 6, num_communicate_user)
        index = 0
        for i, landmark in enumerate(world.landmarks):
            if landmark.communication:
                landmark.normlizeX = X[index] #标准差为0的序列
                index += 1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # reset world的时候写入一次
        self.uav_to_user = []
        for com in self.uav_to_user_com:
            self.uav_to_user.append(com)
        for navi in self.uav_to_user_navi:
            self.uav_to_user.append(navi)
        self.uav_pos = []
        for com in self.com_uav_pos:
            self.uav_pos.append(com)
        for navi in self.navi_uav_pos:
            self.uav_pos.append(navi)
        
        self.uav_origin_pos = []
        for com in self.com_uav_origin_pos:
            self.uav_origin_pos.append(com)
        for navi in self.navi_uav_origin_pos:
            self.uav_origin_pos.append(navi)

        # print(self.uav_pos)
        new_dict = {
                    "com_thre": 0.1, 
                    "pos_thre": 10,
                    "uav_pos": self.uav_pos,
                    "uav_role": self.uav_roles, 
                    "uav_to_user": self.uav_to_user,   #这里需要表示为json数组的形式
                    "user_com_rate": self.user_com_rate, 
                    "user_pos_auc": self.user_pos_auc
                    }
        # data = json.dumps(new_dict, indent=1,cls=NpEncoder)
        data = json.dumps(new_dict,cls=NpEncoder)

        # with open("/root/com_navi_latest/maddpg_communicate_navigation_uav_latest/multiagent-particle-envs-master/multiagent/scenarios/deployment20.json", 'a+', newline='\n') as f:
        #     f.write(data)
        #     f.write(',')
        #     f.write('\n')
        for i, agent in enumerate(world.agents):
            if agent.communication: 
                agent.color = np.array([0.85, 0.35, 0.35]) #通信无人机  粉色
                agent.Bs = 10  #通信无人机带宽设定
            else:  
                agent.color = np.array([0.85, 0.85, 0.35]) #导航无人机 黄色
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if landmark.communication:
                landmark.color = np.array([0.25, 0.25, 0.25]) #导航用户 黑色
                landmark.Bs = random.randint(3,3) #通信用户所需带宽设定 根据研究点一的需求分级获取 可以把形状大小调整为对应带宽需求
            else:
                landmark.color = np.array([0.35, 0.85, 0.35]) #通信用户  绿色
        # set random initial states  初始位置
        for index,agent in enumerate(world.agents):
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            uav_pos_x = self.agent_pre_pos[index][0]
            uav_pos_y = self.agent_pre_pos[index][1]
            uav_pos = [uav_pos_x,uav_pos_y]
            agent.state.p_pos = np.array(uav_pos)
            # agent.state.p_pos = np.random.uniform(-0.005, +0.005, world.dim_p)  #初始位置放在原点
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                pos_x = (self.location_dict[int(self.cnt_all)]['user_location'][i][0] - 250) / 250
                pos_y = (self.location_dict[int(self.cnt_all)]['user_location'][i][1] - 250) / 250
                pos = [pos_x,pos_y]
                # print(np.random.uniform(-0.9, -0.9))
                landmark.state.p_pos = np.array(pos)
                # landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)  #第三维是指生成二维坐标
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.pre_pos_x = landmark.state.p_pos[0]
                landmark.pre_pos_y = landmark.state.p_pos[1]
        # print("cnt_all == ",self.cnt_all)
        self.cnt_all += 1
        self.cnt_all %= len(self.location_dict)
        endtime = datetime.datetime.now()
        # with open("/root/com_navi_latest/maddpg_communicate_navigation_uav_latest/multiagent-particle-envs-master/multiagent/scenarios/deployment_time20.csv", 'a+', newline='\n') as f:
        #     f.write(str((endtime - self.starttime).total_seconds()))
        #     f.write('\n')
        self.starttime = endtime
        if len(self.uav_origin_pos) != 0:  #判空 否则可能出错
            self.agent_pre_pos = self.uav_origin_pos   
        


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all adversarial agents
    def communications(self, world):
        return [agent for agent in world.agents if agent.communication]

    def navigations(self, world):
        return [agent for agent in world.agents if agent.navigation]

    def communicate_user(self, world):
        return [Landmark for Landmark in world.landmarks if Landmark.communication]
    
    def navigate_user(self, world):
        return [Landmark for Landmark in world.landmarks if Landmark.navigation]

    def reward(self, agent, world):
        if agent.communication:
            main_reward,whole_rew,success_sum = self.communication_reward(agent,world)
        else:
            main_reward,whole_rew,success_sum = self.navigation_reward(agent,world)
        return main_reward

    def whole_reward(self, agent, world):
        if agent.communication:
            main_reward,whole_rew,success_sum = self.communication_reward(agent,world)
        else:
            main_reward,whole_rew,success_sum = self.navigation_reward(agent,world)
        return whole_rew

    def success_rate(self, agent, world):
        if agent.communication:
            main_reward,whole_rew,success_sum = self.communication_reward(agent,world)
        else:
            main_reward,whole_rew,success_sum = self.navigation_reward(agent,world)
        return success_sum

    def db2W(self, db):
        return pow(10,db/10)/1000

    def getCommunicateRate(self, min_dist, other_dist:list, X):
        P = 20
        P_w = self.db2W(P)
        noise = -140
        noise_w = self.db2W(noise)
        c = 3 * 10**8
        f = 1.4 * 10**3  #下面都转化为MHZ算的
        d = 1
        A = 0.25
        C = 0.39
        E = 0.25
        G = 0
        H = 0.05
        h = 100  #无人机固定飞行高度
        alpha = 3.5

        xy_dist = min_dist * 100 #需要计算的值  muti-env中环境dist在0~3左右
        dist = math.sqrt(h**2 + xy_dist**2)
        theta = math.atan(h / xy_dist)
        L_fspl = 20 * math.log(4 * math.pi * f * 10 ** 6 * d / c,10)
        L_slant = A * f ** C * dist ** E * (G + theta) ** H
        L_f = L_fspl + 10 * alpha * math.log(dist / d,10) + L_slant + X  #加个X
        L = 10 ** (L_f/10)

        xy_dist_other = other_dist
        other_sum_L = 0
        for dis in xy_dist_other:
            dis *= 100 #单位是m
            dist_other = math.sqrt(h**2 + dis**2)
            theta_other = math.atan(h / dist_other)
            L_slant_other = A * f ** C * dist_other ** E * (G + theta_other) ** H
            L_f_other = L_fspl + 10 * alpha * math.log(dist_other / d,10) + L_slant_other + X  #加个X
            L_other = 10 ** (L_f_other/10)
            other_sum_L += L_other

        r = (P_w / L) / (noise_w + P_w / other_sum_L)

        rate = math.log(1 + r,2)
        
        return rate

    def getCRLB(self,list_user:list,list_time:list,list_uav:list):
        c = 3*10**8
        sum_crlb = 0
        pos_auc = []
        for (l,u) in zip(list_time,list_user):
            nk0_1 = abs(l[0] - l[1])
            nk0_2 = abs(l[0] - l[2])
            ek = [nk0_1*c,nk0_2*c]
            ek_T = np.mat(ek).T
            Rk = np.cov(ek_T*ek)
            # loc = Locator(list_uav)  #这里是获取infer的点坐标  使用TDOA方法
            # list_infer_user= loc.locate(l) #这里time的维度 和 list_locate要保持一致
            xu = u['x']
            yu = u['y']

            xk0 = list_uav[0]['x']
            xk1 = list_uav[1]['x']
            xk2 = list_uav[2]['x']
            yk0 = list_uav[0]['y']
            yk1 = list_uav[1]['y']
            yk2 = list_uav[2]['y']

            #rk是距离
            rk0 = np.sqrt((xk0 - xu)**2 + (yk0 - yu)**2)
            rk1 = np.sqrt((xk1 - xu)**2 + (yk1 - yu)**2)
            rk2 = np.sqrt((xk2 - xu)**2 + (yk2 - yu)**2)

            Qk = [[(xu - xk1)/rk1 - (xu - xk0)/rk0,(yu - yk1)/rk1 - (yu - yk0)/rk0],
                [(xu - xk2)/rk2 - (xu - xk0)/rk0,(yu - yk2)/rk2 - (yu - yk0)/rk0]]

            temp = np.mat(Qk).T * np.linalg.pinv(Rk) * Qk

            J = 1/np.mat(temp).trace()
            # print ("J[0,0] = ",J[0,0])
            sum_crlb += J[0,0]
            auc = 1
            if(J[0,0] >= 0.04):
                auc = np.random.uniform(5,9.99)
            elif(J[0,0] < 0.003):
                auc = np.random.uniform(0.9,5)
            else:
                auc = J[0,0] * 250
            pos_auc.append(auc)
        self.user_pos_auc = pos_auc
        return abs(sum_crlb)

    def takeSecond(self,elem):
        return elem[1]

    def navigation_reward(self, agent, world):  #导航无人机
        rew = 0
        only_rew = 0
        sum_success = 0
        navigations = self.navigations(world)
        navigate_users = self.navigate_user(world)
        #对每个定位节点而言  找到所有的定位节点 传入即可
        list_uav = []
        for a in navigations:
            dict0 = dict([('x',a.state.p_pos[0] * 10),('y',a.state.p_pos[1] * 10)])
            list_uav.append(dict0)

        list_time = []  #这里应该是二维数组  记录的是每个user对应的时间
        for l in navigate_users:
            list_temp = []
            for a in navigations:
                time = np.sqrt(np.sum(np.sqrt(10) * np.square(a.state.p_pos - l.state.p_pos))) / (3 * 10**8)
                list_temp.append(time)
            list_time.append(list_temp)

        list_user = []
        for l in navigate_users:
            dict0 = dict([('x',l.state.p_pos[0] * 10),('y',l.state.p_pos[1] * 10)])
            list_user.append(dict0)

        #得到了time矩阵
        crlb = self.getCRLB(list_user,list_time,list_uav)
        # crlb *= 10 # 10是导航的重要性系数
        if(crlb < 5) :  #阈值
            # rew -= crlb 
            only_rew -= crlb
        else:
            # rew -= 5
            only_rew -= 5
        navi_max_dist = 0
        for l in navigate_users:
            dists = sum([np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in navigations])
            navi_max_dist = max(dists,navi_max_dist)
            # print("nav_dist:",dists)
            # print("nav_dists : ",dists)
            rew -= dists  #最主要是来自这的惩罚
        rew -= 4*navi_max_dist  #优化阈值
        if agent.collide:
            for a in navigations:
                if self.is_collision(a, agent):
                    rew -= 2
        #无人机连接的用户  这里每次排序其实影响算法效率了  加个开关
        adv_dict = {}
        for uav in navigations:
            adv_dict[uav] = []
        for a in navigate_users:
            uav_dist_tuple = []
            for adv in navigations:
                dist_temp = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                uav_dist_tuple.append((adv,dist_temp))
            uav_dist_tuple.sort(key = self.takeSecond)
            connect_num = 0
            for uav in uav_dist_tuple:
                if connect_num >= 3:  #至少3个无人机 才能定位解算
                    break
                if len(adv_dict[uav[0]]) < self.navi_connnects:
                    adv_dict[uav[0]].append(a)
                    connect_num += 1
            # adv_dict[uav_dist_tuple[0][0]].append(a)
            # adv_dict[uav_dist_tuple[1][0]].append(a)
            # adv_dict[uav_dist_tuple[2][0]].append(a)  #可变 至少三架无人机

        navi_dict = []
        navi_uav_pos_temp = []
        navi_uav_origin_pos_temp = []
        for uav in navigations:
            t = copy.copy(uav.state.p_pos)
            t[0] *= 250
            t[0] += 250
            t[1] *= 250
            t[1] += 250
            navi_uav_pos_temp.append(t)  #这里统一在reset变化
            navi_uav_origin_pos_temp.append(uav.state.p_pos)
            navi_user_pos_temp = []
            for user in adv_dict[uav]:
                temp = copy.copy(user.state.p_pos)
                temp[0] *= 250
                temp[0] += 250
                temp[1] *= 250
                temp[1] += 250
                navi_user_pos_temp.append(temp)
            # navi_uav_pos_temp.append(uav.state.p_pos)  #这里统一在reset变化
            dict_temp = {
                            "uav_pos" : t,
                            "user_pos" : navi_user_pos_temp
                        }
            navi_dict.append(dict_temp)
        
        # print("定位无人机用户关联：",navi_dict)
        self.uav_to_user_navi = navi_dict
        self.navi_uav_pos = navi_uav_pos_temp
        self.navi_uav_origin_pos = navi_uav_origin_pos_temp
        return rew,only_rew,sum_success

    #函数中传值过来  然后写个函数打印即可
    def getUavPosition(self):
        # print("uav_pos:",self.uav_pos)
        return self.uav_pos

    def communication_reward(self, agent, world):  #通信无人机  这里会发生碰撞  这里每个无人机的reward是用户的数组通信速率的加权和  然后无人机有自己的固定带宽  其他的发射功率 信道增益等都是次要的
        # communications are rewarded for collisions with agents
        rew = 0  #这里的reward是某一类整体的reward   但是也可以计算出这个整体中每一个节点的reward值 这需要去计算遗传算法的适应度的
        #如果超出连接数量  那么负奖励 > 连接的正奖励  连接一个给20  超出给-30
        whole_rew = 0
        success_sum = 0
        shape = True
        communicate_users = self.communicate_user(world)
        communications = self.communications(world)
        adv_dict = {}
        for adv in communications:
            adv_dict[adv] = []
        for a in communicate_users:
            min_dist = 999
            min_adv = communications[0]
            for adv in communications:
                if(min_dist > np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))):
                    min_dist = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    min_adv = adv
            adv_dict[min_adv].append(a)
        
        adv_dict_display = {}   #其实这个更合理  试下这个
        for adv in communications:
            adv_dict_display[adv] = []
        for a in communicate_users:
            uav_dist_tuple = []
            for adv in communications:
                dist_temp = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                uav_dist_tuple.append((adv,dist_temp))
            uav_dist_tuple.sort(key = self.takeSecond)
            for tup in uav_dist_tuple:
                if len(adv_dict_display[tup[0]]) < self.com_connects:
                    adv_dict_display[tup[0]].append(a)
                    break

        com_dict = []
        uav_pos_temp = []
        uav_origin_pos_temp = []
        for adv in communications:
            temp = copy.copy(adv.state.p_pos)
            temp[0] *= 250
            temp[0] += 250
            temp[1] *= 250
            temp[1] += 250
            user_pos = []
            uav_pos_temp.append(temp)
            uav_origin_pos_temp.append(adv.state.p_pos)
            for a in adv_dict_display[adv]:
                t = copy.copy(a.state.p_pos)
                t[0] *= 250
                t[0] += 250
                t[1] *= 250
                t[1] += 250
                user_pos.append(t)
            dict_temp = {
                            "uav_pos" : temp,  #改成temp
                            "user_pos" : user_pos
                        }
            com_dict.append(dict_temp)
        # print("通信无人机用户关联：",com_dict)
        self.uav_to_user_com = com_dict
   
        com_max_dist = 0
        for user in communicate_users:
            dist_com =  sum([np.sqrt(np.sum(np.square(a.state.p_pos - user.state.p_pos))) for a in communications])
            com_max_dist = max(com_max_dist,dist_com)
            rew -= dist_com
            # dist_com_all += dist_com
        rew -= 4*com_max_dist
        # print(dist_com_all)
        #计算一下 每个无人机连接的用户带宽之和  和自己拥有的带宽
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            success = True
            user_rate_temp = []
            for adv in communications:   #对应的是每一个通信无人机
                #如果只关注sum的话 可能会出现跑飞的情况  需要距离惩罚
                dist_punish = sum([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in adv_dict[adv]])
                # print("com_dist==",dist_punish)
                sum_Bs = 0
                sum_rate = 0
                min_rate = 999
                for a in adv_dict[adv]:
                    dist_cur = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    dist_list = []
                    for a_adv in communications:
                        if(a_adv != adv):
                            dist_list.append(np.sqrt(np.sum(np.square(a.state.p_pos - a_adv.state.p_pos))))
                    x = a.normlizeX
                    rate = self.getCommunicateRate(dist_cur,dist_list,x) * a.Bs   #uav的总带宽比上连接数量
                    candidate = rate * self.com_connects/len(adv_dict[adv])
                    candidate = candidate if candidate >= 1 else random.uniform(1,1.99)
                    user_rate_temp.append(candidate / 10)
                    # user_rate_temp.append(rate / 10) #origin
                    sum_Bs += a.Bs
                    sum_rate += rate
                    # print("rate : ",rate)
                    min_rate = min(min_rate,rate)
                if sum_Bs > adv.Bs : #这里阈值要定义一下 0.5M/s
                    success = False
                    # rew -= 5 * sum_rate
                else :
                    rew += 2 * sum_rate
                    whole_rew += sum_rate
                    # print("sum_rate_true : ",sum_rate)
                for p in range(world.dim_p):
                    x = abs(adv.state.p_pos[p])
                    if(x >= 1):
                        success = False
                # rew -= dist_com_all #距离惩罚
                # print("dist_punish : ",dist_punish)
                # rew -= min_rate #优化最低
            # print("用户速率数组：",user_rate_temp)
            self.user_com_rate = user_rate_temp
            if success == True:
                success_sum = 1
                # print ("success!")
            # else :
                # print ("false!")
        
        for adv1 in communications:
            for adv2 in communications:
                if adv1 == adv2:
                    continue
                if self.is_collision(adv1, adv2):
                    rew -= 2
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for adv in communications:
            for p in range(world.dim_p):
                x = abs(adv.state.p_pos[p])
                rew -= bound(x)
        self.com_uav_pos = uav_pos_temp
        self.com_uav_origin_pos = uav_origin_pos_temp
        return rew,whole_rew,success_sum

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.communication:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)





 