import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
from multiagent.scenarios.Locator import Locator  # 奇怪  必须要绝对路径
import math
from numpy.random import default_rng
from queue import PriorityQueue


class Scenario(BaseScenario):  # 在reset的时候 修改用户比例
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2  # 是定位维度
        num_communications = 3
        num_navigations = 3
        num_agents = num_communications + num_navigations
        num_communicate_user = 6  # 推理 训练皆可变
        num_navigate_user = 3
        num_landmarks = num_communicate_user + num_navigate_user
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.communication = True if i < num_communications else False
            agent.navigation = True if i >= num_communications else False
            agent.size = 0.05 if agent.communication else 0.03
            agent.accel = 3.0 if agent.communication else 1.0
            agent.max_speed = 1.0 if agent.communication else 0.1  # 人员移动速度可控  待会再改
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.communication = True if i < num_communicate_user else False
            landmark.navigation = True if i >= num_communicate_user else False
            if (landmark.communication):
                landmark.movable = True
            else:
                landmark.movable = False
            landmark.size = 0.02
            landmark.boundary = False
        rng = default_rng()
        X = rng.standard_normal(num_communicate_user)
        index = 0
        for i, landmark in enumerate(world.landmarks):
            if landmark.communication:
                landmark.normlizeX = X[index]  # 标准差为6的序列
                index += 1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        # print ("reset......................................................")
        for i, agent in enumerate(world.agents):
            if agent.communication:
                agent.color = np.array([0.85, 0.35, 0.35])  # 通信无人机  粉色
                agent.Bs = 6  # 通信无人机带宽设定
            else:
                agent.color = np.array([0.85, 0.85, 0.35])  # 导航无人机 黄色
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if landmark.communication:
                landmark.color = np.array([0.25, 0.25, 0.25])  # 导航用户 黑色
                landmark.Bs = random.randint(3, 3)  # 通信用户所需带宽设定 根据研究点一的需求分级获取 可以把形状大小调整为对应带宽需求
            else:
                landmark.color = np.array([0.35, 0.85, 0.35])  # 通信用户  绿色
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.pre_pos_x = landmark.state.p_pos[0]
                landmark.pre_pos_y = landmark.state.p_pos[1]

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
            main_reward, whole_rew = self.communication_reward(agent, world)
        else:
            main_reward, whole_rew = self.navigation_reward(agent, world)
        return main_reward

    def whole_reward(self, agent, world):
        if agent.communication:
            main_reward, whole_rew = self.communication_reward(agent, world)
        else:
            main_reward, whole_rew = self.navigation_reward(agent, world)
        return whole_rew

    def db2W(self, db):
        return pow(10, db / 10) / 1000

    def getCommunicateRate(self, min_dist, other_dist: list, X):
        P = 20
        P_w = self.db2W(P)
        noise = -140
        noise_w = self.db2W(noise)
        c = 3 * 10 ** 8
        f = 1.4 * 10 ** 3  # 下面都转化为MHZ算的
        d = 1
        A = 0.25
        C = 0.39
        E = 0.25
        G = 0
        H = 0.05
        h = 100  # 无人机固定飞行高度
        alpha = 3.5

        xy_dist = min_dist * 100  # 需要计算的值  muti-env中环境dist在0~3左右
        dist = math.sqrt(h ** 2 + xy_dist ** 2)
        theta = math.atan(h / xy_dist)
        L_fspl = 20 * math.log(4 * math.pi * f * 10 ** 6 * d / c, 10)
        L_slant = A * f ** C * dist ** E * (G + theta) ** H
        L_f = L_fspl + 10 * alpha * math.log(dist / d, 10) + L_slant + X  # 加个X
        L = 10 ** (L_f / 10)

        xy_dist_other = other_dist
        other_sum_L = 0
        for dis in xy_dist_other:
            dis *= 100  # 单位是m
            dist_other = math.sqrt(h ** 2 + dis ** 2)
            theta_other = math.atan(h / dist_other)
            L_slant_other = A * f ** C * dist_other ** E * (G + theta_other) ** H
            L_f_other = L_fspl + 10 * alpha * math.log(dist_other / d, 10) + L_slant + X  # 加个X
            L_other = 10 ** (L_f_other / 10)
            other_sum_L += L_other

        r = (P_w / L) / (noise_w + P_w / other_sum_L)

        rate = math.log(1 + r, 2)

        return rate

    def getCRLB(self, list_user: list, list_time: list, list_uav: list):
        c = 3 * 10 ** 8
        sum_crlb = 0
        for (l, u) in zip(list_time, list_user):
            nk0_1 = abs(l[0] - l[1])
            nk0_2 = abs(l[0] - l[2])
            ek = [nk0_1 * c, nk0_2 * c]
            ek_T = np.mat(ek).T
            Rk = np.cov(ek_T * ek)
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

            # rk是距离
            rk0 = np.sqrt((xk0 - xu) ** 2 + (yk0 - yu) ** 2)
            rk1 = np.sqrt((xk1 - xu) ** 2 + (yk1 - yu) ** 2)
            rk2 = np.sqrt((xk2 - xu) ** 2 + (yk2 - yu) ** 2)

            Qk = [[(xu - xk1) / rk1 - (xu - xk0) / rk0, (yu - yk1) / rk1 - (yu - yk0) / rk0],
                  [(xu - xk2) / rk2 - (xu - xk0) / rk0, (yu - yk2) / rk2 - (yu - yk0) / rk0]]

            temp = np.mat(Qk).T * np.linalg.pinv(Rk) * Qk

            J = 1 / np.mat(temp).trace()
            # print ("J[0,0] = ",J[0,0])
            sum_crlb += J[0, 0]
        return abs(sum_crlb)

    def navigation_reward(self, agent, world):  # 导航无人机
        rew = 0
        only_rew = 0
        navigations = self.navigations(world)
        navigate_users = self.navigate_user(world)
        # 对每个定位节点而言  找到所有的定位节点 传入即可
        list_uav = []
        for a in navigations:
            dict0 = dict([('x', a.state.p_pos[0] * 10), ('y', a.state.p_pos[1] * 10)])
            list_uav.append(dict0)

        list_time = []  # 这里应该是二维数组  记录的是每个user对应的时间
        for l in navigate_users:
            list_temp = []
            for a in navigations:
                time = np.sqrt(np.sum(np.sqrt(10) * np.square(a.state.p_pos - l.state.p_pos))) / (3 * 10 ** 8)
                list_temp.append(time)
            list_time.append(list_temp)

        list_user = []
        for l in navigate_users:
            dict0 = dict([('x', l.state.p_pos[0] * 10), ('y', l.state.p_pos[1] * 10)])
            list_user.append(dict0)

        # 得到了time矩阵
        crlb = self.getCRLB(list_user, list_time, list_uav)
        crlb *= 10  # 10是导航的重要性系数
        if (crlb < 5):  # 阈值
            rew -= crlb
            only_rew -= crlb
        else:
            rew -= 5
            only_rew -= 5
        for l in navigate_users:
            dists = sum([np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in navigations])
            rew -= dists
        if agent.collide:
            for a in navigations:
                if self.is_collision(a, agent):
                    rew -= 2
        return rew, only_rew

    def communication_reward(self, agent,
                             world):  # 通信无人机  这里会发生碰撞  这里每个无人机的reward是用户的数组通信速率的加权和  然后无人机有自己的固定带宽  其他的发射功率 信道增益等都是次要的
        # communications are rewarded for collisions with agents
        rew = 0  # 这里的reward是某一类整体的reward   但是也可以计算出这个整体中每一个节点的reward值 这需要去计算遗传算法的适应度的
        # 如果超出连接数量  那么负奖励 > 连接的正奖励  连接一个给20  超出给-30
        whole_rew = 0
        shape = True
        communicate_users = self.communicate_user(world)
        communications = self.communications(world)

        adv_dict = {}
        user_dict = {}

        for communication in communications:
            user_dict[communication] = 0

        for adv in communications:
            q = PriorityQueue()
            adv_dict[adv] = q

        for adv in communications:
            q = PriorityQueue()
            for user in communicate_users:
                dist = np.sqrt(np.sum(np.square(user.state.p_pos - adv.state.p_pos)))
                if(q.qsize() < 2):
                    q.put((dist,user))
            adv_dict[adv] = q


        # 计算一下 每个无人机连接的用户带宽之和  和自己拥有的带宽
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            success = True
            for adv in communications:  # 对应的是每一个通信无人机
                dist_punish = sum([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in adv_dict[adv]])
                sum_Bs = 0
                sum_rate = 0
                min_rate = 999
                for a in adv_dict[adv]:
                    dist_cur = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    dist_list = []
                    for a_adv in communications:
                        if (a_adv != adv):
                            dist_list.append(np.sqrt(np.sum(np.square(a.state.p_pos - a_adv.state.p_pos))))
                    x = a.normlizeX
                    rate = self.getCommunicateRate(dist_cur, dist_list, x) * a.Bs
                    sum_Bs += a.Bs
                    if(user_dict[adv] != 1):
                        sum_rate += rate
                        rew += 2 * sum_rate
                    min_rate = min(min_rate, rate)
                    user_dict[adv] = 1
                    whole_rew += sum_rate
                for p in range(world.dim_p):
                    x = abs(adv.state.p_pos[p])
                    if (x >= 1):
                        success = False
                rew -= dist_punish * 6  # 距离惩罚 不能设置过大 系数是通信无人机数量
                rew -= min_rate  # 优化最低
            if success == True:
                rew += 1000

        for adv1 in communications:
            for adv2 in communications:
                if adv1 == adv2:
                    continue
                if self.is_collision(adv1, adv2):
                    rew -= 10

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

        return rew, whole_rew

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
