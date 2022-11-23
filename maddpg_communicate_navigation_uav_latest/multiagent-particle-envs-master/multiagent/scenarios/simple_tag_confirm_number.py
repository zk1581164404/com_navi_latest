import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
from multiagent.scenarios.Locator import Locator #奇怪  必须要绝对路径
from queue import PriorityQueue

#todo  加上碰撞惩罚  通信用户 与 通信节点之间； 通信用户 与 通信用户之间 ；  要不直接加上集体的碰撞惩罚？  先解决不碰撞的问题

#1、将连接策略落实下来  2、将数据量落实下来  如何将一个无人机模拟为数据量大的  可以表现为较大的体积

#碰撞惩罚设置为 -1  效果还可以
# 后面都设置为-5  效果一般  通信节点可能会出现不动的情况  尽量避免吧
#

#需要注意的是  move的地面用户是没有reward的

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2  #是定位维度
        num_good_agents = 6
       # num_adversaries = random.randint(1,3)
        num_adversaries = 3
        num_navigations = 3
        num_agents = num_adversaries + num_good_agents + num_navigations
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.goodagent = True if i >= num_adversaries and i < num_adversaries + num_good_agents else False
            agent.navigation = True if i >= num_adversaries + num_good_agents else False
            # agent.size = 0.075 if agent.adversary else 0.05
            agent.size = 0.05 if agent.adversary else 0.03
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 0.5 if agent.adversary else 1.0
            # agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            # landmark.size = 0.05
            landmark.size = 0.02
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if agent.goodagent:  #通信用户  绿色
                agent.color = np.array([0.35, 0.85, 0.35])
            elif agent.adversary: #通信无人机  粉色
                agent.color = np.array([0.85, 0.35, 0.35])
            else:  #导航无人机 黄色
                agent.color = np.array([0.85, 0.85, 0.35])
           # agent.color = np.array([0.85, 0.35, 0.35]) if not agent.goodagent else np.array([0.35, 0.85, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25]) #导航用户 黑色
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):  # zk found question
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if agent.goodagent]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def navigations(self, world):
        return [agent for agent in world.agents if agent.navigation]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.adversary:
            main_reward = self.adversary_reward(agent,world)
        elif agent.goodagent:
            main_reward = self.agent_reward(agent,world)
        else:
            main_reward = self.navigation_reward(agent,world)
       # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):  #这里是地面的通信用户  是在移动的
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        # shape = False
        # adversaries = self.adversaries(world)
        # if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            # for adv in adversaries:
                # rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        # if agent.collide:
            # for a in adversaries:
                # if self.is_collision(a, agent):
                    # rew -= 5  # rew-=10  from zk

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # 带上墙壁的话 仿真效果不是很好
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def getCRLB(self,list_user:list,list_time:list,list_uav:list):   #包含了tdoa
        c = 340.29
        sum_crlb = 0
        # print ("list_locate = ",list_user)
        # print ("list_time = ",list_time)
        # print ("list_uav = ",list_uav)
        for l in list_time:
            nk0_1 = abs(l[0] - l[1])
            nk0_2 = abs(l[0] - l[2])
            ek = [nk0_1*c,nk0_2*c]
            ek_T = np.mat(ek).T
            Rk = np.cov(ek_T*ek)
            loc = Locator(list_uav)  #这里是获取infer的点坐标
            list_infer_user= loc.locate(l) #这里time的维度 和 list_locate要保持一致
            xu = list_infer_user['x']
            yu = list_infer_user['y']

            # print("xu= ",xu,"yu= ",yu)
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

            # print(rk0,rk1,rk2)

            Qk = [[(xu - xk1)/rk1 - (xu - xk0)/rk0,(yu - yk1)/rk1 - (yu - yk0)/rk0],
                [(xu - xk2)/rk2 - (xu - xk0)/rk0,(yu - yk2)/rk2 - (yu - yk0)/rk0]]

            # print("np.mat(Qk).T = ",np.mat(Qk).T)
            # print("np.mat(Rk).I = ",np.linalg.pinv(Rk))
            # print("Qk = ",Qk)
            # print(np.linalg.det(Rk))
            temp = np.mat(Qk).T * np.linalg.pinv(Rk) * Qk
            # print("temp=",temp)
            J = 1/np.mat(temp).trace()
            # print ("J[0,0] = ",J[0,0])
            sum_crlb += J[0,0]
        return abs(sum_crlb)

    def navigation_reward(self, agent, world):  #导航无人机
        c = 3*10**8
        rew = 0
        shape = False
        agents = self.navigations(world)
        #对每个定位节点而言  找到所有的定位节点 传入即可
        list_uav = []
        for a in agents:
            dict0 = dict([('x',a.state.p_pos[0]),('y',a.state.p_pos[1])])
            list_uav.append(dict0)

        list_time = []  #这里应该是二维数组  记录的是每个user对应的时间
        for l in world.landmarks:
            list_temp = []
            for a in agents:
                time = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) / 340.29
                list_temp.append(time)
            list_time.append(list_temp)

        list_user = []
        for l in world.landmarks:
            dict0 = dict([('x',l.state.p_pos[0]),('y',l.state.p_pos[1])])
            list_user.append(dict0)

        #得到了time矩阵
        crlb = self.getCRLB(list_user,list_time,list_uav)

        # print("*********************crlb=",crlb)
        # print ("crlb=",crlb)
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in agents]
            rew -= min(dists) # *10 --> *1    4.8
            if (crlb < 10):
                rew -= crlb # *10 --> *1    4.8
            else:
                rew -= 10  #改了其实有一些影响的  可能出现reward抖了一下
        if agent.collide:
            for a in agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

#加上通信速率啥的  参数定义好
#加上对比实验的参数调节  然后改成ddpg之后 再看一下结果如何
#完后论文解法建模部分撰写


#这里固定连接策略  每个无人机均匀连接（没法根据需求均匀连接）  最终没被连接的用户  给与大的惩罚
    def adversary_reward(self, agent, world):  #通信无人机  这里会发生碰撞  这里每个无人机的reward是用户的数组通信速率的加权和  然后无人机有自己的固定带宽  其他的发射功率 信道增益等都是次要的
        # Adversaries are rewarded for collisions with agents
        rew = 0  #这里的reward是某一类整体的reward   但是也可以计算出这个整体中每一个节点的reward值 这需要去计算遗传算法的适应度的
        #如果超出连接数量  那么负奖励 > 连接的正奖励  连接一个给20  超出给-30
        shape = True   #by zk
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        #先遍历用户 然后搞个map  假设一个无人机最多连两个
        #这里可以
        adv_dict = {}
        user_dict = {}
        
        for agent in agents:
            user_dict[agent] = 0

        for adv in adversaries:
            q = PriorityQueue()
            adv_dict[adv] = q

        for adv in adversaries:
            q = PriorityQueue() 
            for a in agents:
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                q.put((dist,a))
            adv_dict[adv] = q                    

        adv_dist = {}
        for adv in adversaries:
            adv_dist[adv] = []
        for a in agents:
            min_dist = 0  #定义这个值会不会有风险？
            min_adv = adversaries[1]
            for adv in adversaries:
                if(min_dist < np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) or min_dist == 0):
                    min_dist = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    min_adv = adv
            adv_dist[min_adv].append(a)

        #计算一下 每个无人机连接的用户带宽之和  和自己拥有的带宽

        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            success = True
            for adv in adversaries:   #对应的是每一个通信无人机
                #如果只关注sum的话 可能会出现跑飞的情况  还是需要距离控制一下
                sum_connect = len(agents)
                sum_punish = 0
                q = adv_dict[adv]
                i = 0
                while i < 2:
                    i += 1
                    user_dict[q.get()] = 1
                # print ("adv == ",adv)
                # print ("sum == ",sum)
                dist_punish = sum([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in adv_dist[adv]])   #这里的距离惩罚有问题  应该处理的是连接的节点的距离
                for user in agents:
                    if(user_dict[user] == 0):
                        sum_punish += 1
                rew += 10 * sum_connect
                rew -= 15 * sum_punish
                rew -= dist_punish

        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew -= 10 #hange by zk +20 - 0 4.3  这里会发生碰撞
        #             if self.is_collision(ag,agent):  #agent内部也包含adv了属于是
        #                 rew -= 10
        return rew

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
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
