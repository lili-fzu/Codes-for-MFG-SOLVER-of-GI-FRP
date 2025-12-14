"""
Created on Sun Dec 14 12:08:32 2025

@author: lily and Lin
"""

import numpy as np
import copy
import statistics
import pandas as pd

def build_A_explicit(p1, p2, p3, p4, p5):
    # 10天的紧急概率序列
    p = [p1, p2, p3, p4, p5, p1, p2, p3, p4, p5]
    q = [1-p1, 1-p2, 1-p3, 1-p4, 1-p5, 1-p1, 1-p2, 1-p3, 1-p4, 1-p5]
    
    A = np.zeros((10, 10))
    
    for j in range(10):  # 组别 (限行开始日)
        # 该组的10天周期
        days = [(j + k) % 10 for k in range(10)]
        
        for k in range(10):  # 周期中的第几天
            i = days[k]  # 实际日期
            
            if k < 9:  # 第1-9天
                # 前k天都紧急，第k+1天不紧急
                prob = 1.0
                for m in range(k):
                    prob *= p[days[m]]
                prob *= q[days[k]]
                A[i, j] = prob
            else:  # 第10天 (最后一天)
                # 前9天都紧急
                prob = 1.0
                for m in range(9):
                    prob *= p[days[m]]
                A[i, j] = prob
                
    return A

def solve_agent_distribution(N_group, urgent_prob, agent_distribution_equal):
    agent_distribution = {}
    for v in ['EV','GV']:        
        if agent_distribution_equal[v] == 2:
            agent_distribution[v] = [0]*N_group[v]
            agent_distribution[v][0] = 1
        elif agent_distribution_equal[v] == 1:
            agent_distribution[v] = [1/N_group[v]]*N_group[v]
        else:
            #assume 5 group first
            p1 = urgent_prob[0]
            p2 = urgent_prob[1]
            p3 = urgent_prob[2]
            p4 = urgent_prob[3]
            p5 = urgent_prob[4]            
            if N_group[v] == 10: # 若为10天一限
                A = build_A_explicit(p1, p2, p3, p4, p5)
                B = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
                X = np.linalg.solve(A, B)
                for i in range(10): # update X, remove negative values
                    if X[i] < 0:
                        X[i] = 0
                agent_distribution[v] = [0]*10
                for i in range(10): # get agent population based on updated X
                    agent_distribution[v][i] = X[i] / np.sum(X) # get ideal grouping for 5 groups
            else: # 若不为10天一限，其他天，均先假定5天一限
                A = np.array([[1-p1, p2*p3*p4*p5, p3*p4*p5*(1-p1), p4*p5*(1-p1), p5*(1-p1)],
                              [p1*(1-p2), 1-p2, p3*p4*p5*p1, p4*p5*p1*(1-p2), p5*p1*(1-p2)],
                              [p1*p2*(1-p3), p2*(1-p3), 1-p3, p4*p5*p1*p2, p5*p1*p2*(1-p3)],
                              [p1*p2*p3*(1-p4), p2*p3*(1-p4), p3*(1-p4), 1-p4, p5*p1*p2*p3],
                              [p1*p2*p3*p4, p2*p3*p4*(1-p5), p3*p4*(1-p5), p4*(1-p5), 1-p5]])
                B = np.array([0.2,0.2,0.2,0.2,0.2])                   
                X = np.linalg.solve(A, B)
                for i in range(5): # update X, remove negative values
                    if X[i] < 0:
                        X[i] = 0
                agent_distribution_temp = [0]*5
                for i in range(5): # get agent population based on updated X
                    agent_distribution_temp[i] = X[i] / np.sum(X) # get ideal grouping for 5 groups
                temp = agent_distribution_temp[0:N_group[v]] # keep the first N_group data
                agent_distribution[v] = [0]*N_group[v]
                for i in range(N_group[v]):
                    if i <= len(temp) - 1: #若N_group[v]小于等于5，则所有i满足此条件
                        agent_distribution[v][i] = temp[i] / np.sum(temp) # adjust the first N_group data to ensure the sum to be 1                    
    return agent_distribution

def i2weekday(i,g):
    weekday = (i+g)%5
    return weekday
def i2day(i,g,v): # i refers to the order of day in a certain group's cycle, day means calendar day
    day = (i+g)%(N_day_in_cycle[v]*N_sub_cycle[v])
    return day
def day2i(day,g,v): # i refers to the order of day in a certain group's cycle, day means calendar day
    i = (day-g)%(N_day_in_cycle[v]*N_sub_cycle[v])
    return i
    
def get_N_car_agent(population_pi, self_group): #1st dim: bus_already, 2nd dim: day, 3rd dim: urgent, 4th dim: action
    N = {}
    Nb = {}
    nb = {}
    Ncar_dic = {}
    for v in ['EV','GV']:
        N[v] = [0]*N_group[v]  # N[v][g]:number of people in type v's group g
        Nb[v] = {g:[0]*N_day_in_cycle[v]*N_sub_cycle[v] for g in range(N_group[v])} #Nb[v][g][i]: number of bus-takers on its ith day (in its self cycle) for type v's group g
        nb[v] = [0]*N_day_in_cycle[v]*N_sub_cycle[v] #nb[v][i]: number of bus-takers from first to last day in the big cycle
        Ncar_dic[v] = {}
    for v in ['EV','GV']:
        for g, pi_temp in population_pi[v].items():
            if self_group == g: # if self type, if self_group == None, all types are other types
                N[v][g] = N_agent[v] * agent_distribution[v][g] - 1
            else: # if other types
                N[v][g] = N_agent[v] * agent_distribution[v][g]
            for i in range(N_day_in_cycle[v]*N_sub_cycle[v]):
                weekday = i2weekday(i,g) # obtain the corresponding weekday for day i of g agent
                Nb_sum_temp = 0 # the so-far total number of bus-takers in the current sub_cycle for group g
                cycle_begin = int(i/N_day_in_cycle[v])*N_day_in_cycle[v]
                for j in range(cycle_begin,i):
                    Nb_sum_temp += Nb[v][g][j]
                if (i+1)%N_day_in_cycle[v] != 0: #if day i is not the last day in sub_cycle
                    Nb[v][g][i] = (N[v][g]-Nb_sum_temp) * (pi_temp[0][i][0][1] * (1-urgent_prob[weekday]) + pi_temp[0][i][1][1] * urgent_prob[weekday])
                else:
                    Nb[v][g][i] = N[v][g]-Nb_sum_temp
                    
        for day in range(N_day_in_cycle[v]*N_sub_cycle[v]):
            for g in range(N_group[v]):
                i = day2i(day,g,v)
                nb[v][day] += Nb[v][g][i]
        # Ncar_dic[v]['full'][day] stores the number of vehicle v's agents on calendar day 'day'
        Ncar_dic[v]['full'] = [0]*N_day_in_cycle[v]*N_sub_cycle[v]
        for day in range(N_day_in_cycle[v]*N_sub_cycle[v]):
            if self_group == None:
                Ncar_dic[v]['full'][day] = N_agent[v]-nb[v][day]
            else:
                Ncar_dic[v]['full'][day] = N_agent[v]-nb[v][day]-1
        # Ncar_dic[v][g] stores the number of vehicle v's agent g taking car from first to last day in the big cycle, in order
        for g in range(N_group[v]):
            Ncar_dic[v][g] = [0]*N_day_in_cycle[v]*N_sub_cycle[v]
            for day in range(N_day_in_cycle[v]*N_sub_cycle[v]):
                i = day2i(day,g,v)
                Ncar_dic[v][g][day] = N[v][g] - Nb[v][g][i]
        # Ncar_dic[v]['self_order'][g] stores the number of vehicle v's agent g taking car from its first day (maybe not Monday) to last day
        Ncar_dic[v]['self_order'] = {}
        for g in range(N_group[v]):
            Ncar_dic[v]['self_order'][g] = [0]*N_day_in_cycle[v]*N_sub_cycle[v]
            for i in range(N_day_in_cycle[v]*N_sub_cycle[v]):
                Ncar_dic[v]['self_order'][g][i] = N[v][g] - Nb[v][g][i]
    
    # Ncar_dic['full'][day] stores the number of both vehicles' agents on calendar day 'day'
    max_cycle_length = max(N_day_in_cycle['EV']*N_sub_cycle['EV'], N_day_in_cycle['GV']*N_sub_cycle['GV'])
    Ncar_dic['full']=[0]*max_cycle_length
    for day in range(max_cycle_length):
        car_no = 0
        for v in ['EV','GV']: # sum up both vehicles' agents on calendar day 'day'
            day_temp = day % (N_day_in_cycle[v]*N_sub_cycle[v])
            car_no += Ncar_dic[v]['full'][day_temp]
        Ncar_dic['full'][day] = car_no
    return Ncar_dic

def initialize_Q(v):
    Q = np.full([N_bus_already,N_day_in_cycle[v]*N_sub_cycle[v],N_urgent,N_action], -100, dtype=float)
    for i in range(N_bus_already):
        for j in range(N_day_in_cycle[v]*N_sub_cycle[v]):
            for k in range(N_urgent):
                if i == 1: # already taken bus
                    Q[i][j][k][1] = -200 # must take car
                elif i == 0 and (j+1) % N_day_in_cycle[v] == 0: # have not taken bus and on the last day of any sub_cycle
                    Q[i][j][k][0] = -200 # must take bus
    return Q

def generate_pi(v,car_prob):
    pi_temp = np.ones((N_bus_already,N_day_in_cycle[v]*N_sub_cycle[v],N_urgent,N_action), dtype=np.float64)
    pi_temp[:, :, :, 1] = 0.0
    for i in range(N_day_in_cycle[v]*N_sub_cycle[v]):
        if (i+1)%N_day_in_cycle[v] != 0: # not the last day of any sub_cycle
            if car_prob == None:
                p = np.random.random()
                pi_temp[0, i, 0, 0] = p
                pi_temp[0, i, 0, 1] = 1-p
            else:
                pi_temp[0, i, 0, 0] = car_prob
                pi_temp[0, i, 0, 1] = 1-car_prob
            pi_temp[0, i, 1, 1] = 0.0
        else: # is the last day of a sub_cycle            
            pi_temp[0, i, :, 0] = 0.0
            pi_temp[0, i, :, 1] = 1.0      
    return pi_temp
    
def initialize_pi(N_group, homogenerous_pi0, pc):    
    pi0 = {'EV':{},'GV':{}}
    for v in ['EV','GV']:
        for i in range(N_group[v]):
            if homogenerous_pi0 == 1: # generate same pi0 for different agent types
                pi_temp = generate_pi(v,car_prob=pc)
            else:
                pi_temp = generate_pi(v,car_prob=None)
            pi0[v][i] = copy.deepcopy(pi_temp)
    return pi0

def Value_iteration(population_pi, v, g):
    new_Q = initialize_Q(v)
    N = N_agent[v] * agent_distribution[v][g]
    if N > 0: # if this type of agent is not empty
        car_list = get_N_car_agent(population_pi, self_group=None)['full'] # return the full car number distribution, excluding representative agents
        for Q_epoch in range(Q_epochs):
            Q_temp = copy.deepcopy(new_Q)
            q_diff = 0
            for i in range(N_bus_already):
                for j in range(N_day_in_cycle[v]*N_sub_cycle[v]):
                    next_weekday = i2weekday(j+1,g)
                    day = i2day(j,g,v)
                    for k in range(N_urgent):
                        for m in range(N_action):
                            if i == 0 and (j+1)%N_day_in_cycle[v] != 0 and k == 0: # if have not taken bus, not the last day of the sub_cycle, not urgent
                                if m == 0: # if take car
                                    #car_cost = cons + a * [max(car_no - m0, 0)] ** e
                                    car_no = car_list[day]
                                    reward = cons + a * max(car_no - m0, 0) ** e
                                    next_state = [[i,j+1,0],[i,j+1,1]]
                                    next_state_prob = [1-urgent_prob[next_weekday], urgent_prob[next_weekday]]   
                                else: #if take bus
                                    reward = c2
                                    next_state = [[i+1,j+1,0],[i+1,j+1,1]]
                                    next_state_prob = [1-urgent_prob[next_weekday], urgent_prob[next_weekday]] 
                                target = 0
                                for n in range(len(next_state)):
                                    s = next_state[n]
                                    target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                new_Q[i][j][k][m] = reward + gamma * target
                            elif i == 0 and (j+1)%N_day_in_cycle[v] != 0 and k == 1: # if have not taken bus, not the last day of the sub_cycle, urgent
                                if m == 0: # if take car
                                    car_no = car_list[day]
                                    reward = cons + a * max(car_no - m0, 0) ** e
                                    next_state = [[i,j+1,0],[i,j+1,1]]
                                    next_state_prob = [1-urgent_prob[next_weekday], urgent_prob[next_weekday]]
                                    target = 0
                                    for n in range(len(next_state)):
                                        s = next_state[n]
                                        target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                    new_Q[i][j][k][m] = reward + gamma * target
                            elif i == 1: # already taken bus
                                if m == 0:
                                    car_no = car_list[day]
                                    reward = cons + a * max(car_no - m0, 0) ** e
                                    target = 0
                                    if j < N_day_in_cycle[v] * N_sub_cycle[v] -1: # not the last day of the big cycle
                                        if (j+1)%N_day_in_cycle[v] != 0: # not the last day of the sub_cycle
                                            next_state = [[i,j+1,0],[i,j+1,1]]
                                        else: # the last day of one sub_cycle (not the last sub_cycle)
                                            next_state = [[0,j+1,0],[0,j+1,1]]
                                        next_state_prob = [1-urgent_prob[next_weekday], urgent_prob[next_weekday]]
                                        for n in range(len(next_state)):
                                            s = next_state[n]
                                            target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                    new_Q[i][j][k][m] = reward + gamma * target
                            elif i == 0 and (j+1)%N_day_in_cycle[v] == 0:# if have not taken bus, the last day of the sub_cycle
                                if j == N_day_in_cycle[v]*N_sub_cycle[v]-1: # the last day of the big cycle
                                    if k == 0 and m == 1:
                                        reward = c2
                                        target = 0
                                        new_Q[i][j][k][m] = reward + gamma * target
                                    elif k == 1 and m == 1:
                                        reward = c3
                                        target = 0
                                        new_Q[i][j][k][m] = reward + gamma * target
                                else: # not the last day of the big cycle, but the last day of a sub_cycle
                                    if k == 0 and m == 1:
                                        reward = c2
                                        next_state = [[0,j+1,0],[0,j+1,1]]
                                        next_state_prob = [1-urgent_prob[next_weekday], urgent_prob[next_weekday]]
                                        target = 0
                                        for n in range(len(next_state)):
                                            s = next_state[n]
                                            target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                        new_Q[i][j][k][m] = reward + gamma * target
                                    elif k == 1 and m == 1:
                                        reward = c3
                                        next_state = [[0,j+1,0],[0,j+1,1]]
                                        next_state_prob = [1-urgent_prob[next_weekday], urgent_prob[next_weekday]]
                                        target = 0
                                        for n in range(len(next_state)):
                                            s = next_state[n]
                                            target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                        new_Q[i][j][k][m] = reward + gamma * target
                            q_diff += (new_Q[i][j][k][m] - Q_temp[i][j][k][m])**2
            if q_diff == 0:
                #print(Q_epoch,'Converged!')
                break
    reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp = get_reward(N_group, agent_distribution, population_pi, new_Q, v, g)
    
    return reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp, new_Q

def get_reward(N_group, agent_distribution, population_pi, new_Q, v, g):
    car_list = get_N_car_agent(population_pi,self_group=None)
    car_list_full = car_list['full'] # get the full car number ditribution, including representative agents
    car_list_self = car_list[v]['self_order'][g]
    N = N_agent[v] * agent_distribution[v][g]
    reward_sum = 0
    drive_cost_sum = 0
    transit_cost_sum = 0
    if N > 0:
        for i in range(N_day_in_cycle[v]*N_sub_cycle[v]):
            weekday = i2weekday(i, g)
            day = i2day(i, g, v)
            if (i+1)%N_day_in_cycle[v] != 0: # if not last day of the sub_cycle and is flexible restriction policy
                car_no = car_list_full[day]
                car_cost = cons + a * max(car_no - m0, 0) ** e
                reward_sum += c2 * (N - car_list_self[i]) + car_cost * car_list_self[i]
                drive_cost_sum += car_cost * car_list_self[i]
                transit_cost_sum += c2 * (N - car_list_self[i])
            else: # last day of the sub_cycle or theory == 'PRP'
                car_no = car_list_full[day]
                car_cost = cons + a * max(car_no - m0, 0) ** e
                reward_sum += (c2 * (1-urgent_prob[weekday]) + c3 * urgent_prob[weekday]) * (N - car_list_self[i]) + car_cost * car_list_self[i]
                drive_cost_sum += car_cost * car_list_self[i]
                transit_cost_sum += (c2 * (1-urgent_prob[weekday]) + c3 * urgent_prob[weekday]) * (N - car_list_self[i])
        reward_population_temp = reward_sum / N
        drive_cost_population_temp = drive_cost_sum / N / Total_cycle_length * 5 if N_sub_cycle[v] > 1 else drive_cost_sum / N# the average driving cost per week, drive N_day_in_cycle-1 days in total in each sub_cycle
        transit_cost_population_temp = transit_cost_sum / N / Total_cycle_length * 5 if N_sub_cycle[v] > 1 else transit_cost_sum / N# the average transit cost per week
        reward_agent_temp = new_Q[0][0][0].max() * (1-urgent_prob[0]) + new_Q[0][0][1][0] * urgent_prob[0] if len(new_Q) > 0 else 0
    else:
        reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp = 0, 0, 0, 0
    return reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp

def get_pi(N_group, reward_population, reward_agent, Q, population_pi):
    #print('reward_population:', reward_population, 'reward_agent:', reward_agent)
    new_pi = {'EV':{},'GV':{}}
    agent_pi = {'EV':{},'GV':{}}
    for v in ['EV','GV']:
        for g in range(N_group[v]):
            reward_population_temp = reward_population[v][g]            
            if reward_population_temp == 0: # implies this type has no agents, do not learn
                learning_rate = 0
            else:
                reward_agent_temp = reward_agent[v][g]
                delta_reward = abs(reward_agent_temp - reward_population_temp)
                learning_rate = min(1, beta * delta_reward / abs(reward_population_temp) + 0.001)
            temp_pi = np.zeros([N_bus_already,N_day_in_cycle[v]*N_sub_cycle[v],N_urgent,N_action])
            new_pi_temp = np.zeros([N_bus_already,N_day_in_cycle[v]*N_sub_cycle[v],N_urgent,N_action])
            for i in range(N_bus_already):
                for j in range(N_day_in_cycle[v]*N_sub_cycle[v]):
                    for k in range(N_urgent):
                        for m in range(N_action):
                            if i == 0 and (j+1)%N_day_in_cycle[v] != 0 and k == 0: # only decide in certain scenarios
                                action_max = np.where(Q[v][g][i][j][k] == np.max(Q[v][g][i][j][k]))[0]
                                if abs(Q[v][g][i][j][k][0] - Q[v][g][i][j][k][1]) < min_Q_diff: # q diff is smaller than the min_Q_diff, regard them as the same
                                    temp_pi[i][j][k][m] = population_pi[v][g][i][j][k][m] #keep the same with existing
                                else:
                                    action = action_max[0]
                                    temp_pi[i][j][k][action] = 1
            for i in range(N_bus_already):
                for j in range(N_day_in_cycle[v]*N_sub_cycle[v]):
                    for k in range(N_urgent):
                        for m in range(N_action):
                            if i == 0 and (j+1)%N_day_in_cycle[v] != 0 and k == 0: # only decide in four scenarios
                                new_pi_temp[i][j][k][m] = population_pi[v][g][i][j][k][m] * (1 - learning_rate) + temp_pi[i][j][k][m] * learning_rate
            new_pi[v][g] = copy.deepcopy(new_pi_temp)
            agent_pi[v][g] = copy.deepcopy(temp_pi)
    return new_pi, agent_pi

def get_difference(N_group, new_population_pi, agent_pi):
    diff_pi = {}
    for v in ['EV','GV']:
        diff_pi[v] = [0]*N_group[v]
    for v in ['EV','GV']:
        for g in range(N_group[v]):
            for i in range(N_bus_already):
                for j in range(N_day_in_cycle[v]*N_sub_cycle[v]):
                    for k in range(N_urgent):
                        for m in range(N_action):
                            diff_pi[v][g] += abs(new_population_pi[v][g][i][j][k][m] - agent_pi[v][g][i][j][k][m]) #** 2
    return diff_pi


def get_car_cost_function(cost_function,c2,N_agent_total):
    if cost_function == 'linear':
        cons = 0
        m0 = 0
        a = c2 / N_agent_total # cost parameter for car
        e = 1
    elif cost_function == 'quadratic':
        cons = 0
        m0 = 0
        a = c2 / (N_agent_total * N_agent_total)
        e = 2
    elif cost_function == 'piecewise':
        cons = 1
        m0 = N_agent/2
        a = (c2 - cons) / (N_agent_total - m0)
        e = 1
    return cons, m0, a, e
               
def outer_train(N_group,urgent_prob,c3,agent_distribution):  
    population_pi = initialize_pi(N_group, homogenerous_pi0, pc)  # initialize the policy for population
    for epoch in range(epochs):
        Q = {'EV':{},'GV':{}}
        reward_population = {'EV':{},'GV':{}}
        drive_cost_population = {'EV':{},'GV':{}}
        transit_cost_population = {'EV':{},'GV':{}}
        reward_agent = {'EV':{},'GV':{}}
        for v in ['EV','GV']:
            for g in range(N_group[v]):
                reward_population[v][g], drive_cost_population[v][g], transit_cost_population[v][g], reward_agent[v][g], Q[v][g] = Value_iteration(population_pi,v,g)
        new_population_pi, agent_pi = get_pi(N_group, reward_population, reward_agent, Q, population_pi)
        diff_pi = get_difference(N_group, new_population_pi, agent_pi)
        population_pi = copy.deepcopy(new_population_pi)
        sum_diff_pi = sum(sum(value_list) for value_list in diff_pi.values())
        #print(sum_diff_pi)
        #print(population_pi)
        phi_dic = {}
        for v in ['EV','GV']:
            phi_list = []
            for g in range(N_group[v]):
                phi = []
                for i in range(N_day_in_cycle[v]-1):
                    phi += [population_pi[v][g][0][i][0][1]]
                phi_list.append(phi)
            phi_dic[v] = phi_list
        if sum_diff_pi < 0.0000001: # if the current policy is almost the same with previous policy, stop iteration
            break
    return Q, reward_population, drive_cost_population, transit_cost_population, population_pi

def calculate_theoretical_LB(urgent_prob):
    mean_car_no = N_agent['EV'] * (1-1/N_day_in_cycle['EV']) + N_agent['GV'] * (1-1/N_day_in_cycle['GV'])
    min_drive_cost = cons + a * (max(mean_car_no - m0, 0) ** e)
    weekly_drive_cost = {}
    weekly_transit_cost = {}
    min_total_cost = {}
    for v in ['EV','GV']:
        p_list = []
        for i in range(5):
            j_list = []
            for k in range(N_day_in_cycle[v]):
                j_list.append((i+k)%5)
            p_temp= 1
            for j in j_list:
                p_temp *= urgent_prob[j]
            p_list.append(p_temp)
        p_c3 = statistics.mean(p_list)
        min_transit_cost = c2 * (1-p_c3) + c3 * p_c3
        weekly_drive_cost[v] = min_drive_cost*(N_day_in_cycle[v]-1)/N_day_in_cycle[v]*5 if N_sub_cycle[v]!=1 else min_drive_cost*(N_day_in_cycle[v]-1)
        weekly_transit_cost[v] = min_transit_cost/N_day_in_cycle[v]*5 if N_sub_cycle[v]!=1 else min_transit_cost
        min_total_cost[v] = (min_drive_cost * (N_day_in_cycle[v]-1)+min_transit_cost)/N_day_in_cycle[v]*5 if N_sub_cycle[v]!=1 else min_drive_cost * (N_day_in_cycle[v]-1)+min_transit_cost
    return min_total_cost, weekly_drive_cost, weekly_transit_cost

def calculate_cost_for_PRP():
    mean_car_no = N_agent['EV'] * (1-1/N_day_in_cycle['EV']) + N_agent['GV'] * (1-1/N_day_in_cycle['GV'])
    drive_cost = cons + a * (max(mean_car_no - m0, 0) ** e)
    weekly_drive_cost = {}
    weekly_transit_cost = {}
    total_cost = {}
    agent_distribution = {}
    for v in ['EV','GV']:
        agent_distribution[v] = [1/N_day_in_cycle[v]]*N_day_in_cycle[v]
        transit_cost_list = []
        for i in range(5):
            j_list = []
            for k in range(N_day_in_cycle[v]):
                j_list.append((i+k)%5)
            transit_cost_temp = []
            for j in j_list:
                transit_cost_temp.append(c2*(1-urgent_prob[j])+c3*urgent_prob[j])
            transit_cost_list.append(np.dot(transit_cost_temp,agent_distribution[v]))
        transit_cost = np.dot(transit_cost_list,[0.2]*5)
        weekly_drive_cost[v] = drive_cost*(N_day_in_cycle[v]-1)/N_day_in_cycle[v]*5 if N_sub_cycle[v]!=1 else drive_cost*(N_day_in_cycle[v]-1)
        weekly_transit_cost[v] = transit_cost/N_day_in_cycle[v]*5 if N_sub_cycle[v]!=1 else transit_cost
        total_cost[v] = (drive_cost * (N_day_in_cycle[v]-1)+transit_cost)/N_day_in_cycle[v]*5 if N_sub_cycle[v]!=1 else drive_cost * (N_day_in_cycle[v]-1)+transit_cost
    return total_cost, weekly_drive_cost, weekly_transit_cost


def get_reward_from_pi_1():
    population_pi = initialize_pi(N_group, 1, 0)
    reward_population = {'EV':{},'GV':{}}
    drive_cost_population = {'EV':{},'GV':{}}
    transit_cost_population = {'EV':{},'GV':{}}
    reward_agent = {'EV':{},'GV':{}}
    total_cost_temp = {'EV':{},'GV':{}}
    for v in ['EV','GV']:
        for g in range(N_group[v]):
            reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp = get_reward(N_group, agent_distribution, population_pi, {}, v, g)
            reward_population[v][g], drive_cost_population[v][g], transit_cost_population[v][g], reward_agent[v][g] = reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp
        total_cost_temp[v] = np.dot(list(reward_population[v].values()),agent_distribution[v]) # total cost for all N_day_in_cycle*N_sub_cycle days
    weekly_cost_temp = []
    drive_cost_temp = []
    transit_cost_temp = []
    for v in ['EV','GV']:
        weekly_cost_temp.append(total_cost_temp[v] / Total_cycle_length * 5 if N_sub_cycle[v]!=1 else total_cost_temp[v]) # average weekly cost
        drive_cost_temp.append(np.dot(list(drive_cost_population[v].values()),agent_distribution[v]))
        transit_cost_temp.append(np.dot(list(transit_cost_population[v].values()),agent_distribution[v]))
    weekly_cost = weekly_cost_temp[0]*EV_percent + weekly_cost_temp[1]*(1-EV_percent)
    print(weekly_cost)
    return weekly_cost,drive_cost_temp,transit_cost_temp


def prob_list(mode,max_pi): #vec_N_over_N    vec_N_over_N 是 vector N 除以 N
    pi_ratio = {'a':[1,1,1,1,1],'b':[1,2,3,2,1],'c':[5,4,3,2,1],'d':[1,2,3,4,5],
                'e':[1,5,1,1,1],'f':[2,2,5,2,2],'g':[1,1,4,4,1],'h':[1,3,3,3,1],
                'i':[3,2,1,2,3],'j':[1,5,3,4,2],'k':[2,4,3,5,1],'l':[3,1,3,1,3]}
    if max_pi > 0:
        urgent_prob = [x*max_pi/max(pi_ratio[mode]) for x in pi_ratio[mode]]
    else:
        urgent_prob = [0,0,0,0,0]
    return urgent_prob

def get_cost(agent_distribution,total_cost_dic,drive_cost_dic,transit_cost_dic):
    weekly_cost={}
    drive_cost={}
    transit_cost={}
    for v in ['EV','GV']:
        total_cost = np.dot(list(total_cost_dic[v].values()),agent_distribution[v]) # total cost for all N_day_in_cycle*N_sub_cycle days
        weekly_cost[v] = total_cost / Total_cycle_length * 5 if N_sub_cycle[v]!=1 else total_cost# average weekly cost
        drive_cost[v] = np.dot(list(drive_cost_dic[v].values()),agent_distribution[v])
        transit_cost[v] = np.dot(list(transit_cost_dic[v].values()),agent_distribution[v])
    return weekly_cost, drive_cost, transit_cost

def health_cost(N_day_in_cycle,N_agent):
    # c4=a+bm
    common_cycle = np.lcm(N_day_in_cycle['EV'], N_day_in_cycle['GV'])    
    a = c4_min
    b = (c4_max - c4_min)/N_agent_total
    cyclic_health_cost = a * common_cycle + b * ((N_day_in_cycle['GV'] - 1)/N_day_in_cycle['GV']) * common_cycle * N_agent['GV']
    weekly_health_cost = -cyclic_health_cost / common_cycle * 5
    #print(weekly_health_cost)
    return weekly_health_cost

def average_subject(car_list):
    car_no = [] # 周一到周五，平均每天的小汽车数量
    bus_no = []
    for k in range(5): #周一到周五
        temp = []
        for i in range(Total_cycle_length):      
            if i%5 == k:
                temp.append(car_list[i])
        car_no.append(np.mean(temp))
        bus_no.append(N_agent_total-np.mean(temp))
    return car_no,bus_no

gv_cycle = 3 # base cycle length of GV
ev_cycle = 5 # base cycle length of EV
N_bus_already = 2 # bus_already has two values: 0 means have not taken bus, 1 means have taken bus
N_day_in_cycle = {'GV':gv_cycle,'EV':ev_cycle} #{'EV':5,'GV':3}  # 4天一个限行周期
N_urgent = 2 # urgent has two values: 0 means not urgent, 1 means urgent
N_action = 2 #action has two values: taking bus or driving car

Total_cycle_length = np.lcm(np.lcm(N_day_in_cycle['EV'], N_day_in_cycle['GV']), 5) # hyper cycle length
N_group = N_day_in_cycle
N_sub_cycle = {}
for v in ['EV','GV']:
    N_sub_cycle[v] = int(Total_cycle_length/N_day_in_cycle[v])

homogenerous_pi0 = 1 # set same pi0 for different agent types if 1, 0 otherwise, setting 1 can facilitate the training especially for N_group = 5
pc = 0 # the prob of choosing car when have choice if setting homogenerous pi0
gamma = 1.0 # discount for future reward
beta = 0.01 # 0.01 # learn rate parameter
truncation = 1 # 1 if truncate the L distribution, 0 otherwise

Q_epochs = 5000  # max iteration number for Q-table value iteration
epochs = 50000  # max iteration for population convergence
N_agent_total = 24  # full population

c4_min = 1
c4_max = 3
EV_percent = 0.5
urgent_dist_mode = 'a'
max_pi = 0.5
c3 = -6*3 # bus cost when urgent

N_agent = {}
N_agent['EV'] = N_agent_total*EV_percent
N_agent['GV'] = N_agent_total*(1-EV_percent)        
c2 = -6 # bus cost when not urgent        
cost_function = 'linear' # linear, quadratic, piecewise
# car_cost = cons + a * [max(car_no - m0, 0)] ** e
cons, m0, a, e = get_car_cost_function(cost_function,c2,N_agent_total)

print('-----------------------------------')
print('Restriction cycle:', N_day_in_cycle, 'days')
urgent_prob = prob_list(urgent_dist_mode,max_pi)

min_Q_diff = 0.001 # when q_vule difference in two iterations are <  min_Q_diff, keep the previous pi, otherwise, update the pi to choose action with larger q_value
for i in range(5):
    min_Q_diff *= urgent_prob[i] # the smaller the urgency probability, the smaller the threshold

###### RUN GI-FRP ######
weekly_health_cost = health_cost(N_day_in_cycle,N_agent)
agent_distribution_equal = {}
for v in ['EV','GV']:
    agent_distribution_equal[v] = 1 if N_sub_cycle[v]!=1 else 0 # 1 if agents are equally distributed across different groups, 0 otherwise, 2 means assign all agents to first group
agent_distribution = solve_agent_distribution(N_group, urgent_prob, agent_distribution_equal)
Q, total_cost_dic, drive_cost_dic, transit_cost_dic, population_pi = outer_train(N_group,urgent_prob,c3,agent_distribution)
GI_FRP_weekly_cost, GI_FRP_drive_cost, GI_FRP_transit_cost = get_cost(agent_distribution,total_cost_dic,drive_cost_dic,transit_cost_dic)

weekly_travel_cost = 0
for v in ['EV','GV']:
    weekly_travel_cost += GI_FRP_weekly_cost[v] * N_agent[v] / N_agent_total
total_cost = weekly_travel_cost + weekly_health_cost
print('travel cost:', round(weekly_travel_cost,4) ,'health cost:', round(weekly_health_cost,4) ,'total cost:', round(total_cost,4))
          


