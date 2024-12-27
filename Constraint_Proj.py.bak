import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math

def Projection_X_N(state, action):
    return action

def Projection_Re_L2_005(state, action):
    with gp.Env(empty=True) as reacher_env:
        reacher_env.setParam('OutputFlag', 0)
        reacher_env.start()
        with gp.Model(env=reacher_env) as reacher_m:
            neta1=action[0]
            neta2=action[1]
            
            a1 = reacher_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = reacher_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            
            obj= (a1-neta1)**2+ (a2-neta2)**2
            
            reacher_m.setObjective(obj,GRB.MINIMIZE)
            
            reacher_m.addConstr((a1 * a1 + a2 * a2) <= 0.05)
            reacher_m.optimize()
            
            return reacher_m.X[0:2]

def Projection_Re_L2_01(state, action):
    with gp.Env(empty=True) as reacher_env:
        reacher_env.setParam('OutputFlag', 0)
        reacher_env.start()
        with gp.Model(env=reacher_env) as reacher_m:
            neta1=action[0]
            neta2=action[1]
            
            a1 = reacher_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = reacher_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            
            obj= (a1-neta1)**2+ (a2-neta2)**2
            
            reacher_m.setObjective(obj,GRB.MINIMIZE)
            
            reacher_m.addConstr((a1 * a1 + a2 * a2) <= 0.1)
            reacher_m.optimize()
            
            return reacher_m.X[0:2]

def Projection_Re_S_lr_L2_005(state, action):
    if(action[0]>=0):
        with gp.Env(empty=True) as reacher_env:
            reacher_env.setParam('OutputFlag', 0)
            reacher_env.start()
            with gp.Model(env=reacher_env) as reacher_m:
                neta1=action[0]
                neta2=action[1]
                
                a1 = reacher_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
                a2 = reacher_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
                
                obj= (a1-neta1)**2+ (a2-neta2)**2
                
                reacher_m.setObjective(obj,GRB.MINIMIZE)
                
                reacher_m.addConstr(((a1-0.5) * (a1-0.5) + a2 * a2) <= 0.05)
                reacher_m.optimize()
                
                return reacher_m.X[0:2]
    else:
        with gp.Env(empty=True) as reacher_env:
            reacher_env.setParam('OutputFlag', 0)
            reacher_env.start()
            with gp.Model(env=reacher_env) as reacher_m:
                neta1=action[0]
                neta2=action[1]
                
                a1 = reacher_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
                a2 = reacher_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
                
                obj= (a1-neta1)**2+ (a2-neta2)**2
                
                reacher_m.setObjective(obj,GRB.MINIMIZE)
                
                reacher_m.addConstr(((a1+0.5) * (a1+0.5) + a2 * a2) <= 0.05)
                reacher_m.optimize()
                
                return reacher_m.X[0:2]

def Projection_S_L2_01(state, action):
    with gp.Env(empty=True) as swimmer_env:
        swimmer_env.setParam('OutputFlag', 0)
        swimmer_env.start()
        with gp.Model(env=swimmer_env) as swimmer_m:
            neta1=action[0]
            neta2=action[1]
            
            a1 = swimmer_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = swimmer_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            
            obj= (a1-neta1)**2+ (a2-neta2)**2
            
            swimmer_m.setObjective(obj,GRB.MINIMIZE)
            
            swimmer_m.addConstr((a1 * a1 + a2 * a2) <= 0.1)
            swimmer_m.optimize()
            
            return swimmer_m.X[0:2]

def Projection_S_L2_05(state, action):
    with gp.Env(empty=True) as swimmer_env:
        swimmer_env.setParam('OutputFlag', 0)
        swimmer_env.start()
        with gp.Model(env=swimmer_env) as swimmer_m:
            neta1=action[0]
            neta2=action[1]
            
            a1 = swimmer_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = swimmer_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            
            obj= (a1-neta1)**2+ (a2-neta2)**2
            
            swimmer_m.setObjective(obj,GRB.MINIMIZE)
            
            swimmer_m.addConstr((a1 * a1 + a2 * a2) <= 0.5)
            swimmer_m.optimize()
            
            return swimmer_m.X[0:2]

def Projection_S_L2_1(state, action):
    with gp.Env(empty=True) as swimmer_env:
        swimmer_env.setParam('OutputFlag', 0)
        swimmer_env.start()
        with gp.Model(env=swimmer_env) as swimmer_m:
            neta1=action[0]
            neta2=action[1]
            
            a1 = swimmer_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = swimmer_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            
            obj= (a1-neta1)**2+ (a2-neta2)**2
            
            swimmer_m.setObjective(obj,GRB.MINIMIZE)
            
            swimmer_m.addConstr((a1 * a1 + a2 * a2) <= 1)
            swimmer_m.optimize()
            
            return swimmer_m.X[0:2]      

def Projection_H_L2_01(state, action):
    with gp.Env(empty=True) as hopper_env:
        hopper_env.setParam('OutputFlag', 0)
        hopper_env.start()
        with gp.Model(env=hopper_env) as hopper_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            
            a1 = hopper_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = hopper_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = hopper_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            
            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2
            
            hopper_m.setObjective(obj,GRB.MINIMIZE)
            
            hopper_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3) <= 0.1)
            hopper_m.optimize()
            
            return hopper_m.X[0:3] 

def Projection_H_L2_1(state, action):
    with gp.Env(empty=True) as hopper_env:
        hopper_env.setParam('OutputFlag', 0)
        hopper_env.start()
        with gp.Model(env=hopper_env) as hopper_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            
            a1 = hopper_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = hopper_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = hopper_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            
            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2
            
            hopper_m.setObjective(obj,GRB.MINIMIZE)
            
            hopper_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3) <= 1)
            hopper_m.optimize()
            
            return hopper_m.X[0:3] 

def Projection_HC_O20(state, action):
    with gp.Env(empty=True) as half_env:
        half_env.setParam('OutputFlag', 0)
        half_env.start()
        with gp.Model(env=half_env) as half_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            w1=state[11]
            w2=state[12]
            w3=state[13]
            w4=state[14]
            w5=state[15]
            w6=state[16]
            a1 = half_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = half_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = half_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = half_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = half_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = half_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            u1 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            abs_u1 = half_m.addVar(ub=20, name="abs_u1")
            abs_u2 = half_m.addVar(ub=20, name="abs_u2")
            abs_u3 = half_m.addVar(ub=20, name="abs_u3")
            abs_u4 = half_m.addVar(ub=20, name="abs_u4")
            abs_u5 = half_m.addVar(ub=20, name="abs_u5")
            abs_u6 = half_m.addVar(ub=20, name="abs_u6")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2
            half_m.setObjective(obj,GRB.MINIMIZE)
            
            
            half_m.addConstr(u1==a1*w1)   
            half_m.addConstr(u2==a2*w2)
            half_m.addConstr(u3==a3*w3)   
            half_m.addConstr(u4==a4*w4)
            half_m.addConstr(u5==a5*w5)   
            half_m.addConstr(u6==a6*w6)
            half_m.addConstr(abs_u1==(gp.abs_(u1)))
            half_m.addConstr(abs_u2==(gp.abs_(u2)))
            half_m.addConstr(abs_u3==(gp.abs_(u3)))
            half_m.addConstr(abs_u4==(gp.abs_(u4)))
            half_m.addConstr(abs_u5==(gp.abs_(u5)))
            half_m.addConstr(abs_u6==(gp.abs_(u6)))
            half_m.addConstr((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 ) <= 20)


            half_m.optimize()
            return half_m.X[0:6]
   
def Projection_HC_O10(state, action):
    with gp.Env(empty=True) as half_env:
        half_env.setParam('OutputFlag', 0)
        half_env.start()
        with gp.Model(env=half_env) as half_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            w1=state[11]
            w2=state[12]
            w3=state[13]
            w4=state[14]
            w5=state[15]
            w6=state[16]
            a1 = half_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = half_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = half_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = half_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = half_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = half_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            u1 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            abs_u1 = half_m.addVar(ub=10, name="abs_u1")
            abs_u2 = half_m.addVar(ub=10, name="abs_u2")
            abs_u3 = half_m.addVar(ub=10, name="abs_u3")
            abs_u4 = half_m.addVar(ub=10, name="abs_u4")
            abs_u5 = half_m.addVar(ub=10, name="abs_u5")
            abs_u6 = half_m.addVar(ub=10, name="abs_u6")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2
            half_m.setObjective(obj,GRB.MINIMIZE)
            
            
            half_m.addConstr(u1==a1*w1)   
            half_m.addConstr(u2==a2*w2)
            half_m.addConstr(u3==a3*w3)   
            half_m.addConstr(u4==a4*w4)
            half_m.addConstr(u5==a5*w5)   
            half_m.addConstr(u6==a6*w6)
            half_m.addConstr(abs_u1==(gp.abs_(u1)))
            half_m.addConstr(abs_u2==(gp.abs_(u2)))
            half_m.addConstr(abs_u3==(gp.abs_(u3)))
            half_m.addConstr(abs_u4==(gp.abs_(u4)))
            half_m.addConstr(abs_u5==(gp.abs_(u5)))
            half_m.addConstr(abs_u6==(gp.abs_(u6)))
            half_m.addConstr((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 ) <= 10)


            half_m.optimize()
            return half_m.X[0:6]       

def Projection_HC_O5(state, action):
    with gp.Env(empty=True) as half_env:
        half_env.setParam('OutputFlag', 0)
        half_env.start()
        with gp.Model(env=half_env) as half_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            w1=state[11]
            w2=state[12]
            w3=state[13]
            w4=state[14]
            w5=state[15]
            w6=state[16]
            a1 = half_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = half_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = half_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = half_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = half_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = half_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            u1 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            abs_u1 = half_m.addVar(ub=10, name="abs_u1")
            abs_u2 = half_m.addVar(ub=10, name="abs_u2")
            abs_u3 = half_m.addVar(ub=10, name="abs_u3")
            abs_u4 = half_m.addVar(ub=10, name="abs_u4")
            abs_u5 = half_m.addVar(ub=10, name="abs_u5")
            abs_u6 = half_m.addVar(ub=10, name="abs_u6")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2
            half_m.setObjective(obj,GRB.MINIMIZE)
            
            half_m.addConstr(u1==a1*w1)   
            half_m.addConstr(u2==a2*w2)
            half_m.addConstr(u3==a3*w3)   
            half_m.addConstr(u4==a4*w4)
            half_m.addConstr(u5==a5*w5)   
            half_m.addConstr(u6==a6*w6)
            half_m.addConstr(abs_u1==(gp.abs_(u1)))
            half_m.addConstr(abs_u2==(gp.abs_(u2)))
            half_m.addConstr(abs_u3==(gp.abs_(u3)))
            half_m.addConstr(abs_u4==(gp.abs_(u4)))
            half_m.addConstr(abs_u5==(gp.abs_(u5)))
            half_m.addConstr(abs_u6==(gp.abs_(u6)))
            half_m.addConstr((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 ) <= 5)

            half_m.optimize()
            return half_m.X[0:6]       

def Projection_HC_M10(state, action):
    with gp.Env(empty=True) as half_env:
        half_env.setParam('OutputFlag', 0)
        half_env.start()
        with gp.Model(env=half_env) as half_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            w1=state[11]
            w2=state[12]
            w3=state[13]
            w4=state[14]
            w5=state[15]
            w6=state[16]
            a1 = half_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = half_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = half_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = half_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = half_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = half_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            u1 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            max_u1 = half_m.addVar(lb=0, name="max_u1")
            max_u2 = half_m.addVar(lb=0, name="max_u2")
            max_u3 = half_m.addVar(lb=0, name="max_u3")
            max_u4 = half_m.addVar(lb=0, name="max_u4")
            max_u5 = half_m.addVar(lb=0, name="max_u5")
            max_u6 = half_m.addVar(lb=0, name="max_u6")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2
            half_m.setObjective(obj,GRB.MINIMIZE)
            
            
            half_m.addConstr(u1==a1*w1)   
            half_m.addConstr(u2==a2*w2)
            half_m.addConstr(u3==a3*w3)   
            half_m.addConstr(u4==a4*w4)
            half_m.addConstr(u5==a5*w5)   
            half_m.addConstr(u6==a6*w6)
            half_m.addConstr(max_u1 == gp.max_(u1, 0.0))
            half_m.addConstr(max_u2 == gp.max_(u2, 0.0))
            half_m.addConstr(max_u3 == gp.max_(u3, 0.0))
            half_m.addConstr(max_u4 == gp.max_(u4, 0.0))
            half_m.addConstr(max_u5 == gp.max_(u5, 0.0))
            half_m.addConstr(max_u6 == gp.max_(u6, 0.0))
            half_m.addConstr((max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6) <= 10)


            half_m.optimize()
            return half_m.X[0:6]       

def Projection_An_O20(state, action):
    with gp.Env(empty=True) as ant_env:
        ant_env.setParam('OutputFlag', 0)
        ant_env.start()
        with gp.Model(env=ant_env) as ant_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            neta7=action[6]
            neta8=action[7]
            w1=state[25]
            w2=state[26]
            w3=state[19]
            w4=state[20]
            w5=state[21]
            w6=state[22]
            w7=state[23]
            w8=state[24]
            a1 = ant_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = ant_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = ant_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = ant_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = ant_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = ant_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            a7 = ant_m.addVar(lb=-1,ub=1, name="a7",vtype=GRB.CONTINUOUS)
            a8 = ant_m.addVar(lb=-1,ub=1, name="a8",vtype=GRB.CONTINUOUS)
            u1 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            u7 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u7")
            u8 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u8")
            abs_u1 = ant_m.addVar(ub=20, name="abs_u1")
            abs_u2 = ant_m.addVar(ub=20, name="abs_u2")
            abs_u3 = ant_m.addVar(ub=20, name="abs_u3")
            abs_u4 = ant_m.addVar(ub=20, name="abs_u4")
            abs_u5 = ant_m.addVar(ub=20, name="abs_u5")
            abs_u6 = ant_m.addVar(ub=20, name="abs_u6")
            abs_u7 = ant_m.addVar(ub=20, name="abs_u7")
            abs_u8 = ant_m.addVar(ub=20, name="abs_u8")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2+ (a7-neta7)**2+ (a8-neta8)**2
            ant_m.setObjective(obj,GRB.MINIMIZE)
            
            
            ant_m.addConstr(u1==a1*w1)   
            ant_m.addConstr(u2==a2*w2)
            ant_m.addConstr(u3==a3*w3)   
            ant_m.addConstr(u4==a4*w4)
            ant_m.addConstr(u5==a5*w5)   
            ant_m.addConstr(u6==a6*w6)
            ant_m.addConstr(u7==a7*w7)   
            ant_m.addConstr(u8==a8*w8)
            ant_m.addConstr(abs_u1==(gp.abs_(u1)))
            ant_m.addConstr(abs_u2==(gp.abs_(u2)))
            ant_m.addConstr(abs_u3==(gp.abs_(u3)))
            ant_m.addConstr(abs_u4==(gp.abs_(u4)))
            ant_m.addConstr(abs_u5==(gp.abs_(u5)))
            ant_m.addConstr(abs_u6==(gp.abs_(u6)))
            ant_m.addConstr(abs_u7==(gp.abs_(u7)))
            ant_m.addConstr(abs_u8==(gp.abs_(u8)))
            ant_m.addConstr((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 + abs_u7 + abs_u8 ) <= 20)


            ant_m.optimize()
            return ant_m.X[0:8]       
def Projection_An_L2_2(state, action):
    with gp.Env(empty=True) as ant_env:
        ant_env.setParam('OutputFlag', 0)
        ant_env.start()
        with gp.Model(env=ant_env) as ant_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            neta7=action[6]
            neta8=action[7]
            a1 = ant_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = ant_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = ant_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = ant_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = ant_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = ant_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            a7 = ant_m.addVar(lb=-1,ub=1, name="a7",vtype=GRB.CONTINUOUS)
            a8 = ant_m.addVar(lb=-1,ub=1, name="a8",vtype=GRB.CONTINUOUS)
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2+ (a7-neta7)**2+ (a8-neta8)**2
            ant_m.setObjective(obj,GRB.MINIMIZE)
            ant_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7 + a8 * a8) <= 2)
            ant_m.optimize()
            return ant_m.X[0:8]     


def Projection_An_O30(state, action):
    with gp.Env(empty=True) as ant_env:
        ant_env.setParam('OutputFlag', 0)
        ant_env.start()
        with gp.Model(env=ant_env) as ant_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            neta7=action[6]
            neta8=action[7]
            w1=state[25]
            w2=state[26]
            w3=state[19]
            w4=state[20]
            w5=state[21]
            w6=state[22]
            w7=state[23]
            w8=state[24]
            a1 = ant_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = ant_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = ant_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = ant_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = ant_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = ant_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            a7 = ant_m.addVar(lb=-1,ub=1, name="a7",vtype=GRB.CONTINUOUS)
            a8 = ant_m.addVar(lb=-1,ub=1, name="a8",vtype=GRB.CONTINUOUS)
            u1 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            u7 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u7")
            u8 = ant_m.addVar(lb=-gp.GRB.INFINITY, name="u8")
            abs_u1 = ant_m.addVar(ub=30, name="abs_u1")
            abs_u2 = ant_m.addVar(ub=30, name="abs_u2")
            abs_u3 = ant_m.addVar(ub=30, name="abs_u3")
            abs_u4 = ant_m.addVar(ub=30, name="abs_u4")
            abs_u5 = ant_m.addVar(ub=30, name="abs_u5")
            abs_u6 = ant_m.addVar(ub=30, name="abs_u6")
            abs_u7 = ant_m.addVar(ub=30, name="abs_u7")
            abs_u8 = ant_m.addVar(ub=30, name="abs_u8")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2+ (a7-neta7)**2+ (a8-neta8)**2
            ant_m.setObjective(obj,GRB.MINIMIZE)
            
            
            ant_m.addConstr(u1==a1*w1)   
            ant_m.addConstr(u2==a2*w2)
            ant_m.addConstr(u3==a3*w3)   
            ant_m.addConstr(u4==a4*w4)
            ant_m.addConstr(u5==a5*w5)   
            ant_m.addConstr(u6==a6*w6)
            ant_m.addConstr(u7==a7*w7)   
            ant_m.addConstr(u8==a8*w8)
            ant_m.addConstr(abs_u1==(gp.abs_(u1)))
            ant_m.addConstr(abs_u2==(gp.abs_(u2)))
            ant_m.addConstr(abs_u3==(gp.abs_(u3)))
            ant_m.addConstr(abs_u4==(gp.abs_(u4)))
            ant_m.addConstr(abs_u5==(gp.abs_(u5)))
            ant_m.addConstr(abs_u6==(gp.abs_(u6)))
            ant_m.addConstr(abs_u7==(gp.abs_(u7)))
            ant_m.addConstr(abs_u8==(gp.abs_(u8)))
            ant_m.addConstr((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 + abs_u7 + abs_u8 ) <= 30)


            ant_m.optimize()
            return ant_m.X[0:8]     

def Projection_H_M_10(state, action):
    with gp.Env(empty=True) as hopper_env:
        hopper_env.setParam('OutputFlag', 0)
        hopper_env.start()
        with gp.Model(env=hopper_env) as hopper_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            w1=state[8]
            w2=state[9]
            w3=state[10]
            a1 = hopper_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = hopper_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = hopper_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            u1 = hopper_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = hopper_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = hopper_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            max_u1 = hopper_m.addVar(lb=0, name="max_u1")
            max_u2 = hopper_m.addVar(lb=0, name="max_u2")
            max_u3 = hopper_m.addVar(lb=0, name="max_u3")
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2
            hopper_m.setObjective(obj,GRB.MINIMIZE)
            
            hopper_m.addConstr(u1 == a1*w1)   
            hopper_m.addConstr(u2 == a2*w2)
            hopper_m.addConstr(u3 == a3*w3)  
            hopper_m.addConstr(max_u1 == gp.max_(u1, 0.0))
            hopper_m.addConstr(max_u2 == gp.max_(u2, 0.0))
            hopper_m.addConstr(max_u3 == gp.max_(u3, 0.0))
            hopper_m.addConstr((max_u1 + max_u2 + max_u3) <= 10)
            
            hopper_m.optimize()
            return hopper_m.X[0:3]

def Projection_W_M5(state, action):
    with gp.Env(empty=True) as walker_env:
        walker_env.setParam('OutputFlag', 0)
        walker_env.start()
        with gp.Model(env=walker_env) as walker_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            w1=state[11]
            w2=state[12]
            w3=state[13]
            w4=state[14]
            w5=state[15]
            w6=state[16]
            a1 = walker_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = walker_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = walker_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = walker_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = walker_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = walker_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            u1 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            max_u1 = walker_m.addVar(lb=0, name="max_u1")
            max_u2 = walker_m.addVar(lb=0, name="max_u2")
            max_u3 = walker_m.addVar(lb=0, name="max_u3")
            max_u4 = walker_m.addVar(lb=0, name="max_u4")
            max_u5 = walker_m.addVar(lb=0, name="max_u5")
            max_u6 = walker_m.addVar(lb=0, name="max_u6")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2
            walker_m.setObjective(obj,GRB.MINIMIZE)
            
            walker_m.addConstr(u1 == a1*w1)   
            walker_m.addConstr(u2 == a2*w2)
            walker_m.addConstr(u3 == a3*w3)  
            walker_m.addConstr(u4 == a4*w4)   
            walker_m.addConstr(u5 == a5*w5)
            walker_m.addConstr(u6 == a6*w6) 
            walker_m.addConstr(max_u1 == gp.max_(u1, 0.0))
            walker_m.addConstr(max_u2 == gp.max_(u2, 0.0))
            walker_m.addConstr(max_u3 == gp.max_(u3, 0.0))
            walker_m.addConstr(max_u4 == gp.max_(u4, 0.0))
            walker_m.addConstr(max_u5 == gp.max_(u5, 0.0))
            walker_m.addConstr(max_u6 == gp.max_(u6, 0.0))
            walker_m.addConstr((max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6) <= 5)
            
            walker_m.optimize()
            return walker_m.X[0:6]

def Projection_W_M10(state, action):
    with gp.Env(empty=True) as walker_env:
        walker_env.setParam('OutputFlag', 0)
        walker_env.start()
        with gp.Model(env=walker_env) as walker_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            w1=state[11]
            w2=state[12]
            w3=state[13]
            w4=state[14]
            w5=state[15]
            w6=state[16]
            a1 = walker_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = walker_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = walker_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = walker_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = walker_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a6 = walker_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
            u1 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            u4 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
            u5 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
            u6 = walker_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
            max_u1 = walker_m.addVar(lb=0, name="max_u1")
            max_u2 = walker_m.addVar(lb=0, name="max_u2")
            max_u3 = walker_m.addVar(lb=0, name="max_u3")
            max_u4 = walker_m.addVar(lb=0, name="max_u4")
            max_u5 = walker_m.addVar(lb=0, name="max_u5")
            max_u6 = walker_m.addVar(lb=0, name="max_u6")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2
            walker_m.setObjective(obj,GRB.MINIMIZE)
            
            walker_m.addConstr(u1 == a1*w1)   
            walker_m.addConstr(u2 == a2*w2)
            walker_m.addConstr(u3 == a3*w3)  
            walker_m.addConstr(u4 == a4*w4)   
            walker_m.addConstr(u5 == a5*w5)
            walker_m.addConstr(u6 == a6*w6) 
            walker_m.addConstr(max_u1 == gp.max_(u1, 0.0))
            walker_m.addConstr(max_u2 == gp.max_(u2, 0.0))
            walker_m.addConstr(max_u3 == gp.max_(u3, 0.0))
            walker_m.addConstr(max_u4 == gp.max_(u4, 0.0))
            walker_m.addConstr(max_u5 == gp.max_(u5, 0.0))
            walker_m.addConstr(max_u6 == gp.max_(u6, 0.0))
            walker_m.addConstr((max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6) <= 10)
            
            walker_m.optimize()
            return walker_m.X[0:6]

def Projection_MA_umaze_L2_08(state, action):
    with gp.Env(empty=True) as maze_env:
        maze_env.setParam('OutputFlag', 0)
        maze_env.start()
        with gp.Model(env=maze_env) as maze_m:
            neta1=action[0]
            neta2=action[1]
            
            a1 = maze_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = maze_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            
            obj= (a1-neta1)**2+ (a2-neta2)**2
            
            maze_m.setObjective(obj,GRB.MINIMIZE)
            
            maze_m.addConstr((a1 * a1 + a2 * a2) <= 0.8)
            maze_m.optimize()
            
            return maze_m.X[0:2]

def Projection_MA_medium_L2_08(state, action):
    with gp.Env(empty=True) as maze_env:
        maze_env.setParam('OutputFlag', 0)
        maze_env.start()
        with gp.Model(env=maze_env) as maze_m:
            neta1=action[0]
            neta2=action[1]
            
            a1 = maze_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = maze_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            
            obj= (a1-neta1)**2+ (a2-neta2)**2
            
            maze_m.setObjective(obj,GRB.MINIMIZE)
            
            maze_m.addConstr((a1 * a1 + a2 * a2) <= 0.8)
            maze_m.optimize()
            
            return maze_m.X[0:2]

def Projection_Pu_L2_1(state, action):
    with gp.Env(empty=True) as push_env:
        push_env.setParam('OutputFlag', 0)
        push_env.start()
        with gp.Model(env=push_env) as push_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]

            a1 = push_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = push_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = push_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = push_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)

            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            
            push_m.setObjective(obj,GRB.MINIMIZE)
            
            push_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3) <= 1)
            push_m.optimize()
            
            return push_m.X[0:4] 

def Projection_Sl_L2_1(state, action):
    with gp.Env(empty=True) as slide_env:
        slide_env.setParam('OutputFlag', 0)
        slide_env.start()
        with gp.Model(env=slide_env) as slide_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]

            a1 = slide_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = slide_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = slide_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = slide_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)

            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            
            slide_m.setObjective(obj,GRB.MINIMIZE)
            
            slide_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3) <= 1)
            slide_m.optimize()
            
            return slide_m.X[0:4] 


def Projection_Pandp_L2_1(state, action):
    with gp.Env(empty=True) as pandp_env:
        pandp_env.setParam('OutputFlag', 0)
        pandp_env.start()
        with gp.Model(env=pandp_env) as pandp_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]

            a1 = pandp_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = pandp_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = pandp_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = pandp_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)

            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            
            pandp_m.setObjective(obj,GRB.MINIMIZE)
            
            pandp_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3) <= 1)
            pandp_m.optimize()
            
            return pandp_m.X[0:4] 

def Projection_Sl_L2_08(state, action):
    with gp.Env(empty=True) as slide_env:
        slide_env.setParam('OutputFlag', 0)
        slide_env.start()
        with gp.Model(env=slide_env) as slide_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]

            a1 = slide_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = slide_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = slide_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = slide_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)

            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            
            slide_m.setObjective(obj,GRB.MINIMIZE)
            
            slide_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4) <= 0.8)
            slide_m.optimize()
            
            return slide_m.X[0:4] 

def Projection_Humanoid_L2_2(state, action):
    with gp.Env(empty=True) as human_env:
        human_env.setParam('OutputFlag', 0)
        human_env.start()
        with gp.Model(env=human_env) as huuman_m:

            netas = action
            variables = [huuman_m.addVar(lb=-0.4, ub=0.4, name=f"a{i+1}", vtype=GRB.CONTINUOUS) for i in range(17)]
            
            obj = sum((variables[i] - netas[i]) ** 2 for i in range(17))
            huuman_m.setObjective(obj, GRB.MINIMIZE)
            
            huuman_m.addConstr(sum(variables[i] ** 2 for i in range(17)) <= 2)
            
            huuman_m.optimize()
            
            return [var.X for var in variables]

def Projection_Humanoid_M30(state, action):
    action_to_state_map = [
        29, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44
    ]
    selected_states = [state[idx] for idx in action_to_state_map]

    with gp.Env(empty=True) as human_env:
        human_env.setParam('OutputFlag', 0)
        human_env.start()
        with gp.Model(env=human_env) as huuman_m:
            variables = [
                huuman_m.addVar(lb=-0.4, ub=0.4, name=f"a{i+1}", vtype=GRB.CONTINUOUS)
                for i in range(17)
            ]
            max_u = [
                huuman_m.addVar(lb=0, name=f"max_u{i+1}", vtype=GRB.CONTINUOUS)
                for i in range(17)
            ]
            obj = sum((variables[i] - action[i]) ** 2 for i in range(17))
            huuman_m.setObjective(obj, GRB.MINIMIZE)
            for i in range(17):
                u_i = huuman_m.addVar(lb=-gp.GRB.INFINITY, name=f"u{i+1}")
                huuman_m.addConstr(u_i == selected_states[i] * variables[i])
                huuman_m.addConstr(max_u[i] >= u_i)
                huuman_m.addConstr(max_u[i] >= 0)
            huuman_m.addConstr(gp.quicksum(max_u) <= 30 + 1e-6)
            huuman_m.optimize()
            return [var.X for var in variables]

def Projection_Humanoid_O30(state, action):
    action_to_state_map = [
        29, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44
    ]
    selected_states = [state[idx] for idx in action_to_state_map]

    with gp.Env(empty=True) as human_env:
        human_env.setParam('OutputFlag', 0)
        human_env.start()
        with gp.Model(env=human_env) as huuman_m:
            variables = [
                huuman_m.addVar(lb=-0.4, ub=0.4, name=f"a{i+1}", vtype=GRB.CONTINUOUS)
                for i in range(17)
            ]
            abs_u = [
                huuman_m.addVar(lb=0, name=f"abs_u{i+1}", vtype=GRB.CONTINUOUS)
                for i in range(17)
            ]
            obj = sum((variables[i] - action[i]) ** 2 for i in range(17))
            huuman_m.setObjective(obj, GRB.MINIMIZE)
            for i in range(17):
                u_i = huuman_m.addVar(lb=-gp.GRB.INFINITY, name=f"u{i+1}")
                huuman_m.addConstr(u_i == selected_states[i] * variables[i])
                huuman_m.addConstr(abs_u[i] >= u_i)
                huuman_m.addConstr(abs_u[i] >= -u_i)
            huuman_m.addConstr(gp.quicksum(abs_u) <= 30 + 1e-6)
            huuman_m.optimize()
            return [var.X for var in variables]


def Projection_Rea_L2_08(state, action):
    with gp.Env(empty=True) as reach_env:
        reach_env.setParam('OutputFlag', 0)
        reach_env.start()
        with gp.Model(env=reach_env) as reach_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]

            a1 = reach_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = reach_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = reach_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = reach_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)

            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            
            reach_m.setObjective(obj,GRB.MINIMIZE)
            
            reach_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4) <= 0.8)
            reach_m.optimize()
            
            return reach_m.X[0:4] 


def Projection_Pandp_L2_08(state, action):
    with gp.Env(empty=True) as pandp_env:
        pandp_env.setParam('OutputFlag', 0)
        pandp_env.start()
        with gp.Model(env=pandp_env) as pandp_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]

            a1 = pandp_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = pandp_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = pandp_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = pandp_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)

            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            
            pandp_m.setObjective(obj,GRB.MINIMIZE)
            
            pandp_m.addConstr((a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4) <= 0.8)
            pandp_m.optimize()
            
            return pandp_m.X[0:4] 

def Projection_Sl_O_001(state, action):
    with gp.Env(empty=True) as slide_env:
        slide_env.setParam('OutputFlag', 0)
        slide_env.start()
        with gp.Model(env=slide_env) as slide_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            w1=state[20]
            w2=state[21]
            w3=state[22]
            a1 = slide_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = slide_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = slide_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = slide_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            u1 = slide_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = slide_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = slide_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            abs_u1 = slide_m.addVar(ub=20, name="abs_u1")
            abs_u2 = slide_m.addVar(ub=20, name="abs_u2")
            abs_u3 = slide_m.addVar(ub=20, name="abs_u3")
            abs_u4 = slide_m.addVar(ub=20, name="abs_u4")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 
            slide_m.setObjective(obj,GRB.MINIMIZE)
            
            
            slide_m.addConstr(u1==a1*w1)   
            slide_m.addConstr(u2==a2*w2)
            slide_m.addConstr(u3==a3*w3)   
            slide_m.addConstr(abs_u1==(gp.abs_(u1)))
            slide_m.addConstr(abs_u2==(gp.abs_(u2)))
            slide_m.addConstr(abs_u3==(gp.abs_(u3)))

            slide_m.addConstr((abs_u1 + abs_u2 + abs_u3) <= 0.01)

            slide_m.optimize()
            return slide_m.X[0:4]    

def Projection_Pandp_O_001(state, action):
    with gp.Env(empty=True) as pandp_env:
        pandp_env.setParam('OutputFlag', 0)
        pandp_env.start()
        with gp.Model(env=pandp_env) as pandp_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            w1=state[20]
            w2=state[21]
            w3=state[22]
            a1 = pandp_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = pandp_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = pandp_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = pandp_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            u1 = pandp_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
            u2 = pandp_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
            u3 = pandp_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
            abs_u1 = pandp_m.addVar(ub=20, name="abs_u1")
            abs_u2 = pandp_m.addVar(ub=20, name="abs_u2")
            abs_u3 = pandp_m.addVar(ub=20, name="abs_u3")
            obj = (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2
            pandp_m.setObjective(obj,GRB.MINIMIZE)
            
            
            pandp_m.addConstr(u1==a1*w1)   
            pandp_m.addConstr(u2==a2*w2)
            pandp_m.addConstr(u3==a3*w3)  
            pandp_m.addConstr(abs_u1==(gp.abs_(u1)))
            pandp_m.addConstr(abs_u2==(gp.abs_(u2)))
            pandp_m.addConstr(abs_u3==(gp.abs_(u3)))

            pandp_m.addConstr((abs_u1 + abs_u2 + abs_u3) <= 0.01)

            pandp_m.optimize()
            return pandp_m.X[0:4]  


def Projection_BSS3z_S(state, action):
    with gp.Env(empty=True) as BSS_env:
        BSS_env.setParam('OutputFlag', 0)
        BSS_env.start()
        with gp.Model(env=BSS_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]

            a1 = bss_m.addVar(lb=0, ub=35, name="a1", vtype=GRB.INTEGER)
            a2 = bss_m.addVar(lb=0, ub=35, name="a2", vtype=GRB.INTEGER)
            a3 = bss_m.addVar(lb=0, ub=35, name="a3", vtype=GRB.INTEGER)

            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)

            bss_m.addConstr(a1+a2+a3 == 90) 

            bss_m.optimize()

            return bss_m.X[0:3] 
    
def Projection_BSS5z_S(state, action):
    with gp.Env(empty=True) as BSS_env:
        BSS_env.setParam('OutputFlag', 0)
        BSS_env.start()
        with gp.Model(env=BSS_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]
            net_a4 = action[3]
            net_a5 = action[4]

            a1 = bss_m.addVar(lb=0, ub=35, name="a1", vtype=GRB.INTEGER)
            a2 = bss_m.addVar(lb=0, ub=35, name="a2", vtype=GRB.INTEGER)
            a3 = bss_m.addVar(lb=0, ub=35, name="a3", vtype=GRB.INTEGER)
            a4 = bss_m.addVar(lb=0, ub=35, name="a4", vtype=GRB.INTEGER)
            a5 = bss_m.addVar(lb=0, ub=35, name="a5", vtype=GRB.INTEGER)

            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2+(a4-net_a4)**2+(a5-net_a5)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)

            bss_m.addConstr(a1+a2+a3+a4+a5 == 150)  # x+y+z==90

            bss_m.optimize()

            return bss_m.X[0:5] 

def Projection_BSS5z_S_D35(state, action):
    with gp.Env(empty=True) as BSS_env:
        BSS_env.setParam('OutputFlag', 0)
        BSS_env.start()
        with gp.Model(env=BSS_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]
            net_a4 = action[3]
            net_a5 = action[4]

            a1 = bss_m.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
            a2 = bss_m.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
            a3 = bss_m.addVar(lb=-1, ub=1, name="a3", vtype=GRB.CONTINUOUS)
            a4 = bss_m.addVar(lb=-1, ub=1, name="a4", vtype=GRB.CONTINUOUS)
            a5 = bss_m.addVar(lb=-1, ub=1, name="a5", vtype=GRB.CONTINUOUS)

            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2+(a4-net_a4)**2+(a5-net_a5)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)

            bss_m.addConstr(a1 + a2 + a3 + a4 + a5 == 3)

            bss_m.optimize()

            return bss_m.X[0:5]

def Projection_BSS3z_S_D40(state, action):
    with gp.Env(empty=True) as BSS_env:
        BSS_env.setParam('OutputFlag', 0)
        BSS_env.start()
        with gp.Model(env=BSS_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]

            a1 = bss_m.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
            a2 = bss_m.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
            a3 = bss_m.addVar(lb=-1, ub=1, name="a3", vtype=GRB.CONTINUOUS)

            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)

            bss_m.addConstr(a1+a2+a3 == 1) 

            bss_m.optimize()

            return bss_m.X[0:3] 

def Projection_BSS5z_S_D40(state, action):
    with gp.Env(empty=True) as BSS_env:
        BSS_env.setParam('OutputFlag', 0)
        BSS_env.start()
        with gp.Model(env=BSS_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]
            net_a4 = action[3]
            net_a5 = action[4]

            a1 = bss_m.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
            a2 = bss_m.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
            a3 = bss_m.addVar(lb=-1, ub=1, name="a3", vtype=GRB.CONTINUOUS)
            a4 = bss_m.addVar(lb=-1, ub=1, name="a4", vtype=GRB.CONTINUOUS)
            a5 = bss_m.addVar(lb=-1, ub=1, name="a5", vtype=GRB.CONTINUOUS)

            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2+(a4-net_a4)**2+(a5-net_a5)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)

            bss_m.addConstr(a1 + a2 + a3 + a4 + a5 == 2.5)

            bss_m.optimize()

            return np.array(bss_m.X[0:5])

def Projection_Sl_S(state, action):
    with gp.Env(empty=True) as slide_env:
        slide_env.setParam('OutputFlag', 0)
        slide_env.setParam('NonConvex', 2)
        slide_env.start()
        with gp.Model(env=slide_env) as slide_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            a1 = slide_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = slide_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = slide_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = slide_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a1_2 = slide_m.addVar(lb=-1,ub=1, name="a1_2",vtype=GRB.CONTINUOUS)
            a1_3 = slide_m.addVar(lb=-1,ub=1, name="a1_3",vtype=GRB.CONTINUOUS)
            a2_2 = slide_m.addVar(lb=-1,ub=1, name="a2_2",vtype=GRB.CONTINUOUS)
            a2_3 = slide_m.addVar(lb=-1,ub=1, name="a2_3",vtype=GRB.CONTINUOUS)
            a3_2 = slide_m.addVar(lb=-1,ub=1, name="a3_2",vtype=GRB.CONTINUOUS)
            m1 = slide_m.addVar(lb=-19,ub=17, name="m1",vtype=GRB.CONTINUOUS)
            m2 = slide_m.addVar(lb=-2025,ub=2025, name="m2",vtype=GRB.CONTINUOUS)
            m3 = slide_m.addVar(lb=-1,ub=1, name="m3",vtype=GRB.CONTINUOUS)
            m1_3 = slide_m.addVar(lb=-6859,ub=6137, name="m1_3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2
            slide_m.setObjective(obj,GRB.MINIMIZE)
            slide_m.addGenConstrPow(a1, a1_2, 2)
            slide_m.addGenConstrPow(a1, a1_3, 3)
            slide_m.addGenConstrPow(a2, a2_2, 2)
            slide_m.addGenConstrPow(a2, a2_3, 3)
            slide_m.addGenConstrPow(a3, a3_2, 2)
            slide_m.addConstr(m1 == (9 * a1_2 + 9 * a2_2 - 1))
            slide_m.addConstr(m2 == (2025 * a1_2 * a2_3))
            #slide_m.addConstr(m3 == (a3_2))
            slide_m.addGenConstrPow(m1, m1_3, 3)
            slide_m.addConstr(m1_3 - m2 <= 0)
            #slide_m.addConstr(m1_3 - m2 + m3 <= 0)
            slide_m.optimize()
            #((3x)^2+(3y)^2-1)^3 - 25*(3x)^2*(3y)^3<0
            return slide_m.X[0:4]


def Projection_BSS5z_S2(state, action):
    with gp.Env(empty=True) as bss_env:
        bss_env.setParam('OutputFlag', 0)
        bss_env.setParam('NonConvex', 2)
        bss_env.start()
        with gp.Model(env=bss_env) as bss_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            a1 = bss_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = bss_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = bss_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = bss_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = bss_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a1_exp = bss_m.addVar(name="a1_exp",vtype=GRB.CONTINUOUS)
            a2_exp = bss_m.addVar(name="a2_exp",vtype=GRB.CONTINUOUS)
            a3_exp = bss_m.addVar(name="a3_exp",vtype=GRB.CONTINUOUS)
            a4_exp = bss_m.addVar(name="a4_exp",vtype=GRB.CONTINUOUS)
            a5_exp = bss_m.addVar(name="a5_exp",vtype=GRB.CONTINUOUS)
            a_sum_exp = bss_m.addVar(name="a_sum_exp",vtype=GRB.CONTINUOUS)
            a_sum_exp_revise = bss_m.addVar(name="a_sum_exp_revise",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2
            bss_m.setObjective(obj,GRB.MINIMIZE)
            bss_m.addGenConstrExp(a1, a1_exp)
            bss_m.addGenConstrExp(a2, a2_exp)
            bss_m.addGenConstrExp(a3, a3_exp)
            bss_m.addGenConstrExp(a4, a4_exp)
            bss_m.addGenConstrExp(a5, a5_exp)
            bss_m.addGenConstrPow(a_sum_exp, a_sum_exp_revise, -1)
            bss_m.addConstr(a_sum_exp == (a1_exp + a2_exp + a3_exp + a4_exp + a5_exp))
            bss_m.addConstr(a1_exp * a_sum_exp_revise <= 35/150)
            bss_m.addConstr(a2_exp * a_sum_exp_revise <= 35/150)
            bss_m.addConstr(a3_exp * a_sum_exp_revise <= 35/150)
            bss_m.addConstr(a4_exp * a_sum_exp_revise <= 35/150)
            bss_m.addConstr(a5_exp * a_sum_exp_revise <= 35/150)
            bss_m.addConstr(a1_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a2_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a3_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a4_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a5_exp * a_sum_exp_revise >= 10/150)
            bss_m.optimize()
            return bss_m.X[0:5]

def Projection_BSS5z_S2_D40(state, action):
    with gp.Env(empty=True) as bss_env:
        bss_env.setParam('OutputFlag', 0)
        bss_env.setParam('NonConvex', 2)
        bss_env.start()
        with gp.Model(env=bss_env) as bss_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            a1 = bss_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = bss_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = bss_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = bss_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = bss_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a1_exp = bss_m.addVar(lb=0.36,ub=2.72, name="a1_exp",vtype=GRB.CONTINUOUS)
            a2_exp = bss_m.addVar(lb=0.36,ub=2.72, name="a2_exp",vtype=GRB.CONTINUOUS)
            a3_exp = bss_m.addVar(lb=0.36,ub=2.72, name="a3_exp",vtype=GRB.CONTINUOUS)
            a4_exp = bss_m.addVar(lb=0.36,ub=2.72, name="a4_exp",vtype=GRB.CONTINUOUS)
            a5_exp = bss_m.addVar(lb=0.36,ub=2.72, name="a5_exp",vtype=GRB.CONTINUOUS)
            a_sum_exp = bss_m.addVar(lb=1.8,ub=13.6, name="a_sum_exp",vtype=GRB.CONTINUOUS)
            a_sum_exp_revise = bss_m.addVar(lb=0,ub=1, name="a_sum_exp_revise",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2
            bss_m.setObjective(obj,GRB.MINIMIZE)
            bss_m.addGenConstrExp(a1, a1_exp)
            bss_m.addGenConstrExp(a2, a2_exp)
            bss_m.addGenConstrExp(a3, a3_exp)
            bss_m.addGenConstrExp(a4, a4_exp)
            bss_m.addGenConstrExp(a5, a5_exp)
            bss_m.addGenConstrPow(a_sum_exp, a_sum_exp_revise, -1)
            bss_m.addConstr(a_sum_exp == (a1_exp + a2_exp + a3_exp + a4_exp + a5_exp))
            bss_m.addConstr(a1_exp * a_sum_exp_revise <= 40/150)
            bss_m.addConstr(a2_exp * a_sum_exp_revise <= 40/150)
            bss_m.addConstr(a3_exp * a_sum_exp_revise <= 40/150)
            bss_m.addConstr(a4_exp * a_sum_exp_revise <= 40/150)
            bss_m.addConstr(a5_exp * a_sum_exp_revise <= 40/150)
            '''
            bss_m.addConstr(a1_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a2_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a3_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a4_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a5_exp * a_sum_exp_revise >= 10/150)
            '''
            bss_m.optimize()
            return bss_m.X[0:5]

def Projection_BSS5z_S2_D40_ver2(state, action):
    with gp.Env(empty=True) as bss_env:
        bss_env.setParam('OutputFlag', 0)
        bss_env.setParam('NonConvex', 2)
        bss_env.start()
        with gp.Model(env=bss_env) as bss_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            a1 = bss_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = bss_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = bss_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = bss_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a5 = bss_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
            a1_add = bss_m.addVar(lb=1e-6,ub=2 + 1e-6, name="a1_add",vtype=GRB.CONTINUOUS)
            a2_add = bss_m.addVar(lb=1e-6,ub=2 + 1e-6, name="a2_add",vtype=GRB.CONTINUOUS)
            a3_add = bss_m.addVar(lb=1e-6,ub=2 + 1e-6, name="a3_add",vtype=GRB.CONTINUOUS)
            a4_add = bss_m.addVar(lb=1e-6,ub=2 + 1e-6, name="a4_add",vtype=GRB.CONTINUOUS)
            a5_add = bss_m.addVar(lb=1e-6,ub=2 + 1e-6, name="a5_add",vtype=GRB.CONTINUOUS)
            a_sum_add = bss_m.addVar(lb=5e-6,ub=10+5e-6, name="a_sum_add",vtype=GRB.CONTINUOUS)
            a_sum_add_revise = bss_m.addVar(name="a_sum_exp_revise",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2
            bss_m.setObjective(obj,GRB.MINIMIZE)
            bss_m.addConstr(a1_add == a1 + 1e-6) 
            bss_m.addConstr(a2_add == a2 + 1e-6) 
            bss_m.addConstr(a3_add == a3 + 1e-6) 
            bss_m.addConstr(a4_add == a4 + 1e-6) 
            bss_m.addConstr(a5_add == a5 + 1e-6) 
            bss_m.addConstr(a_sum_add == (a1_add + a2_add + a3_add + a4_add + a5_add))
            bss_m.addGenConstrPow(a_sum_add, a_sum_add_revise, -1)
            bss_m.addConstr(a1_add * a_sum_add_revise <= 40/150)
            bss_m.addConstr(a2_add * a_sum_add_revise <= 40/150)
            bss_m.addConstr(a3_add * a_sum_add_revise <= 40/150)
            bss_m.addConstr(a4_add * a_sum_add_revise <= 40/150)
            bss_m.addConstr(a5_add * a_sum_add_revise <= 40/150)
            '''
            bss_m.addConstr(a1_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a2_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a3_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a4_exp * a_sum_exp_revise >= 10/150)
            bss_m.addConstr(a5_exp * a_sum_exp_revise >= 10/150)
            '''
            bss_m.optimize()
            return bss_m.X[0:5]


def Projection_BSS3z_S2_INT40(state, action):
    with gp.Env(empty=True) as bss_env:
        bss_env.setParam('OutputFlag', 0)
        bss_env.start()
        with gp.Model(env=bss_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]

            a1 = bss_m.addVar(lb=10, ub=40, name="a1", vtype=GRB.INTEGER)
            a2 = bss_m.addVar(lb=10, ub=40, name="a2", vtype=GRB.INTEGER)
            a3 = bss_m.addVar(lb=10, ub=40, name="a3", vtype=GRB.INTEGER)
            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)
            bss_m.addConstr(a1+a2+a3 == 90)  # x+y+z==90
            bss_m.optimize()
            return bss_m.X[0:3]

def Projection_BSS5z_S2_INT40(state, action):
    with gp.Env(empty=True) as bss_env:
        bss_env.setParam('OutputFlag', 0)
        bss_env.start()
        with gp.Model(env=bss_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]
            net_a4 = action[3]
            net_a5 = action[4]

            a1 = bss_m.addVar(lb=0, ub=40, name="a1", vtype=GRB.INTEGER)
            a2 = bss_m.addVar(lb=0, ub=40, name="a2", vtype=GRB.INTEGER)
            a3 = bss_m.addVar(lb=0, ub=40, name="a3", vtype=GRB.INTEGER)
            a4 = bss_m.addVar(lb=0, ub=40, name="a4", vtype=GRB.INTEGER)
            a5 = bss_m.addVar(lb=0, ub=40, name="a5", vtype=GRB.INTEGER)
            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2+(a4-net_a4)**2+(a5-net_a5)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)
            bss_m.addConstr(a1+a2+a3+a4+a5 == 150)  # x+y+z==90
            bss_m.optimize()
            return bss_m.X[0:5]

def Projection_BSS5z_S2_INT35(state, action):
    with gp.Env(empty=True) as bss_env:
        bss_env.setParam('OutputFlag', 0)
        bss_env.start()
        with gp.Model(env=bss_env) as bss_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]
            net_a4 = action[3]
            net_a5 = action[4]

            a1 = bss_m.addVar(lb=10, ub=35, name="a1", vtype=GRB.INTEGER)
            a2 = bss_m.addVar(lb=10, ub=35, name="a2", vtype=GRB.INTEGER)
            a3 = bss_m.addVar(lb=10, ub=35, name="a3", vtype=GRB.INTEGER)
            a4 = bss_m.addVar(lb=10, ub=35, name="a4", vtype=GRB.INTEGER)
            a5 = bss_m.addVar(lb=10, ub=35, name="a5", vtype=GRB.INTEGER)
            obj = (a1-net_a1)**2 + (a2-net_a2)**2 + \
                (a3-net_a3)**2+(a4-net_a4)**2+(a5-net_a5)**2
            bss_m.setObjective(obj, GRB.MINIMIZE)
            bss_m.addConstr(a1+a2+a3+a4+a5 == 150)  # x+y+z==90
            bss_m.optimize()
            return bss_m.X[0:5]


def Projection_Sl_S_ellipsoid2(state, action):
    cos20 = math.cos(20*math.pi/180)
    cos40 = math.cos(40*math.pi/180)
    sin20 = math.sin(20*math.pi/180)
    sin40 = math.sin(40*math.pi/180)
    diff1 = 0
    diffa1 = []
    diff2 = 0
    diffa2 = []
    with gp.Env(empty=True) as slide_env1:
        slide_env1.setParam('OutputFlag', 0)
        slide_env1.setParam('NonConvex', 2)
        slide_env1.start()
        with gp.Model(env=slide_env1) as slide_m1:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            a1 = slide_m1.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = slide_m1.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = slide_m1.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = slide_m1.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            v1 = slide_m1.addVar(name="v1",vtype=GRB.CONTINUOUS)
            v2 = slide_m1.addVar(name="v2",vtype=GRB.CONTINUOUS)
            v3 = slide_m1.addVar(name="v3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            slide_m1.setObjective(obj,GRB.MINIMIZE)
            slide_m1.addConstr(v1 == (cos40 * a1 + sin40 * a3))
            slide_m1.addConstr(v2 == (a2))
            slide_m1.addConstr(v3 == (sin40 * a1 - cos40 * a3))
            slide_m1.addConstr(v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
            slide_m1.optimize()
            diff1 = slide_m1.objVal
            diffa1 = slide_m1.X[0:4]
    with gp.Env(empty=True) as slide_env2:
        slide_env2.setParam('OutputFlag', 0)
        slide_env2.setParam('NonConvex', 2)
        slide_env2.start()
        with gp.Model(env=slide_env2) as slide_m2:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            a1 = slide_m2.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = slide_m2.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = slide_m2.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = slide_m2.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            v1 = slide_m2.addVar(name="v1",vtype=GRB.CONTINUOUS)
            v2 = slide_m2.addVar(name="v2",vtype=GRB.CONTINUOUS)
            v3 = slide_m2.addVar(name="v3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            slide_m2.setObjective(obj,GRB.MINIMIZE)
            slide_m2.addConstr(v1 == a1)
            slide_m2.addConstr(v2 == (cos20 * a2 + sin20 * a3))
            slide_m2.addConstr(v3 == (sin20 * a2 - cos20 * a3))
            slide_m2.addConstr(v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
            slide_m2.optimize()
            diff2 = slide_m2.objVal
            diffa2 = slide_m2.X[0:4]
    if diff1 > diff2:
        return diffa2
    return diffa1

def Projection_PandP_S_ellipsoid2(state, action):
    cos20 = math.cos(20*math.pi/180)
    cos40 = math.cos(40*math.pi/180)
    sin20 = math.sin(20*math.pi/180)
    sin40 = math.sin(40*math.pi/180)
    diff1 = 0
    diffa1 = []
    diff2 = 0
    diffa2 = []
    with gp.Env(empty=True) as pandp_env1:
        pandp_env1.setParam('OutputFlag', 0)
        pandp_env1.setParam('NonConvex', 2)
        pandp_env1.start()
        with gp.Model(env=pandp_env1) as pandp_m1:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            a1 = pandp_m1.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = pandp_m1.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = pandp_m1.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = pandp_m1.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            v1 = pandp_m1.addVar(name="v1",vtype=GRB.CONTINUOUS)
            v2 = pandp_m1.addVar(name="v2",vtype=GRB.CONTINUOUS)
            v3 = pandp_m1.addVar(name="v3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            pandp_m1.setObjective(obj,GRB.MINIMIZE)
            pandp_m1.addConstr(v1 == (cos40 * a1 + sin40 * a3))
            pandp_m1.addConstr(v2 == (a2))
            pandp_m1.addConstr(v3 == (sin40 * a1 - cos40 * a3))
            pandp_m1.addConstr(v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
            pandp_m1.optimize()
            diff1 = pandp_m1.objVal
            diffa1 = pandp_m1.X[0:4]
    with gp.Env(empty=True) as pandp_env2:
        pandp_env2.setParam('OutputFlag', 0)
        pandp_env2.setParam('NonConvex', 2)
        pandp_env2.start()
        with gp.Model(env=pandp_env2) as pandp_m2:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            a1 = pandp_m2.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = pandp_m2.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = pandp_m2.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = pandp_m2.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            v1 = pandp_m2.addVar(name="v1",vtype=GRB.CONTINUOUS)
            v2 = pandp_m2.addVar(name="v2",vtype=GRB.CONTINUOUS)
            v3 = pandp_m2.addVar(name="v3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            pandp_m2.setObjective(obj,GRB.MINIMIZE)
            pandp_m2.addConstr(v1 == a1)
            pandp_m2.addConstr(v2 == (cos20 * a2 + sin20 * a3))
            pandp_m2.addConstr(v3 == (sin20 * a2 - cos20 * a3))
            pandp_m2.addConstr(v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
            pandp_m2.optimize()
            diff2 = pandp_m2.objVal
            diffa2 = pandp_m2.X[0:4]
    if diff1 > diff2:
        return diffa2
    return diffa1

def Projection_Pu_S(state, action):
    with gp.Env(empty=True) as push_env:
        push_env.setParam('OutputFlag', 0)
        push_env.setParam('NonConvex', 2)
        push_env.start()
        with gp.Model(env=push_env) as push_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            a1 = push_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = push_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = push_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = push_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            a1_2 = push_m.addVar(lb=-1,ub=1, name="a1_2",vtype=GRB.CONTINUOUS)
            a1_3 = push_m.addVar(lb=-1,ub=1, name="a1_3",vtype=GRB.CONTINUOUS)
            a2_2 = push_m.addVar(lb=-1,ub=1, name="a2_2",vtype=GRB.CONTINUOUS)
            a2_3 = push_m.addVar(lb=-1,ub=1, name="a2_3",vtype=GRB.CONTINUOUS)
            a3_2 = push_m.addVar(lb=-1,ub=1, name="a3_2",vtype=GRB.CONTINUOUS)
            m1 = push_m.addVar(lb=-19,ub=17, name="m1",vtype=GRB.CONTINUOUS)
            m2 = push_m.addVar(lb=-2025,ub=2025, name="m2",vtype=GRB.CONTINUOUS)
            m3 = push_m.addVar(lb=-1,ub=1, name="m3",vtype=GRB.CONTINUOUS)
            m1_3 = push_m.addVar(lb=-6859,ub=6137, name="m1_3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2
            push_m.setObjective(obj,GRB.MINIMIZE)
            push_m.addGenConstrPow(a1, a1_2, 2)
            push_m.addGenConstrPow(a1, a1_3, 3)
            push_m.addGenConstrPow(a2, a2_2, 2)
            push_m.addGenConstrPow(a2, a2_3, 3)
            push_m.addGenConstrPow(a3, a3_2, 2)
            push_m.addConstr(m1 == (9 * a1_2 + 9 * a2_2 - 1))
            push_m.addConstr(m2 == (2025 * a1_2 * a2_3))
            #push_m.addConstr(m3 == (a3_2))
            push_m.addGenConstrPow(m1, m1_3, 3)
            push_m.addConstr(m1_3 - m2 <= 0)
            #push_m.addConstr(m1_3 - m2 + m3 <= 0)
            push_m.optimize()
            #((3x)^2+(3y)^2-1)^3 - 25*(3x)^2*(3y)^3<0
            return push_m.X[0:4]

def Projection_Point_Safe(state, action):
    with gp.Env(empty=True) as point_env:
        point_env.setParam('OutputFlag', 0)
        point_env.setParam('NonConvex', 2)
        point_env.start()
        with gp.Model(env=point_env) as point_m1:
            neta1=action[0]
            neta2=action[1]
            a1 = point_m1.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = point_m1.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            v1 = point_m1.addVar(name="v1",vtype=GRB.CONTINUOUS)
            v2 = point_m1.addVar(name="v2",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 
            point_m1.setObjective(obj,GRB.MINIMIZE)
            point_m1.addConstr(v1 <= 0.5)
            point_m1.addConstr(v2 <= 0.5)
            point_m1.optimize()
            return point_m1.X[0:2]

def Projection_Point_Safe2(state, action):
    with gp.Env(empty=True) as point_env:
        point_env.setParam('OutputFlag', 0)
        point_env.setParam('NonConvex', 2)
        point_env.start()
        with gp.Model(env=point_env) as point_m1:
            neta1=action[0]
            neta2=action[1]
            a1 = point_m1.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = point_m1.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 
            point_m1.setObjective(obj,GRB.MINIMIZE)
            point_m1.addConstr(a1 <= 0.7)
            point_m1.addConstr(a1 >= 0.3)
            point_m1.addConstr(a2 <= 0.7)
            point_m1.addConstr(a2 >= 0.3)
            point_m1.optimize()
            return point_m1.X[0:2]

def Projection_Point_Safe3(state, action):
    with gp.Env(empty=True) as point_env:
        point_env.setParam('OutputFlag', 0)
        point_env.setParam('NonConvex', 2)
        point_env.start()
        with gp.Model(env=point_env) as point_m1:
            neta1 = action[0]
            neta2 = action[1]
            a1 = point_m1.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
            a2 = point_m1.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
            obj = (a1 - neta1) ** 2 + (a2 - neta2) ** 2
            point_m1.setObjective(obj, GRB.MINIMIZE)
            a1_lower_bound = 0.3
            a1_upper_bound = 0.7
            a2_lower_bound = 0.3
            a2_upper_bound = 0.7
            if neta1 >= 0:
                a1_lower_bound = 0.3
                a1_upper_bound = 0.7
            else:
                a1_lower_bound = -0.7
                a1_upper_bound = -0.3
            if neta2 >= 0:
                a2_lower_bound = 0.3
                a2_upper_bound = 0.7
            else:
                a2_lower_bound = -0.7
                a2_upper_bound = -0.3
            point_m1.addConstr(a1 <= a1_upper_bound)
            point_m1.addConstr(a1 >= a1_lower_bound)
            point_m1.addConstr(a2 <= a2_upper_bound)
            point_m1.addConstr(a2 >= a2_lower_bound)
            point_m1.optimize()
            return point_m1.X[0:2]

def Projection_Point_Safe4(state, action):
    with gp.Env(empty=True) as point_env:
        point_env.setParam('OutputFlag', 0)
        point_env.setParam('NonConvex', 2)
        point_env.start()
        with gp.Model(env=point_env) as point_m1:
            neta1=action[0]
            neta2=action[1]
            a1 = point_m1.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = point_m1.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 
            point_m1.setObjective(obj,GRB.MINIMIZE)
            point_m1.addConstr(a1 <= 0.5)
            point_m1.addConstr(a1 >= -0.5)
            point_m1.addConstr(a2 <= 0.5)
            point_m1.addConstr(a2 >= -0.5)
            point_m1.optimize()
            return point_m1.X[0:2]

def Projection_Pu_S_ellipsoid2(state, action):
    cos20 = math.cos(20*math.pi/180)
    cos40 = math.cos(40*math.pi/180)
    cos70 = math.cos(70*math.pi/180)
    sin20 = math.sin(20*math.pi/180)
    sin40 = math.sin(40*math.pi/180)
    sin70 = math.sin(70*math.pi/180)
    diff1 = 0
    diffa1 = []
    diff2 = 0
    diffa2 = []
    with gp.Env(empty=True) as push_env1:
        push_env1.setParam('OutputFlag', 0)
        push_env1.setParam('NonConvex', 2)
        push_env1.start()
        with gp.Model(env=push_env1) as push_m1:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            a1 = push_m1.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = push_m1.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = push_m1.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = push_m1.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            v1 = push_m1.addVar(name="v1",vtype=GRB.CONTINUOUS)
            v2 = push_m1.addVar(name="v2",vtype=GRB.CONTINUOUS)
            v3 = push_m1.addVar(name="v3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            push_m1.setObjective(obj,GRB.MINIMIZE)
            push_m1.addConstr(v1 == (cos40 * a1 + sin40 * a3))
            push_m1.addConstr(v2 == (a2))
            push_m1.addConstr(v3 == (sin40 * a1 - cos40 * a3))
            push_m1.addConstr(v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
            push_m1.optimize()
            diff1 = push_m1.objVal
            diffa1 = push_m1.X[0:4]
    with gp.Env(empty=True) as push_env2:
        push_env2.setParam('OutputFlag', 0)
        push_env2.setParam('NonConvex', 2)
        push_env2.start()
        with gp.Model(env=push_env2) as push_m2:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            a1 = push_m2.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
            a2 = push_m2.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
            a3 = push_m2.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
            a4 = push_m2.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
            v1 = push_m2.addVar(name="v1",vtype=GRB.CONTINUOUS)
            v2 = push_m2.addVar(name="v2",vtype=GRB.CONTINUOUS)
            v3 = push_m2.addVar(name="v3",vtype=GRB.CONTINUOUS)
            obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2
            push_m2.setObjective(obj,GRB.MINIMIZE)
            push_m2.addConstr(v1 == a1)
            push_m2.addConstr(v2 == (cos20 * a2 + sin20 * a3))
            push_m2.addConstr(v3 == (sin20 * a2 - cos20 * a3))
            push_m2.addConstr(v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
            push_m2.optimize()
            diff2 = push_m2.objVal
            diffa2 = push_m2.X[0:4]
    if diff1 > diff2:
        return diffa2
    return diffa1

def Projection_NSFnetV3(state, action):
    with gp.Env(empty=True) as net_env:
        net_env.setParam('OutputFlag', 0)
        net_env.start()
        with gp.Model(env=net_env) as net_m:
            # Extract action values
            neta = [action[i] for i in range(len(action))]

            # Define decision variables
            a = [net_m.addVar(lb=-1, ub=1, name=f"a{i+1}", vtype=GRB.CONTINUOUS) for i in range(len(action))]

            # Objective function to minimize squared differences
            obj = sum((a[i] - neta[i]) ** 2 for i in range(len(action)))
            net_m.setObjective(obj, GRB.MINIMIZE)

            # Add constraints based on the provided indices
            constraints = [
                (16, [0, 3, 6]),
                (17, [1, 2, 14, 16]),
                (18, []),
                (19, [0, 3, 15]),
                (20, [11, 19]),
                (21, [0, 3, 6, 10, 15]),
                (22, [2, 9, 11, 14, 17, 18]),
                (23, [3, 4, 15]),
                (24, [0, 5, 7, 10]),
                (25, [4, 12, 14, 17, 18]),
                (26, [3, 12, 14, 15, 18]),
                (27, [2, 9, 11]),
                (28, [3, 8, 13]),
                (29, [6, 7, 8, 13]),
                (30, [5, 9, 11, 13])
            ]

            # Loop to add constraints dynamically
            for idx, indices in constraints:
                if indices:  # Only add constraint if there are indices
                    constraint_value = 2 - len(indices)  # Determine constraint value based on number of indices
                    net_m.addConstr(gp.quicksum(a[i] for i in indices) <= constraint_value, name=f"constraint_{idx}")

            # Optimize model
            net_m.optimize()

            # Extract optimized values
            return [a[i].X for i in range(len(action))]


def Projection_sumo_c8(state, action):
    import gurobipy as gp
    from gurobipy import GRB

    with gp.Env(empty=True) as net_env:
        net_env.setParam('OutputFlag', 0) 
        net_env.start()
        with gp.Model(env=net_env) as net_m:
            n = len(action)  
            neta = action

            a = [net_m.addVar(lb=-1, ub=1, name=f"a{i+1}", vtype=GRB.CONTINUOUS) for i in range(n)]

            obj = gp.quicksum((a[i] - neta[i]) ** 2 for i in range(n))
            net_m.setObjective(obj, GRB.MINIMIZE)

            for i in range(0, n, 4):
                group_sum = gp.quicksum(a[i:i+4])
                net_m.addConstr(group_sum >= -3.2, name=f"group_sum_lower_{i//4}")
                net_m.addConstr(group_sum <= 2.4, name=f"group_sum_upper_{i//4}")

            net_m.optimize()

            if net_m.status == GRB.OPTIMAL:
                return [a[i].X for i in range(n)]
            else:
                raise ValueError("Optimization did not converge.")