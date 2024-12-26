import numpy as np
import math

def Check_X_N(state, action):
    return True, 0

def Check_L2_005(state, action):
    a1=action[0]
    a2=action[1]
    test = ((a1 * a1 + a2 * a2) <= 0.05+1e-6)
    dif = a1 * a1 + a2 * a2 - 0.05     
    return test, dif

def Check_L2_01(state, action):
    a1=action[0]
    a2=action[1]
    test = ((a1 * a1 + a2 * a2) <= 0.1+1e-6)
    dif = a1 * a1 + a2 * a2 - 0.1     
    return test, dif

def Check_L2_05(state, action):
    a1=action[0]
    a2=action[1]
    test = ((a1 * a1 + a2 * a2) <= 0.5+1e-6)
    dif = a1 * a1 + a2 * a2 - 0.5    
    return test, dif

def Check_L2_08(state, action):
    a1=action[0]
    a2=action[1]
    test = ((a1 * a1 + a2 * a2) <= 0.8+1e-6)
    dif = a1 * a1 + a2 * a2 - 0.8  
    return test, dif

def Check_L2_1(state, action):
    a1=action[0]
    a2=action[1]
    test = (a1 * a1 + a2 * a2 <= 1+1e-6)
    dif = a1 * a1 + a2 * a2 - 1     
    return test, dif



def Check_Re_S_lr_L2_005(state, action):
    a1=action[0]
    a2=action[1]
    if(action[0]>=0):
        test = (((a1-0.5) * (a1-0.5) + a2 * a2)<= 0.05+1e-6)
        dif = (((a1-0.5) * (a1-0.5) + a2 * a2) - 0.05)  
    else:
        test = (((a1+0.5) * (a1+0.5) + a2 * a2)<= 0.05+1e-6)
        dif = (((a1+0.5) * (a1+0.5) + a2 * a2) - 0.05)  
    return test, dif

def Check_HC_O5(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    abs_u1 = abs(a1 * w1)
    abs_u2 = abs(a2 * w2)
    abs_u3 = abs(a3 * w3)
    abs_u4 = abs(a4 * w4)
    abs_u5 = abs(a5 * w5)
    abs_u6 = abs(a6 * w6)
    test = ((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6) <= 5+1e-6)
    dif = abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 - 5   
    return test, dif

def Check_HC_O10(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    abs_u1 = abs(a1 * w1)
    abs_u2 = abs(a2 * w2)
    abs_u3 = abs(a3 * w3)
    abs_u4 = abs(a4 * w4)
    abs_u5 = abs(a5 * w5)
    abs_u6 = abs(a6 * w6)
    test = ((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6) <= 10+1e-6)
    dif = abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 - 10    
    return test, dif

def Check_HC_M10(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    max_u1 = max(a1 * w1, 0)
    max_u2 = max(a2 * w2, 0)
    max_u3 = max(a3 * w3, 0)
    max_u4 = max(a4 * w4, 0)
    max_u5 = max(a5 * w5, 0)
    max_u6 = max(a6 * w6, 0)
    test = ((max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6) <= 10+1e-6)
    dif = max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6 - 10    
    return test, dif

def Check_HC_O20(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    abs_u1 = abs(a1 * w1)
    abs_u2 = abs(a2 * w2)
    abs_u3 = abs(a3 * w3)
    abs_u4 = abs(a4 * w4)
    abs_u5 = abs(a5 * w5)
    abs_u6 = abs(a6 * w6)
    test = ((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6) <= 20+1e-6)
    dif = abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 - 20    
    return test, dif

def Check_An_L2_2(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    a7=action[6]
    a8=action[7]
    test = (a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7 + a8 * a8 <= 2+1e-6)
    dif = a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7 + a8 * a8 - 2
    return test, dif

def Check_An_O20(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    a7=action[6]
    a8=action[7]
    w1=state[25]
    w2=state[26]
    w3=state[19]
    w4=state[20]
    w5=state[21]
    w6=state[22]
    w7=state[23]
    w8=state[24]
    abs_u1 = abs(a1 * w1)
    abs_u2 = abs(a2 * w2)
    abs_u3 = abs(a3 * w3)
    abs_u4 = abs(a4 * w4)
    abs_u5 = abs(a5 * w5)
    abs_u6 = abs(a6 * w6)
    abs_u7 = abs(a7 * w7)
    abs_u8 = abs(a8 * w8)
    test = ((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 + abs_u7 + abs_u8) <= 20+1e-6)
    dif = abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 + abs_u7 + abs_u8 - 20    
    return test, dif

def Check_An_O30(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    a7=action[6]
    a8=action[7]
    w1=state[25]
    w2=state[26]
    w3=state[19]
    w4=state[20]
    w5=state[21]
    w6=state[22]
    w7=state[23]
    w8=state[24]
    abs_u1 = abs(a1 * w1)
    abs_u2 = abs(a2 * w2)
    abs_u3 = abs(a3 * w3)
    abs_u4 = abs(a4 * w4)
    abs_u5 = abs(a5 * w5)
    abs_u6 = abs(a6 * w6)
    abs_u7 = abs(a7 * w7)
    abs_u8 = abs(a8 * w8)
    test = ((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 + abs_u7 + abs_u8) <= 30+1e-6)
    dif = abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 + abs_u7 + abs_u8 - 30    
    return test, dif


def Check_H_M10(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    w1=state[8]
    w2=state[9]
    w3=state[10]
    max_u1 = max(a1 * w1, 0)
    max_u2 = max(a2 * w2, 0)
    max_u3 = max(a3 * w3, 0)
    test = ((max_u1 + max_u2 + max_u3) <= 10+1e-6)
    dif = max_u1 + max_u2 + max_u3 - 10    
    return test, dif

def Check_W_M10(state, action):
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    max_u1 = max(a1 * w1, 0)
    max_u2 = max(a2 * w2, 0)
    max_u3 = max(a3 * w3, 0)
    max_u4 = max(a4 * w4, 0)
    max_u5 = max(a5 * w5, 0)
    max_u6 = max(a6 * w6, 0)
    test = ((max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6) <= 10+1e-6)
    dif = max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6 - 10    
    return test, dif

def Check_W_M5(state, action):
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    w1=state[11]
    w2=state[12]
    w3=state[13]
    w4=state[14]
    w5=state[15]
    w6=state[16]
    max_u1 = max(a1 * w1, 0)
    max_u2 = max(a2 * w2, 0)
    max_u3 = max(a3 * w3, 0)
    max_u4 = max(a4 * w4, 0)
    max_u5 = max(a5 * w5, 0)
    max_u6 = max(a6 * w6, 0)
    test = ((max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6) <= 5+1e-6)
    dif = max_u1 + max_u2 + max_u3 + max_u4 + max_u5 + max_u6 - 5    
    return test, dif

def Check_BSS3z_S(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    test = (abs(a1 + a2 + a3 - 90) <= 0.5)
    dif = abs(a1 + a2 + a3 - 90)  
    return test, dif

def Check_BSS5z_S(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    test = (abs(a1 + a2 + a3 + a4 + a5 - 150) <= 0.5+1e-6)
    dif = abs(a1 + a2 + a3 + a4 + a5 - 150)  
    return test, dif

def Check_BSS5z_S_D40(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    constraints = [
        a1 <= 1,
        a1 >= -1,
        a2 <= 1,
        a2 >= -1,
        a3 <= 1,
        a3 >= -1,
        a4 <= 1,
        a4 >= -1,
        a5 <= 1,
        a5 >= -1,
        abs(a1 + a2 + a3 + a4 + a5 - 2.5)  <= 0.1
    ]
    test = all(constraints)
    dif = abs(a1 + a2 + a3 + a4 + a5 - 2.5)  
    return test, dif

def Check_BSS3z_S_D40(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    constraints = [
        a1 <= 1,
        a1 >= -1,
        a2 <= 1,
        a2 >= -1,
        a3 <= 1,
        a3 >= -1,
        abs(a1 + a2 + a3 - 1)  <= 0.1
    ]
    test = all(constraints)
    dif = abs(a1 + a2 + a3 - 1)  
    return test, dif

def Check_BSS5z_S_D35(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    constraints = [
        a1 <= 1,
        a1 >= -1,
        a2 <= 1,
        a2 >= -1,
        a3 <= 1,
        a3 >= -1,
        a4 <= 1,
        a4 >= -1,
        a5 <= 1,
        a5 >= -1,
        abs(a1 + a2 + a3 + a4 + a5 - 3)  <= 0.1
    ]
    test = all(constraints)
    dif = abs(a1 + a2 + a3 + a4 + a5 - 3)  
    return test, dif

def Check_BSS5z_S2(state, action):
    action = np.exp(action)/np.sum(np.exp(action))
    a = (action<=35/150).all()
    b = (action>=10/150).all()
    return a and b, 0

def Check_BSS5z_S2_D40(state, action):
    action = np.exp(action)/np.sum(np.exp(action))
    a = (action<=40/150).all()
    b = (action>=0/150).all()
    return a and b, 0

def Check_BSS5z_S2_D40_ver2(state, action):
    action = (action + 1 + 1e-6)/(np.sum(action + 1 + 1e-6))
    a = (action<=40/150).all()
    b = (action>=0/150).all()
    return a and b, 0

def Check_SL_O001(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    w1=state[20]
    w2=state[21]
    w3=state[22]
    abs_u1 = abs(a1 * w1)
    abs_u2 = abs(a2 * w2)
    abs_u3 = abs(a3 * w3)

    test = ((abs_u1 + abs_u2 + abs_u3 ) <= 0.01+1e-6)
    dif = abs_u1 + abs_u2 + abs_u3 - 0.01    
    return test, dif

def Check_Pandp_O001(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    w1=state[20]
    w2=state[21]
    w3=state[22]
    abs_u1 = abs(a1 * w1)
    abs_u2 = abs(a2 * w2)
    abs_u3 = abs(a3 * w3)

    test = ((abs_u1 + abs_u2 + abs_u3 ) <= 0.01+1e-6)
    dif = abs_u1 + abs_u2 + abs_u3 - 0.01    
    return test, dif


def Check_SL_S(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    test = ((3 * a1) ** 2 + (3 * a2) ** 2 - 1) ** 3 - (3 * a1) ** 2 * (3 * a2) ** 3 < 1e-6
    dif = 0 - ((3 * a1) ** 2 + (3 * a2) ** 2 - 1) ** 3 + (3 * a1) ** 2 * (3 * a2) ** 3
    return test, dif


def Check_Pu_S(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    test = ((3 * a1) ** 2 + (3 * a2) ** 2 - 1) ** 3 - (3 * a1) ** 2 * (3 * a2) ** 3 < 1e-6
    dif = 0 - ((3 * a1) ** 2 + (3 * a2) ** 2 - 1) ** 3 + (3 * a1) ** 2 * (3 * a2) ** 3
    return test, dif

def Check_Pu_S_ellipsoid2(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    cos20 = math.cos(20*math.pi/180)
    cos40 = math.cos(40*math.pi/180)
    cos70 = math.cos(70*math.pi/180)
    sin20 = math.sin(20*math.pi/180)
    sin40 = math.sin(40*math.pi/180)
    sin70 = math.sin(70*math.pi/180)
    v1 = (cos40 * a1 + sin40 * a3)
    v2 = (a2)
    v3 = (sin40 * a2 - cos40 * a3)
    check1 = (v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
    diff1 = v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 - 1
    v1 = (a1)
    v2 = (cos20 * a2 + sin20 * a3)
    v3 = (sin20 * a2 - cos20 * a3)
    check2 = (v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
    diff2 = v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 - 1
    if check1 or check2:
        return True, min(diff1, diff2)
    else:
        return False, min(diff1, diff2)

def Check_Sl_S_ellipsoid2(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    cos20 = math.cos(20*math.pi/180)
    cos40 = math.cos(40*math.pi/180)
    cos70 = math.cos(70*math.pi/180)
    sin20 = math.sin(20*math.pi/180)
    sin40 = math.sin(40*math.pi/180)
    sin70 = math.sin(70*math.pi/180)
    v1 = (cos40 * a1 + sin40 * a3)
    v2 = (a2)
    v3 = (sin40 * a2 - cos40 * a3)
    check1 = (v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
    diff1 = v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 - 1
    v1 = (a1)
    v2 = (cos20 * a2 + sin20 * a3)
    v3 = (sin20 * a2 - cos20 * a3)
    check2 = (v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
    diff2 = v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 - 1
    if check1 or check2:
        return True, min(diff1, diff2)
    else:
        return False, min(diff1, diff2)

def Check_Pandp_S_ellipsoid2(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    cos20 = math.cos(20*math.pi/180)
    cos40 = math.cos(40*math.pi/180)
    cos70 = math.cos(70*math.pi/180)
    sin20 = math.sin(20*math.pi/180)
    sin40 = math.sin(40*math.pi/180)
    sin70 = math.sin(70*math.pi/180)
    v1 = (cos40 * a1 + sin40 * a3)
    v2 = (a2)
    v3 = (sin40 * a2 - cos40 * a3)
    check1 = (v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
    diff1 = v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 - 1
    v1 = (a1)
    v2 = (cos20 * a2 + sin20 * a3)
    v3 = (sin20 * a2 - cos20 * a3)
    check2 = (v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 <= 1)
    diff2 = v1 * v1 + 9 * v2 * v2 + 9 * v3 * v3 - 1
    if check1 or check2:
        return True, min(diff1, diff2)
    else:
        return False, min(diff1, diff2)
def Check_Point_Safe(state, action):
    a1=action[0]
    a2=action[1]
    return (a1 <= 0.5 and a2 <= 0.5), 0

def Check_Point_Safe2(state, action):
    a1=action[0]
    a2=action[1]
    constraints = [
        a1 <= 0.7,
        a2 <= 0.7,
        a1 >= 0.3,
        a2 >= 0.3
    ]
    return all(constraints), 0



def Check_Point_Safe3(state, action):
    a1 = action[0]
    a2 = action[1]
    constraints = [
        (a1 <= 0.7 and a1 >= 0.3) or (a1 >= -0.3 and a1 <= -0.7),
        (a2 <= 0.7 and a2 >= 0.3) or (a2 >= -0.3 and a2 <= -0.7)
    ]
    return all(constraints), 0

def Check_Point_Safe4(state, action):
    a1 = action[0]
    a2 = action[1]
    constraints = [
        (a1 <= 0.5 and a1 >= -0.5) ,
        (a2 <= 0.5 and a2 >= -0.5)
    ]
    return all(constraints), 0

def Check_NSFnet2(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]
    a4=action[3]
    a5=action[4]
    a6=action[5]
    a7=action[6]
    a8=action[7]
    a9=action[8]
    constraints = [
        a1 + a2 + a4 + a5 <= -3,
        a3 + a6 <= -1,
        a1 + a2 + a4 <= -2,
        a3 + a5 + a6 <= -2,
        a2 + a4 + a7 + a9 <= -3,
        a1 + a8 <= -1,
        a2 + a4 + a9 <= -2,
        a2 + a3 + a9 <= -2
    ]
    return all(constraints), 0


def Check_NSFnetV3(state, action):
    constraints = [
        action[0] + action[3] + action[6] <= 2 - len([0, 3, 6]),  # 16
        action[1] + action[2] + action[14] + action[16] <= 2 - len([1, 2, 14, 16]),  # 17
        0 <= 2 - len([]),  # 18 (no specific variables, constraint is always valid)
        action[0] + action[3] + action[15] <= 2 - len([0, 3, 15]),  # 19
        action[11] + action[19] <= 2 - len([11, 19]),  # 20
        action[0] + action[3] + action[6] + action[10] + action[15] <= 2 - len([0, 3, 6, 10, 15]),  # 21
        action[2] + action[9] + action[11] + action[14] + action[17] + action[18] <= 2 - len([2, 9, 11, 14, 17, 18]),  # 22
        action[3] + action[4] + action[15] <= 2 - len([3, 4, 15]),  # 23
        action[0] + action[5] + action[7] + action[10] <= 2 - len([0, 5, 7, 10]),  # 24
        action[4] + action[12] + action[14] + action[17] + action[18] <= 2 - len([4, 12, 14, 17, 18]),  # 25
        action[3] + action[12] + action[14] + action[15] + action[18] <= 2 - len([3, 12, 14, 15, 18]),  # 26
        action[2] + action[9] + action[11] <= 2 - len([2, 9, 11]),  # 27
        action[3] + action[8] + action[13] <= 2 - len([3, 8, 13]),  # 28
        action[6] + action[7] + action[8] + action[13] <= 2 - len([6, 7, 8, 13]),  # 29
        action[5] + action[9] + action[11] + action[13] <= 2 - len([5, 9, 11, 13])  # 30
    ]
    return all(constraints), 0



def Check_Sl_L2_1(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]

    test = ((a1 * a1 + a2 * a2 + a3 * a3) <= 1+1e-6)
    dif = (a1 * a1 + a2 * a2 + a3 * a3) - 1
    return test, dif

def Check_Pu_L2_1(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]

    test = ((a1 * a1 + a2 * a2 + a3 * a3) <= 1+1e-6)
    dif = (a1 * a1 + a2 * a2 + a3 * a3) - 1
    return test, dif

def Check_Pandp_L2_1(state, action):
    a1=action[0]
    a2=action[1]
    a3=action[2]

    test = ((a1 * a1 + a2 * a2 + a3 * a3) <= 1+1e-6)
    dif = (a1 * a1 + a2 * a2 + a3 * a3) - 1
    return test, dif

def Check_Humanoid_L2_2(state, action):

    square_sum = sum(a ** 2 for a in action)
    
    test = square_sum <= 2 + 1e-6
    
    dif = square_sum - 2
    
    return test, dif

def Check_Humanoid_M30(state, action):
    action_to_state_map = [
        29,  # Action 0 -> State 29
        28,  # Action 1 -> State 28
        30,  # Action 2 -> State 30
        31,  # Action 3 -> State 31
        32,  # Action 4 -> State 32
        33,  # Action 5 -> State 33
        34,  # Action 6 -> State 34
        35,  # Action 7 -> State 35
        36,  # Action 8 -> State 36
        37,  # Action 9 -> State 37
        38,  # Action 10 -> State 38
        39,  # Action 11 -> State 39
        40,  # Action 12 -> State 40
        41,  # Action 13 -> State 41
        42,  # Action 14 -> State 42
        43,  # Action 15 -> State 43
        44,  # Action 16 -> State 44
    ]
    
    selected_states = [state[idx] for idx in action_to_state_map]
    constraint_sum = sum(max(s_i * a_i, 0) for s_i, a_i in zip(selected_states, action))
    test = constraint_sum <= 30 + 1e-6  
    dif = constraint_sum - 30
    
    return test, dif

def Check_Humanoid_O30(state, action):
    action_to_state_map = [
        29,  # Action 0 -> State 29
        28,  # Action 1 -> State 28
        30,  # Action 2 -> State 30
        31,  # Action 3 -> State 31
        32,  # Action 4 -> State 32
        33,  # Action 5 -> State 33
        34,  # Action 6 -> State 34
        35,  # Action 7 -> State 35
        36,  # Action 8 -> State 36
        37,  # Action 9 -> State 37
        38,  # Action 10 -> State 38
        39,  # Action 11 -> State 39
        40,  # Action 12 -> State 40
        41,  # Action 13 -> State 41
        42,  # Action 14 -> State 42
        43,  # Action 15 -> State 43
        44,  # Action 16 -> State 44
    ]
    
    selected_states = [state[idx] for idx in action_to_state_map]
    constraint_sum = sum(abs(s_i * a_i) for s_i, a_i in zip(selected_states, action))
    test = constraint_sum <= 30 + 1e-6  
    dif = constraint_sum - 30
    
    return test, dif


def Check_sumo_c8(state, action):

    for i in range(0, len(action), 4):  # Iterate over groups of 4 actions
        action_group = action[i:i+4]
        group_sum = sum(action_group)
        if not (-3.2 <= group_sum <= 2.4):  # Check the sum constraint
            return False, 0

    return True, 0
