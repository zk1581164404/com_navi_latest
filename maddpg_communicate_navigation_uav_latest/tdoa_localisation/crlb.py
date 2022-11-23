import numpy as np
import example

def getCRLB2():
    c = 340.29
    # print (c)
    nk0_1 = example.getTimediffK0_1()
    nk0_2 = example.getTimediffK0_2()
    ek = [nk0_1*c,nk0_2*c]
    ek_T = np.mat(ek).T
    Rk = np.cov(ek_T*ek)  
    # print("Rk = ",Rk)
    list_locate = example.getlocate()
    print("list_locate:",list_locate)
    xu = list_locate['x']
    yu = list_locate['y']

    # print("xu= ",xu,"yu= ",yu)

    xk0 = 18
    xk1 = 19
    xk2 = 13

    yk0 = 40
    yk1 = 70
    yk2 = 16

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
    print(temp)
    J = 1/np.mat(temp).trace()
    print("J =",J[0,0])

getCRLB2()
