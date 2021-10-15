import numpy as np

mcmillan = np.genfromtxt('ProductionRunBig2_Nora_10.tab')

for i in range(len(mcmillan)):
    mcm = mcmillan[i]
    # mcm = mcmillan[0]

    Usun = mcm[32] * 1000.
    Vsun = mcm[33] * 1000.
    Wsun = mcm[34] * 1000.
    R0 = mcm[-6]
    V0 = mcm[-5]

    M200 = 4. * np.pi * mcm[26] * mcm[30]**3. * (np.log(1. + mcm[-9] / mcm[30]) - mcm[-9] / (mcm[-9] + mcm[30])) / (1.e10)

    c200 = mcm[-9] / mcm[30]
    rs = mcm[30]

    M_NFW = M200 
    rs_NFW = rs 
    c_NFW = c200

    print(i, M_NFW)

