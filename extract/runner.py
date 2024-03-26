run = True
if run:
    #put python script here
    ### hilda optimizer

    import numpy as np
    import pandas as pd

    jit = 150
    jdat = 150
    obf = np.zeros(jit + 1)  # Adding 1 since Fortran arrays are 1-indexed
    abf = np.zeros(jit + 1)
    ocfit = np.zeros(jit + 1)

    atmoco2_SSP1 = './data/csvs/atmco2-SSP1.csv'
    file_path = atmoco2_SSP1
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame to verify it's loaded correctly
    time = df.year.to_numpy()
    atm = df.atmco2.to_numpy()
    pulse0 = np.zeros_like(atm)

    # print(atm[0])
    for i in range(1,len(pulse0)):
        pulse0[i] = (atm[i] - atm[i-1]) * 2.123 #gcb, table1


    oceansink = './data/csvs/Cflx-Pgyr-ozoneFIXED-SSP1_SO.csv'
    file_path = oceansink
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    rocdat = df.Cflx.to_numpy()

    def get_ocfit(rocdat,ra0,ra1,ra2,ra3,rt1,rt2,rt3):

        for it in range(1, jit + 1):  # Adjusting for Python's 0-indexing
            rit = float(it)
            abf[it] = ra0 + ra1 * np.exp(-rit / rt1) + ra2 * np.exp(-rit / rt2) + ra3 * np.exp(-rit / rt3)

            if it > 1:
                obf[it] = abf[it - 1] - abf[it]
            else:
                obf[it] = 1. - abf[it]

            ocfit[it] = 0
            for jt in range(1, it + 1):
                ocfit[it] += pulse0[jt - 1] * obf[it - jt + 1] 

        # Calculate RMSE
        rsum = np.sum((rocdat[:jdat] - ocfit[:jdat]) ** 2)
        #print(rsum)

        return ocfit, rsum


    ra0 = 0.7; ra0mi = 0.35; ra0ra = 0.7
    ra1 = 0; ra1mi = 0; ra1ra = 2
    ra2 = 0.1; ra2mi = 0; ra2ra = (ra2-ra2mi) *2
    ra3 = 0.2; ra3mi = 0; ra3ra = (ra3-ra3mi) *2
    rt1 = 1; rt1mi = 0.1; rt1rt = (rt1-rt1mi) *2
    rt2 = 38; rt2mi = 20; rt2rt = (rt2-rt2mi) *2
    rt3 = 200; rt3mi = 100; rt3rt = (rt3-rt3mi) *2

    jpmax = 8

    ra0_i = np.zeros(jpmax**7)
    ra1_i = np.zeros(jpmax**7)
    ra2_i = np.zeros(jpmax**7)
    ra3_i = np.zeros(jpmax**7)
    rt1_i = np.zeros(jpmax**7)
    rt2_i = np.zeros(jpmax**7)
    rt3_i = np.zeros(jpmax**7)
    rsum_i = np.zeros(jpmax**7)

    ind = 0
    for ra0i in range(0,jpmax):
        ra0mi = 0.4; ra0ra = 0.8 - 0.4
        ra0 = ra0mi + (ra0ra/(jpmax-1))*ra0i

        for ra1i in range(0,jpmax):
            ra1mi = 0; ra1ra = 2
            ra1 = ra1mi + (ra1ra/(jpmax-1))*ra1i

            for ra2i in range(0,jpmax):
                ra2mi = 0; ra2ra = 0.14
                ra2 = ra2mi + (ra2ra/(jpmax-1))*ra2i

                for ra3i in range(0,jpmax):
                    ra3mi = 0.2; ra3ra = 0.3
                    ra3 = ra3mi + (ra3ra/(jpmax-1))*ra3i

                    for rt1i in range(0,jpmax):
                        rt1mi = 0.01; rt1ra = 0.49
                        rt1 = rt1mi + (rt1rt/(jpmax-1))*rt1i

                        for rt2i in range(0,jpmax):
                            rt2mi = 0.01; rt2ra = 0.49
                            rt2 = rt2mi + (rt2rt/(jpmax-1))*rt2i

                            for rt3i in range(0,jpmax):
                                rt3mi = 100; rt3ra = 300
                                rt3 = rt3mi + (rt3rt/(jpmax-1))*rt3i


                                ra0_i[ind] = ra0
                                ra1_i[ind] = ra1 
                                ra2_i[ind] = ra2 
                                ra3_i[ind] = ra3 
                                rt1_i[ind] = rt1 
                                rt2_i[ind] = rt2 
                                rt3_i[ind] = rt3 
                                ocfit, rsum = get_ocfit(rocdat,ra0,ra1,ra2,ra3,rt1,rt2,rt3)
                                rsum_i[ind] = rsum
                                ind = ind+1
                                if ind%1000 == 0:
                                    print(ind)


    df = pd.DataFrame([ra0_i,ra1_i,ra2_i,ra3_i,rt1_i,rt2_i,rt3_i,rsum_i]).T
    df.columns = ['ra0','ra1','ra2','ra3','rt1','rt2','rt3','rsum']
    df.wheremade = '/scratch/BOE-SOcarbon/hilda-optimisation.ipynb'
    df.to_csv('./data/hildaoptimization_thirdpass.csv')
