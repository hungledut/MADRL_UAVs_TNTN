import gymnasium as gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class UAV_Environment(gym.Env):

    def __init__(self, users = 250, uavs = 3,
                size = 2000, # Target area: size x size
                v_0 = 30, # (m/s)
                tau = 1, # (s)
                UAV_coverage = 500, # (m)
                mBS_coverage = 3000, # (m)
                ###################### Channel model of UAVS
                psi_L = 1,
                psi_M = 1,
                K = 50,
                d = 1, # (m)
                lambda_c = 0.05, # (m) = v/f = 3x10^8 (m/s)/ 5.8x10^9 (Hz) ?
                h = 500, # (m)
                alpha = 2.7,
                P_UAV = 2.5, # (W)
                sigma_square = 1/(1e12),  # -90 (dBm) -> W
                ###################### Channel model of mBS
                P_mBS = 54, # 76 (dBm) -> dB
                D_hb = 100, # (m) ?
                f_c = 2, # GHz ?
                sigma_logF = 2, # mean of logF ?
                ###################### Bandwidth
                W = 20e6, # Hz
                ###################### Data Rate Threshold
                r_th = 20e6, # bps
                ###################### Heatmap
                grid_num = 10,
                max_step = 50
                ):


        self.users = users
        self.uavs = uavs
        self.size = size
        self.v_0 = v_0
        self.tau = tau
        self.UAV_coverage = UAV_coverage
        self.mBS_coverage = mBS_coverage

        self.psi_L = psi_L
        self.psi_M = psi_M
        self.K = K
        self.d = d
        self.lambda_c = lambda_c
        self.h = h
        self.alpha = alpha
        self.P_UAV = P_UAV
        self.sigma_square = sigma_square
        self.P_mBS = P_mBS
        self.D_hb = D_hb
        self.f_c = f_c
        self.sigma_logF = sigma_logF
        self.W = W
        self.r_th = r_th
        self.grid_num = grid_num
        self.max_step = max_step
        self.grid_size = self.size/self.grid_num

        self.connect = np.zeros((self.uavs+1, self.users))
        self.unsatisfied_users = np.zeros(self.users)
        self.data_rate_max_index = np.zeros(self.users) - 1000

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self.uavs_location = np.zeros((2, self.uavs)) + self.size/2
        self.users_location = np.random.uniform( -self.size/2, self.size/2, (2, self.users))
        self.mBS = np.zeros((2,1)) + self.size/2
        # UAVs behaviour
        self.UAV0_behavior = np.zeros((2, self.max_step)) + self.size/2
        self.UAV1_behavior = np.zeros((2, self.max_step)) + self.size/2
        self.UAV2_behavior = np.zeros((2, self.max_step)) + self.size/2
        # Heatmap
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_satisfied = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        # Step 
        self.step_ = 0

        # Each location is encoded as an element of {0, ..., `size`-1}^2
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(-size/2, size/2, shape=(2,uavs), dtype=float),
        #     }
        # )

        self.observation_space = gym.spaces.Box(-self.size/2, self.size/2, shape=(2,uavs), dtype=float),


        # We have 5 actions, corresponding to "right", "up", "left", "down", "remain stationary"
        self.action_space = gym.spaces.Discrete(5)
        # Dictionary maps the abstract actions to the directions
        self._action_to_direction = {
            0: np.array([0, 0]),  # remain stationary
            1: np.array([0,self.v_0*self.tau]),  # up
            2: np.array([-self.v_0*self.tau, 0]),  # left
            3: np.array([0, -self.v_0*self.tau]),  # down
            4: np.array([self.v_0*self.tau, 0]),  # right
        }
        self.user_action_to_direction = {
            0: np.array([0, 0]),  # remain stationary
            1: np.array([0, 5]),  # up
            2: np.array([5, 0]),  # left
            3: np.array([0, -5]),  # down
            4: np.array([-5, 0]),  # right
        }
    def step(self, actions):

        ########################### Reset each step ##################################################
        self.connect = np.zeros((self.uavs+1, self.users))
        self.unsatisfied_users = np.zeros(self.users)
        self.data_rate_max_index = np.zeros(self.users) - 1000
        # reset heatmap each step
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_satisfied = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        ##############################################################################################

        ################################# UAVs take the actions ################################################
        for (action,k) in zip(actions,range(self.uavs)):
            tem = self.uavs_location[:,k] + self._action_to_direction[action]
            if (-self.size/2 > tem[0] or tem[0] > self.size/2) or (-self.size/2 > tem[1] or tem[1] > self.size/2): # restrict the UAVs' coordinates
                continue
            self.uavs_location[:,k] = tem

        ################################# users take the actions ################################################
        user_actions = [random.randint(0, 4) for _ in range(250)]
        for (action,k) in zip(user_actions,range(self.users)):
            tem = self.users_location[:,k] + self.user_action_to_direction[action]
            if (-self.size/2 > tem[0] or tem[0] > self.size/2) or (-self.size/2 > tem[1] or tem[1] > self.size/2): # restrict the users' coordinates
                continue
            self.users_location[:,k] = tem
            
        ########################################################################################################

        ## Distance
        d_UAV = np.zeros((self.uavs,self.users))
        for n in range(self.users):
            for k in range(self.uavs):
                d_UAV[k,n] = np.linalg.norm(self.users_location[:,n] - self.uavs_location[:,k], ord = 2)

        d_mBS = np.zeros((1,self.users))
        for n in range(self.users):
            d_mBS[0,n] = np.linalg.norm(self.users_location[:,n] - self.mBS[:,0], ord = 2)

        # connect_tem =
        # [ 0 1 0 0 1 ... 0 1]
        # [ 0 0 0 1 1 ... 0 1]
        # [ 0 0 0 1 0 ... 0 0]
        # [ 0 1 0 0 1 ... 0 1]
        connect_tem = np.zeros((self.uavs+1,self.users))
        for n in range(self.users):
            for k in range(self.uavs):
                if d_UAV[k,n] <= self.UAV_coverage:   # if < 400 (m) -> connect , may be connected to multiple UAVs
                    connect_tem[k,n] = 1

        for n in range(self.users):
            if d_mBS[0,n] <= self.mBS_coverage:    #mBS
              connect_tem[3,n] = 1


        ## Signal-to-noise ratio(SNR)
        gamma_UAV = np.zeros((self.uavs,self.users))
        for n in range(self.users):
            for k in range(self.uavs):
                psi_UAV = math.sqrt(self.K/(1+self.K))*self.psi_L + math.sqrt(1/(1+self.K))*self.psi_M
                theta = -20*math.log10(4*3.14*self.d/self.lambda_c) # (dB)
                theta = 10**(theta/10)
                g_UAV = (abs(psi_UAV)**2)*theta*(math.sqrt((d_UAV[k,n])**2 + (self.h)**2)/self.d)**(-self.alpha)
                gamma_UAV[k,n] = self.P_UAV*g_UAV/self.sigma_square

        gamma_mBS = np.zeros((1,self.users))
        for n in range(self.users):
            # LogF = random.gauss(0, self.sigma_logF)
            L_mBS = 40*(1-0.004*self.D_hb)*math.log10(d_mBS[0,n]/1000) - 18*math.log10(self.D_hb) + 21*math.log10(self.f_c) + 80 # d_mBS (km) ; D_hb (m); fc (MHz)
            gamma_mBS[0,n] = self.P_mBS - L_mBS #- LogF  # dB

        ## Data rate
        rate_UAV = np.zeros((self.uavs,self.users))
        for n in range(self.users):
            for k in range(self.uavs):
                rate_UAV[k,n] = self.W*math.log2(1 + gamma_UAV[k,n])

        rate_mBS = np.zeros((1,self.users))
        for n in range(self.users):
            gamma_mBS[0,n] = 10**(gamma_mBS[0,n]/10)
            rate_mBS[0,n] = self.W*math.log2(1 + gamma_mBS[0,n])

        ######################################### Connection #######################################################

        ##### self.connect =
        #####    [ ------------ UAV1 -----------]
        #####    [ ------------ UAV2 -----------]
        #####    [ ------------ UAV3 -----------]
        #####    [ ------------ mBS  -----------]

        # print('UAV data rate: ', rate_UAV,'\n mBS data rate: ' ,rate_mBS)
        rate_UAV_mBS = np.concatenate((rate_UAV, rate_mBS), axis=0) # Shape(4,250) -> 3 UAVs + 1 mBS ; 250 Users


        # rate_UAV_mBS*connect_tem =
        # [ 0 11 0 0 31 ... 0 91]
        # [ 0 0 0 13 61 ... 0 11]
        # [ 0 0 0 17 0 ...  0  0]
        # [ 0 12 0 0 13 ... 0 11]
        rate_UAV_mBS = np.multiply(rate_UAV_mBS, connect_tem)

        # data_rate_max_index - [0 2 1 3 ... 1 1 0]
        self.data_rate_max_index = np.argmax(rate_UAV_mBS, axis=0) # UAV0 = 0, UAV1 = 1, UAV2 = 2, mBS = 3
        for (index,user) in zip(self.data_rate_max_index,range(self.users)):
            # print('print = ', index, user)
            # if not connect any UAVs/mBS -> max index = -1000
            if connect_tem[0,user] == connect_tem[1,user] == connect_tem[2,user] == connect_tem[3,user] == 0:
                self.data_rate_max_index[user] = -1000
                continue

            self.connect[index,user] = 1

        ###################################################################################################################

        data_rate = np.multiply(rate_UAV_mBS,self.connect)
        data_rate = np.sum(data_rate,axis=0)

        for i in range(data_rate.shape[0]):
            if data_rate[i] < self.r_th:
                self.unsatisfied_users[i] = 0 # -> unsatisfied user
            else:
                self.unsatisfied_users[i] = 1 # -> satisfied user

        # S (sum of satisfied users)
        S = np.sum(self.unsatisfied_users)


        ################################################ The numbers of users connected to each UAV ##############
        N_UAVs = np.sum(self.connect, axis=1)
        N_UAV0 = N_UAVs[0]
        # print('N_UAV0=',N_UAV0)
        N_UAV1 = N_UAVs[1]
        # print('N_UAV1=',N_UAV1)
        N_UAV2 = N_UAVs[2]
        # print('N_UAV2=',N_UAV2)
        ##########################################################################################################

        ################################################# Heatmap  ################################################

        UAV0_users_location = []
        for i in range(self.users):
            if self.connect[0,i] == 1:
                UAV0_users_location.append([self.users_location[0,i],self.users_location[1,i]])
        UAV1_users_location = []
        for i in range(self.users):
            if self.connect[1,i] == 1:
                UAV1_users_location.append([self.users_location[0,i],self.users_location[1,i]])
        UAV2_users_location = []
        for i in range(self.users):
            if self.connect[2,i] == 1:
                UAV2_users_location.append([self.users_location[0,i],self.users_location[1,i]])
        # heatmap of unsatisfied users
        satisfied_users = []
        for i in range(self.users):
            if self.unsatisfied_users[i] == 0:
                satisfied_users.append([self.users_location[0,i],self.users_location[1,i]])
        # heatmap of all users in the target area
        users_list = []
        for i in range(self.users):
            users_list.append([self.users_location[0,i],self.users_location[1,i]])

        if len(UAV0_users_location) > 0:
            for i in range(len(UAV0_users_location)):
                x = int((UAV0_users_location[i][0]+self.size/2)//self.grid_size)
                y = int((UAV0_users_location[i][1]+self.size/2)//self.grid_size)
                self.heatmap_UAV0[x,y] += 1
        if len(UAV1_users_location) > 0:
            for i in range(len(UAV1_users_location)):
                x = int((UAV1_users_location[i][0]+self.size/2)//self.grid_size)
                y = int((UAV1_users_location[i][1]+self.size/2)//self.grid_size)
                self.heatmap_UAV1[x,y] += 1
        if len(UAV2_users_location) > 0:
            for i in range(len(UAV2_users_location)):
                x = int((UAV2_users_location[i][0]+self.size/2)//self.grid_size)
                y = int((UAV2_users_location[i][1]+self.size/2)//self.grid_size)
                self.heatmap_UAV2[x,y] += 1
        # heatmap of unsatisfied users
        if len(satisfied_users) > 0:
            for i in range(len(satisfied_users)):
                x = int((satisfied_users[i][0]+self.size/2)//self.grid_size)
                y = int((satisfied_users[i][1]+self.size/2)//self.grid_size)
                self.heatmap_satisfied[x,y] += 1
        # heatmap of all users in the target area
        if len(users_list) > 0:
            for i in range(len(users_list)):
                x = int((users_list[i][0]+self.size/2)//self.grid_size)
                y = int((users_list[i][1]+self.size/2)//self.grid_size)
                self.heatmap_users[x,y] += 1

        ###############################################################################################################

        ####################################################### Observation ###########################################
        O_UAV0 = np.concatenate((self.uavs_location[:,0],self.uavs_location[:,1],self.uavs_location[:,2],np.reshape(self.heatmap_UAV0,self.grid_num**2),np.reshape(self.heatmap_satisfied,self.grid_num**2)))
        # print('O_UAV0=',O_UAV0.shape)
        O_UAV1 = np.concatenate((self.uavs_location[:,1],self.uavs_location[:,0],self.uavs_location[:,2],np.reshape(self.heatmap_UAV1,self.grid_num**2),np.reshape(self.heatmap_satisfied,self.grid_num**2)))
        # print('O_UAV1=',O_UAV1.shape)
        O_UAV2 = np.concatenate((self.uavs_location[:,2],self.uavs_location[:,0],self.uavs_location[:,1],np.reshape(self.heatmap_UAV2,self.grid_num**2),np.reshape(self.heatmap_satisfied,self.grid_num**2)))
        # print('O_UAV2=',O_UAV2.shape)

        ###############################################################################################################

        ####################################### UAV behavior ##########################################################
        self.UAV0_behavior[:,self.step_] = self.uavs_location[:,0]
        self.UAV1_behavior[:,self.step_] = self.uavs_location[:,1]
        self.UAV2_behavior[:,self.step_] = self.uavs_location[:,2]
        ###############################################################################################################

        ########################################################################################################
        self.step_ += 1

        return O_UAV0, O_UAV1, O_UAV2, S, N_UAV0, N_UAV1, N_UAV2

    def plot(self):
        plt.figure()
        # plt.scatter(self.users_location[0,:],self.users_location[1,:], marker = 'o', color = 'c')
        plt.scatter(self.uavs_location[0,0],self.uavs_location[1,0], label = 'UAV0', marker = ',', color = 'r')
        plt.scatter(self.uavs_location[0,1],self.uavs_location[1,1], label = 'UAV1', marker = ',', color = 'g')
        plt.scatter(self.uavs_location[0,2],self.uavs_location[1,2], label = 'UAV2', marker = ',', color = 'b')
        plt.scatter(self.mBS[0],self.mBS[1], label = 'mBS', marker = ',', color = 'm')
        # print(self.data_rate_max_index)
        # print(self.unsatisfied_users)
        for i in range(self.unsatisfied_users.shape[0]):
            # UAV0
            if self.data_rate_max_index[i] == 0 and self.unsatisfied_users[i] == 0:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = '^', color = 'r')
            elif self.data_rate_max_index[i] == 0 and self.unsatisfied_users[i] == 1:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = 'o', color = 'r')

            # UAV1
            if self.data_rate_max_index[i] == 1 and self.unsatisfied_users[i] == 0:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = '^', color = 'g')
            elif self.data_rate_max_index[i] == 1 and self.unsatisfied_users[i] == 1:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = 'o', color = 'g')

            # UAV2
            if self.data_rate_max_index[i] == 2 and self.unsatisfied_users[i] == 0:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = '^', color = 'b')
            elif self.data_rate_max_index[i] == 2 and self.unsatisfied_users[i] == 1:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = 'o', color = 'b')

            # mBS
            if self.data_rate_max_index[i] == 3 and self.unsatisfied_users[i] == 0:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = '^', color = 'm')
            elif self.data_rate_max_index[i] == 3 and self.unsatisfied_users[i] == 1:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = 'o', color = 'm')

            # not connected
            if self.data_rate_max_index[i] == -1000:
                plt.scatter(self.users_location[0,i],self.users_location[1,i], marker = 'x', color = 'gray')

        plt.scatter(self.UAV0_behavior[0,:],self.UAV0_behavior[1,:], color = 'r', s=1)
        plt.scatter(self.UAV1_behavior[0,:],self.UAV1_behavior[1,:], color = 'g', s=1)
        plt.scatter(self.UAV2_behavior[0,:],self.UAV2_behavior[1,:], color = 'b', s=1)

        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.title('UAV Environment')
        plt.legend()
        plt.savefig("result.png")
        # plt.show()
        plt.close()

        # print('UAV0 heatmap = \n',self.heatmap_UAV0)
        # print('UAV1 heatmap = \n',self.heatmap_UAV1)
        # print('UAV2 heatmap = \n',self.heatmap_UAV2)
        # print('satisfied-users heatmap = \n',self.heatmap_satisfied)
        # print('all-users heatmap = \n',self.heatmap_users)

    def reset(self):
        self.uavs_location = np.zeros((2, self.uavs)) + self.size/2
        self.connect = np.zeros((self.uavs+1, self.users))
        self.unsatisfied_users = np.zeros(self.users)
        self.users_location = np.random.uniform(-self.size/2, self.size/2, (2, self.users))

        ####################### UAVs behaviour ########################################
        self.UAV0_behavior = np.zeros((2, self.max_step)) + self.size/2
        self.UAV1_behavior = np.zeros((2, self.max_step)) + self.size/2
        self.UAV2_behavior = np.zeros((2, self.max_step)) + self.size/2

        ####################### Heatmap ########################################
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_satisfied = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))

        self.step_ = 0

        O_UAV0 = np.concatenate((self.uavs_location[:,0],self.uavs_location[:,1],self.uavs_location[:,2],np.reshape(self.heatmap_UAV0,self.grid_num**2),np.reshape(self.heatmap_satisfied,self.grid_num**2)))
        # print('O_UAV0=',O_UAV0.shape)
        O_UAV1 = np.concatenate((self.uavs_location[:,1],self.uavs_location[:,0],self.uavs_location[:,2],np.reshape(self.heatmap_UAV1,self.grid_num**2),np.reshape(self.heatmap_satisfied,self.grid_num**2)))
        # print('O_UAV1=',O_UAV1.shape)
        O_UAV2 = np.concatenate((self.uavs_location[:,2],self.uavs_location[:,0],self.uavs_location[:,1],np.reshape(self.heatmap_UAV2,self.grid_num**2),np.reshape(self.heatmap_satisfied,self.grid_num**2)))
        # print('O_UAV2=',O_UAV2.shape)
        return O_UAV0, O_UAV1, O_UAV2
