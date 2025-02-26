## Multi-agent Deep Reinforcement Learning for UAVs in TNTN Network🎮
### Install fundamental libraries for customed UAV environment☀️
Install GYM to custome environment and mathplotlib to visualize the environment
~~~
pip install gymnasium 
~~~
~~~
pip install matplotlib
~~~
### Train model🤖
~~~
py train_ppo.py 
~~~
### Test model🌗
~~~
py test_ppo.py 
~~~
### Experimental Results of MADRL PPO⚡️
Apply Multi-agent DRL with Proximal Policy Optimization (PPO)
<br> **The total of rewards on each episode** <br>
![fig1](images/result_reward.png)
<br> **The percentage of satisfied users on each episode** <br>
![fig2](images/result_user.png)
<br> **Test case 1** <br>
![fig3](images/TC1_behavior.png)
![fig4](images/TC1_step.png)
<br> **Test case 2** <br>
![fig3](images/TC2_behavior.png)
![fig4](images/TC2_step.png)

