# Real_Robot_Challenge_Phase2_AE_attemp
We(team name: thriftysnipe) are the first place winner of Phase1 in 2021 Real Robot Challenge. \
Please see this page for more details: https://real-robot-challenge.com/leaderboard \
To see more details about out Phase1 works: https://github.com/wq13552463699/Real_Robot_challenge \
We were granted the access to Phase 2.

I am sorry, the project is too complex with too much large files, It is too hard to upload them all on Github. I just attached a part of the core code here for you to take a quick lreview. If you think my attempts is approriate, you can go to this Google Drive to download the full project file(all codes, results, trained models, environmental files,.etc):\
https://drive.google.com/file/d/14vjCrWU6vzMdXxVSR2FeskMvuQpgqWqM/view?usp=sharing 

## RRC phase2 task description:
Randomly place 25 dices with the size of 0.01x0.01x0.01m in the environment. Use own controller to drive the three-finger robot to rearrange the dice to a specific pattern. 
Unfortunately, due to the set task is too difficult, no team could complete the task on the actual robot, so all teams with record are awarded third place in this phase. But I think our attempt has a reference value, if later scholars conduct related research, our method may be useful.
<img src="https://github.com/wq13552463699/Real_Robot_Challenge_Phase2_AE_attemp/blob/main/pictures/2.jfif" width="633" >

## Our considerations：
We consider using a reinforcement learning algorithm as the controller in this phase. However, in this phase, information that can play as observations, such as coordinates and orientation of the dices, cannot be obtained from the environment directly but they are crucial for RL to run. \
The alternative observations we can use are the images of the three cameras set in 3 different angles in the environment and their segmentation masks. We picked segmentation masks rather than the raw images since the attendance of noise and redundancy in the raw images were too much. Please see the following segmentation mask example(RGB's 3 channels represent segmentation masks from 3 different angles).\
<img src="https://github.com/wq13552463699/Real_Robot_Challenge_Phase2_AE_attemp/blob/main/pictures/43028.jpg" width="300" >\
The segmentation masks have the dimension of 270x270x3, if directly passing it to the RL agent, which would lead to computational explosion and hard to converge. Hence, we planned to use some means to extract the principal components that can play as observations from it. In addition, the observation value also includes readable read-robot data(joint angle of the robot arm, end effector position, end effector speed, etc.).
			
## Segmentation mask dimensionality reduction
This is the most important part of this task. We tried different methods, such as GAN, VAE, AE, to extract the principal conponents from the images. The quality of data dimensionality reduction can be easily seem from the discripency of reconstructed and oringinal images or the loss curves. After many trials(adjusting hyperparameters, network structure, depth, etc.), we got different trained VAE, GAN and AE models. We conducted offline tests on the obtained model and compared the results, we were surprised to find that the AE performed the best, because AE had the most simple structure among GAN, VAE, AE. When the latent of AE is 384, the quality of the reconstructed image is the best. The result is shown in the figure below.\
<img src="https://github.com/wq13552463699/Real_Robot_Challenge_Phase2_AE_attemp/blob/main/pictures/1.png" width="450" >

The loss function also converges to an acceptable range:\
<img src="https://github.com/wq13552463699/Real_Robot_Challenge_Phase2_AE_attemp/blob/main/pictures/3.png" width="1000" >

## Build up observation and trian RL agent.
We use the best AE encoder to deal with the segmentation masks to generate the observation and stitch with the readable data. The structure of the overall obervation is shown as follow:\
<img src="https://github.com/wq13552463699/Real_Robot_Challenge_Phase2_AE_attemp/blob/main/pictures/4.png" width="1000" >
We fed the above observations to several current cutting-edge model based and model free reinforcement learning algorithms, such as **DDPG+HER, PPO, SLAC, PlaNet and Dreamer**. We thought it would work and enable the agent to learn for somewhat anyway. But it is a pity that after many attempts, the model still didn't have any trend to converge.
Due to time limited, our attempts were over here.

## Some reasons might lead to fail
1. We used AE as the observation model. Although the AE's dimensionality reduction capability were the best, the latent space of AE were disordered and didn't make sense to RL agent. The observations passed to the RL must be fixed and orderly. Continuous delivery of unfixed data has caused a dimensional disaster. For example, the third number in the observation vector passed at time 1 represents 'aaaaa', and the number on the same position at time 2 represents the 'bbbbb'. This disorderly change makes RL very confused.
2. The extracted latent space from segmentation mask dominates the observations, making RL ignore the existence of robots. The latent space size is 384, but which for the robot data is 27. The two are far apart, and there is a big data bias. 
3. Robot arm blocked the dices, segmentation masks can only represent a part of the dice. This problem cannot be avoided and can only be solved by more powerful image processing technology. This is also a major challenge in the current Image-based RL industry
			
## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change
Please make sure to update tests as appropriate.
