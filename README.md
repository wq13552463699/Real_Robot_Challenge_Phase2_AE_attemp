# Real_Robot_Challenge_Phase2_AE_attemp
We(team name: thriftysnipe) are the first place winner of Phase1 in 2021 Real Robot Challenge. \
Please see this page for more details: https://real-robot-challenge.com/leaderboard\
We were granted the access to Phase 2.


I am sorry, the project is too complex with too much bigger files, It is too hard to upload them all on Github. I will only state the reults I got and raise some problems. If you think my attempts is approriate, you can go to this google drive to download the project file.\

Our attempt: Segmentation map was used as observation for feeding the RL agent. Autoencoder was used to reduce the dimensionality of the data to extract the main component. The dimension of data was reduced from the original 270*270*3 to 384.
The reconstructed image from Autoencoder was decent. However, when we fed the RL agent with a latent vector with the size of 384, it cannot make the agent learn effectively. We believe that the composition of the vector of the latent space of the autoencoder changes with the change of the input images. Although the reconstructed picture looks good, the vector of the latent space makes no sense, RL agent can hardly extract information from it
It is recommended that people who have the same idea can use VAE to try it.

## RRC phase2 task description:：
			在环境中随机放置25个大小为0.01*0.01*0.01m的色子，需要自己设计算法使三指机器人将色子rearrange到特定的pattern。
			很遗憾，由于所设置的任务过于难，没有队伍在实际的机器人应用中完成任务。但是我们的attempt有一定的参考价值，后面的学者
			进行相关方面的研究的话我们的数据集和自编码器可以用来借鉴。

## Our attemp：
			我们的考虑：
			我们在这个阶段考虑使用强化学习算法作为控制器。然而在这个阶段dices的位置，方向等信息无法通过编程的方法（直接从模拟器中读取）得到，
			这对强化学习算法又是必须的。我们可以使用的作为替代的观察值就是环境中三个摄像头的图像和segmentation mask。我们没有考虑使用图像作为输入，因为图像中
			包含的背景噪声过多，我们考虑的是使用segmentation mask作为输入，并通过一定的方法对从segmentation mask中提取dices在环境中的信息。然后将这些信息作为观察值
			传递给强化学习算法。此外，观察值中还包括可以直接读取的信息-机器人的数据（机械手臂的关节角度，末端执行器位置，末端执行器速度等。）。
			
			segmentation mask数据降维
			我们尝试了如GAN,VAE,AE等多种不同的图像数据降维方法，以reconstructed的图片与原始图片的discripency作为评判标准，判断数据降维的好坏。在经过了多轮尝试对比
			后，我们惊奇的发现AE的表现是最好的。结果如下图所示。
			损失函数也收敛到了一个可接受的范围内。
			
			GAN,VAE,AE等生成式的神经网络的浅层被期盼着是原始图片的有效表达方式。我们使用浅层的向量作为观察值反应环境中色子的分布情况，并添加进机器人数据，如下如所示的数据结构，并
			将该数据作为观察值传递给强化学习算法。
			
			强化学习算法选择：我们分别尝试了几种目前最前沿的Model based和model free的强化学习算法。分别是DDPG+HER，PPO, PlaNet和Dreamer。理论上来讲，尽管观察值在一定程度上存在误差，但是误差值
			在可接受的范围之内，无论如何这种观察值都会对强化学习算法产生贡献，使算法能够学习。但是经过了几周不断对观察模型和强化学习模型的调整后，依旧不能够解决这个问题，任何的强化学习算法根本就没有
			一个收敛的趋势。
			
			
Appendix:
			我们考虑我们失败的原因：1. 我们使用的是AE作为观察模型.AE的浅空间并不像VAE那样有实际的意义，AE的潜空间是无序且不固定的。而传递给强化学习算法的观察值必须是固定且有序的。连续传递
			不固定的数据造成了维度灾难。举个例子，在时间1处所传递的矢量中第三个数字代表的是意义a，而在时间2处所传递的矢量中第三个数字代表意义b，这种无序的变化使RL很困惑。2.潜空间大小在观察值中
			占主导地位，RL很难get到机器人的实时情况。经过调试，我们发现潜空间大小为384时候所产生的的reconstructed图片最清楚，但是robot data的维度是27.二者之间相距甚远，有很大的数据偏向。3.机械手臂遮挡了
			segmentation map，segmentation map只能反映出一部分色子的情况，而一部分被机械手臂遮挡了。
			
Contribution
