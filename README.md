# Real_Robot_Challenge_Phase2_AE_attemp
I am sorry, the project is too complex with too much bigger files, It is too hard to upload them all on Github. I will only state the reults I got and raise some problems. If you think my attempts is approriate, you can go to this google drive to download the project file.\

Our attempt: Segmentation map was used as observation for feeding the RL agent. Autoencoder was used to reduce the dimensionality of the data to extract the main component. The dimension of data was reduced from the original 270*270*3 to 384.
The reconstructed image from Autoencoder was decent. However, when we fed the RL agent with a latent vector with the size of 384, it cannot make the agent learn effectively. We believe that the composition of the vector of the latent space of the autoencoder changes with the change of the input images. Although the reconstructed picture looks good, the vector of the latent space makes no sense, RL agent can hardly extract information from it
It is recommended that people who have the same idea can use VAE to try it.
