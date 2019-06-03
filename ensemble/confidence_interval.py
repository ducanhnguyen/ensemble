import numpy as np

# The data was drawn from a distribution with unknown mean and unknown standard deviation
# we need to estimate the range of true mean (confidence interval of 95% for true mean)
# method: use bootstrap

data = np.array([61, 88, 89, 89, 90, 92, 93, 94, 98, 98, 101, 102, 105, 108, 109, 113, 114, 115, 120, 138])

# STEP 1: generate samples
N = data.shape[0] # number of observations in a sample
B = 100000 # number of samples

samples = np.zeros(shape=(B, N))
for i in range(B):
    # sampling with replacement
    samples[i] = np.random.choice(data, N, replace=True)

# STEP 2: calculate mean and standard deviation of sample mean
samples_mean = samples.mean(axis=1)
mean = np.mean(samples_mean)
std = np.sqrt(np.var(samples_mean))

# for CI of 95%, z = 1.96
Z = 1.96
lower = mean - Z * std
upper = mean + Z * std

print("CI of 95% for true mean: [" + str(lower) + " ," + str(upper) + "]")
