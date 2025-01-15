rm(list=ls()) # to clear the environment

#### loading some packages and functions ####
library("plot3D")
library("MASS")
source("kernFun.R")

#### Example with the Exp. kernel  ####
x <- seq(0, 1, 0.01) # regular grid
param <- c(1, 0.2) # covariance parameters
k1 <- expKern(x, x, param) # computing the covariance matrix using an exp. kernel
image2D(k1, theta = 0, xlab = "x", ylab = "y") # plotting the covariance matrix
# Q: what can you observe from the covariance matrix?
?mvrnorm # using the help from RStudio

## to complete  ##
## simulating some samples using the "mvrnorm" function
# samples <- mvrnorm(...)
# ?matplot # a function to plot the samples. The samples are indexed by columns
# Q: what can you observe from the samples?
n = 101
x <- seq(0, 1, 0.01) # regular grid
param <- c(1, 0.2) # covariance parameters
k1 <- mat5_2Kern(x, x, param) # computing the covariance matrix using an exp. kernel
par = (mfrow = c(1,1))
image2D(k1, theta = 0, xlab = "x", ylab = "y") # plotting the covariance matrix
# Q: what can you observe from the covariance matrix?
?mvrnorm # using the help from RStudio
samples = mvrnorm(n = 3, mu = rep(0,n), Sigma = k1)
matplot(x, t(samples), type = "l", main = "sample paths", ylab = "")

# theta : Longueur de corrélation
# nu : parametres du kernel   