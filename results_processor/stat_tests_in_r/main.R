#sudo apt install r-base
#sudo apt install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev libv8-dev
if (!require("devtools")) {
  install.packages("devtools")
}

if (!require("scmamp")){
  devtools::install_github("b0rxa/scmamp")
}
if (!require("ggplot2")){
  install.packages("ggplot2")
}
if (!require("rstan")){
  install.packages("rstan")
}

library("scmamp")
library("ggplot2")

data <- read.csv("processed_results/csv/acc.csv", row.names=1)
print(data)

# In Demšar (2006) the author proposes a plot to visually check the differences,
# the critical differece plot. This kind of plot can be created using the plotCD
# function, which has two parameters, the data.
# matrix and the significance level. In the plot, those algorithms that
# are not joined by a line can be regarded as different.
plotCD(data, alpha=0.05, cex=1.25)

data_bayesian <- bPlackettLuceModel(data, min=FALSE, nsim=5000, nchains=10, parallel=TRUE)
print(data_bayesian$expected.win.prob)
print(data_bayesian$expected.mode.rank)
#plotDensities(data)