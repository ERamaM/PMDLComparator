# R Version: 3.6.3
#sudo apt install r-base
#sudo apt install build-essential libcurl5-gnutls-dev libxml2-dev libssl-dev libv8-dev libgeos-dev

if (!require("devtools")) {
  install.packages("devtools")
}

if (!require("scmamp")){
  devtools::install_github("b0rxa/scmamp")
}
if (!require("ggplot2")){
  install.packages("ggplot2", dependencies = TRUE)
  install.packages("geometry")
  install.packages("metRology")
  install.packages("MCMCpack")
}
if (!require("rstan")){
  install.packages("rstan")
}
if (!require("here")){
  install.packages("here")
}
if (!require("xtable")){
  install.packages("xtable")
}

library("scmamp")
library("ggplot2")
library("here")

if(dir.exists(file.path(getwd(), "results_processor/stat_tests_in_r"))){
  setwd(file.path(getwd(), "results_processor/stat_tests_in_r"))
}
data_acc <- read.csv("../processed_results/csv/next_activity/results.csv", row.names=1)

# Perform ranking with PlackettLuce
ranking <- bPlackettLuceModel(data_acc, min=FALSE, nsim=5000, nchains=10, parallel=TRUE)

sorted_probs <- data.frame(ranking$expected.win.prob)
sorted_probs$approach <- rownames(sorted_probs)
sorted_probs <- data.frame(sorted_probs[order(sorted_probs$ranking.expected.win.prob),])
write.csv(sorted_probs, "../processed_results/csv/next_activity/plackett_sorted_probs.csv")
print(xtable(sorted_probs, digits=4, caption="Approaches probability calculated by the Bayesian Plackett-Luce model"), file="../processed_results/latex/next_activity/plackett_sorted_probs.txt")

sorted_rankings <- data.frame(ranking$expected.mode.rank)
sorted_rankings$approach <- rownames(sorted_rankings)
sorted_rankings <- data.frame(sorted_rankings[order(sorted_rankings$ranking.expected.mode.rank),])
write.csv(sorted_rankings, "../processed_results/csv/next_activity/plackett_sorted_rankings.csv")
print(xtable(sorted_rankings, digits=4, caption="Approaches probability calculated by the Bayesian Plackett-Luce model"), file="../processed_results/latex/next_activity/plackett_sorted_rankings.txt")

# TODO: continue