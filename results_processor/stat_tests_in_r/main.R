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
if (!require("plyr")){
  install.packages("plyr")
}

library("scmamp")
library("ggplot2")
library("here")
library("plyr")

metric <- "accuracy"
#metric <- "recall"
#metric <- "precision"
#metric <- "f1-score"
#metric <- "brier"
#metric <- "mcc"

NO_CAMARGO <- FALSE
NO_SEPSIS <- TRUE

if(dir.exists(file.path(getwd(), "results_processor/stat_tests_in_r"))){
  setwd(file.path(getwd(), "results_processor/stat_tests_in_r"))
}
data_acc <- read.csv(paste("../processed_results/csv/next_activity/", metric, "_results.csv", sep=""), row.names=1)

print("Data acc")
if(NO_CAMARGO){
    data_acc <- data_acc[, !(colnames(data_acc) %in% c("Camargo"))]
    subproblem <- "delete_camargo"
}
if(NO_SEPSIS){
  data_acc <- data_acc[!(rownames(data_acc) %in% c("Sepsis", "Nasa")), ]
  subproblem <- "delete_sepsis"
}
# Perform ranking with PlackettLuce
# Save rankings and probs in files
ranking <- bPlackettLuceModel(data_acc, min=FALSE, nsim=500000, nchains=10, parallel=TRUE, seed=42)

# Plot the boxplot of rankings
#boxplot(ranking$posterior.weights)
probs <- data.frame(ranking$posterior.weights)
print(colnames(probs))
stack_probs <- stack(probs)
colnames(stack_probs)[colnames(stack_probs) == "ind"] <- "Approach"
colnames(stack_probs)[colnames(stack_probs) == "values"] <- "Probability"
if (NO_CAMARGO) {
  approaches <- c("Pasquadibisceglie", "Tax", "Hinkka", "Evermann", "Theis_no_resource", "Mauro", "Theis_resource")
}
if (NO_SEPSIS) {
  approaches <- c("Pasquadibisceglie", "Tax", "Camargo", "Hinkka", "Evermann", "Theis_no_resource", "Mauro", "Theis_resource")
}
#stack_probs$Approach <- factor(stack_probs$Approach, levels=c("Pasquadibisceglie", "Tax", "Camargo", "Hinkka", "Evermann", "Khan", "Theis w/o attr", "Mauro", "Theis w/ attr", "GRNN"))
stack_probs$Approach <- factor(stack_probs$Approach, levels=approaches)
dd_y95 <- ddply(stack_probs, .(Approach), function(x) quantile(x$Probability, 0.95))
dd_y05 <- ddply(stack_probs, .(Approach), function(x) quantile(x$Probability, 0.05))
dd_y50 <- ddply(stack_probs, .(Approach), function(x) median(x$Probability))
df <- data.frame(
  #x = c("Pasquadibisceglie", "Tax", "Camargo", "Hinkka", "Evermann", "Khan", "Theis_no_resource", "Mauro", "Theis_resource", "GRNN"),
  x = approaches,
  y05 = dd_y05[2],
  y50 = dd_y50[2],
  y95 = dd_y95[2]
)
colnames(df) <- c("Approaches", "y05", "y50", "y95")
print(df)
png(paste("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_ranking_boxplot.png", sep=""), width=1100, height=650)
ggplot(df, aes(x=Approaches)) + geom_point(aes(y=y50), colour="blue", size=3) + geom_errorbar(aes(ymin=y05, ymax=y95), width=0.5) + theme(text = element_text(size=25), axis.title.x=element_blank(), axis.title.y=element_blank()) + scale_x_discrete(labels=c("Theis_no_resource" = "Theis \n (w/o attr)", "Theis_resource" = "Theis \n (w/ attr)"))
dev.off()

sorted_probs <- data.frame(ranking$expected.win.prob)
sorted_probs$approach <- rownames(sorted_probs)
sorted_probs <- data.frame(sorted_probs[order(sorted_probs$ranking.expected.win.prob),])
write.csv(sorted_probs, paste("../processed_results/csv/next_activity/", subproblem, "/", metric, "_plackett_sorted_probs.csv", sep=""))
print(xtable(sorted_probs, digits=4, caption="Approaches probability calculated by the Bayesian Plackett-Luce model"), file=paste("../processed_results/latex/next_activity/", metric, "_plackett_sorted_probs.txt", sep=""))

sorted_rankings <- data.frame(ranking$expected.mode.rank)
sorted_rankings$approach <- rownames(sorted_rankings)
sorted_rankings <- data.frame(sorted_rankings[order(sorted_rankings$ranking.expected.mode.rank),])
write.csv(sorted_rankings, paste("../processed_results/csv/next_activity/", subproblem, "/", metric, "_plackett_sorted_rankings.csv", sep=""))
print(xtable(sorted_rankings, digits=4, caption="Approaches probability calculated by the Bayesian Plackett-Luce model"), file=paste("../processed_results/latex/next_activity/", metric, "_plackett_sorted_rankings.txt", sep=""))

# Plot the uncertainty of the three best models
index <- which(ranking$expected.mode.rank <= sort(ranking$expected.mode.rank, decreasing=FALSE)[3], arr.ind=TRUE)
weights <- ranking$posterior.weights[, index]
weights <- weights / rowSums(weights)
png(paste("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_best_three_uncertainty.png", sep=""), width=1100, height=650)
plotBarycentric(weights)
dev.off()

# Perform the signed_rank_test pairwise
best_one <- data_acc[, names(index)[1]]
best_two <- data_acc[, names(index)[2]]
best_three <- data_acc[, names(index)[3]]
# Pair 1
signed_test <- bSignedRankTest(best_one, best_two, rope=c(-0.01, 0.01), seed=42)
filename <- c("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_signed_rank_test_", names(index)[1], "_vs_", names(index)[2], ".png")
png(paste(filename, collapse=""))
plotSimplex(signed_test, A=names(index)[1], B=names(index)[2], plot.density=FALSE, alpha=0.5, posterior.label = TRUE)
dev.off()
# Pair 2
signed_test <- bSignedRankTest(best_two, best_three, rope=c(-0.01, 0.01), seed=42)
filename <- c("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_signed_rank_test_", names(index)[2], "_vs_", names(index)[3], ".png")
png(paste(filename, collapse=""))
plotSimplex(signed_test, A=names(index)[2], B=names(index)[3], plot.density=FALSE, alpha=0.5, posterior.label = TRUE)
dev.off()
# Pair 3
signed_test <- bSignedRankTest(best_two, best_three, rope=c(-0.01, 0.01), seed=42)
filename <- c("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_signed_rank_test_", names(index)[1], "_vs_", names(index)[3], ".png")
png(paste(filename, collapse=""))
plotSimplex(signed_test, A=names(index)[1], B=names(index)[3], plot.density=FALSE, alpha=0.5, posterior.label = TRUE)
dev.off()

# Perform the hierarchical tests
raw_data_acc <- read.csv(paste("../processed_results/csv/next_activity/", metric, "_raw_results.csv", sep=""), row.names=1)
if(NO_CAMARGO){
  raw_data_acc <- raw_data_acc[raw_data_acc$approach != "Camargo", ]
  subproblem <- "delete_camargo"
}
if(NO_SEPSIS){
  raw_data_acc <- raw_data_acc[raw_data_acc$log != "sepsis", ]
  raw_data_acc <- raw_data_acc[raw_data_acc$log != "nasa", ]
  subproblem <- "delete_sepsis"
}
# Number 1 is the best performing one
subset_1 <- raw_data_acc[raw_data_acc$approach == names(index[1]),][c("log", "fold", metric)]
reshaped_1 <- reshape(subset_1, direction="wide", idvar="log", timevar="fold")
rownames(reshaped_1) <- reshaped_1[,"log"]
# N2
subset_2 <- raw_data_acc[raw_data_acc$approach == names(index[2]),][c("log", "fold", metric)]
reshaped_2 <- reshape(subset_2, direction="wide", idvar="log", timevar="fold")
rownames(reshaped_2) <- reshaped_2[,"log"]
# N3
subset_3 <- raw_data_acc[raw_data_acc$approach == names(index[3]),][c("log", "fold", metric)]
reshaped_3 <- reshape(subset_3, direction="wide", idvar="log", timevar="fold")
rownames(reshaped_3) <- reshaped_3[,"log"]
reshaped_3$log <- NULL
reshaped_2$log <- NULL
reshaped_1$log <- NULL

matrix_3 <- data.matrix(reshaped_3)
matrix_2 <- data.matrix(reshaped_2)
matrix_1 <- data.matrix(reshaped_1)

print("Raw data acc: ")
print(raw_data_acc)
print("Matrix 2: ")
print(matrix_2)
print("Matrix 1: ")
print(matrix_1)

# Plot the pairwise comparisons
# P1
results <- bHierarchicalTest(matrix_1, matrix_2, rho=0.1, rope=c(-0.01, 0.01), nsim=50000, nchains=10, parallel=TRUE, seed=42)
filename <- c("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_hierarchical_test_", names(index)[1], "_vs_", names(index)[2], ".png")
png(paste(filename, collapse=""))
plotSimplex(results, A=names(index[1]), B=names(index[2]), posterior.label=TRUE, alpha=0.5)
dev.off()
# P2
results <- bHierarchicalTest(matrix_2, matrix_3, rho=0.1, rope=c(-0.01, 0.01), nsim=50000, nchains=10, parallel=TRUE, seed=42)
filename <- c("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_hierarchical_test_", names(index)[2], "_vs_", names(index)[3], ".png")
png(paste(filename, collapse=""))
plotSimplex(results, A=names(index[2]), B=names(index[3]), posterior.label=TRUE, alpha=0.5)
dev.off()
# P3
results <- bHierarchicalTest(matrix_1, matrix_3, rho=0.1, rope=c(-0.01, 0.01), nsim=50000, nchains=10, parallel=TRUE, seed=42)
filename <- c("../processed_results/latex/next_activity/plots/", subproblem, "/", metric, "_hierarchical_test_", names(index)[1], "_vs_", names(index)[3], ".png")
png(paste(filename, collapse=""))
plotSimplex(results, A=names(index[1]), B=names(index[3]), posterior.label=TRUE, alpha=0.5)
dev.off()

# Save results per dataset
rownames(results$additional$per.dataset) <- rownames(reshaped_3)
write.csv(data.frame(results$additional$per.dataset), paste("../processed_results/csv/next_activity/", subproblem, "/", metric, "_hierarchical_results_per_dataset.csv", sep=""))
