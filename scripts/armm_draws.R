# Draws assignments from ARMM (Yang et al. 2023, R package CARM) for one
# covariate matrix. Units are put in a fresh random order for every draw, as the
# ARM procedure prescribes (Qin et al. 2024, step 1); ARMM itself takes the
# order as given.
# Usage: Rscript armm_draws.R covariates.csv K ndraws seed out.csv
suppressPackageStartupMessages(library(CARM))
args <- commandArgs(trailingOnly = TRUE)
x <- read.csv(args[1])
k <- as.integer(args[2])
ndraws <- as.integer(args[3])
set.seed(as.integer(args[4]))
n <- nrow(x)
out <- matrix(NA_integer_, ndraws, n)
for (d in seq_len(ndraws)) {
  perm <- sample(n)
  r <- ARMM(covariate = x[perm, , drop = FALSE], assignment = NA, K = k,
            q = 0.75, method = "mean")
  out[d, perm] <- as.integer(r$assignment) - 1L
}
write.csv(out, args[5], row.names = FALSE)
