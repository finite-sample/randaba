# Records R anticlust's exchange and local-maximum methods (k-means objective)
# from given initial splits, so the Python replica can be checked without R in CI.
library(anticlust)
library(jsonlite)
set.seed(20261015)
cases <- list()
for (i in 1:6) {
  n <- c(20, 40, 60, 100, 30, 90)[i]
  p <- c(1, 1, 3, 5, 2, 4)[i]
  k <- c(2, 2, 2, 2, 3, 3)[i]
  x <- matrix(rnorm(n * p), n, p)
  init <- sample(rep(seq_len(k), n / k))
  out <- anticlustering(x, K = init, objective = "variance", method = "exchange")
  cases[[i]] <- list(x = x, k = k, init = init - 1, out = out - 1,
                     method = "exchange")
}
for (i in 1:10) {
  n <- c(20, 40, 60, 48, 60, 96, 120, 200, 150, 240)[i]
  p <- c(1, 2, 3, 2, 5, 3, 1, 2, 4, 3)[i]
  k <- c(2, 2, 2, 4, 6, 8, 10, 20, 15, 40)[i]
  x <- matrix(rnorm(n * p), n, p)
  init <- sample(rep(seq_len(k), n / k))
  out <- anticlustering(x, K = init, objective = "variance",
                        method = "local-maximum")
  cases[[6 + i]] <- list(x = x, k = k, init = init - 1, out = out - 1,
                         method = "local-maximum")
}
write_json(list(anticlust = as.character(packageVersion("anticlust")),
                r = R.version.string, cases = cases),
           "tests/fixtures/anticlust_exchange.json", digits = NA, auto_unbox = TRUE)
