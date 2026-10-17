# ==================== 1. 读取数据 ====================
dat <- read.csv(file.choose(), stringsAsFactors = FALSE)
cat("列名：\n"); print(colnames(dat))

# ==================== 2. 配对变量定义 ====================
var_pairs <- list(
  c("Proximal ureteral width1",         "Proximal ureteral width2",         "Proximal ureteral width "),
  c("UWT1",          "UWT2",          "UWT"),
  c("Stone.length1", "Stone.length2", "StoneLength"),
  c("Stone.width1",  "Stone.width2",  "StoneWidth"),
  c("Hydronephrosis area 1",          "Hydronephrosis area 2",          "Hydronephrosis area "),
  c("UWA1",          "UWA2",          "UWA"),
  c("Stone attenuation1",     "Stone attenuation2",     "Stone attenuation")
)

# ==================== 3. 手动计算 ICC(2,1) ====================
manual_icc21 <- function(x, y) {
  # 删除缺失值
  ok <- complete.cases(x, y)
  x <- x[ok]; y <- y[ok]
  
  n <- length(x)
  k <- 2
  m <- cbind(x, y)
  
  row_means  <- rowMeans(m)
  col_means  <- colMeans(m)
  grand_mean <- mean(m)
  
  SS_total <- sum((m - grand_mean)^2)
  SS_rows  <- k * sum((row_means - grand_mean)^2)
  SS_cols  <- n * sum((col_means - grand_mean)^2)
  SS_error <- SS_total - SS_rows - SS_cols
  
  df_rows  <- n - 1
  df_cols  <- k - 1
  df_error <- (n - 1) * (k - 1)
  
  MSR <- SS_rows / df_rows
  MSC <- SS_cols / df_cols
  MSE <- SS_error / df_error
  
  # ICC(2,1) 点估计
  icc_val <- (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
  
  # 95% 置信区间（McGraw & Wong 1996）
  alpha  <- 0.05
  F_U    <- qf(1 - alpha/2, df_rows, df_error)   # 上临界值
  F_L    <- qf(alpha/2,     df_rows, df_error)   # 下临界值
  
  ci_low  <- (MSR - MSE * F_U) /
    (MSR + (k - 1) * MSE * F_U + k * (MSC - MSE) / n)
  ci_high <- (MSR - MSE * F_L) /
    (MSR + (k - 1) * MSE * F_L + k * (MSC - MSE) / n)
  
  list(icc = icc_val, low = ci_low, high = ci_high, n = n)
}

# ==================== 4. 循环计算 ====================
res_list <- list()

for (i in seq_along(var_pairs)) {
  v1_name <- var_pairs[[i]][1]
  v2_name <- var_pairs[[i]][2]
  lab     <- var_pairs[[i]][3]
  
  x <- as.numeric(dat[[v1_name]])
  y <- as.numeric(dat[[v2_name]])
  
  r <- manual_icc21(x, y)
  
  # Bland–Altman
  diff    <- x - y
  bias    <- mean(diff, na.rm = TRUE)
  mad     <- mean(abs(diff), na.rm = TRUE)
  sd_diff <- sd(diff, na.rm = TRUE)
  loa_low  <- bias - 1.96 * sd_diff
  loa_high <- bias + 1.96 * sd_diff
  
  res_list[[i]] <- data.frame(
    variable    = lab,
    ICC21       = round(r$icc, 4),
    CI95_low    = round(r$low, 4),
    CI95_high   = round(r$high, 4),
    BA_bias     = round(bias, 4),
    BA_MAD      = round(mad, 4),
    BA_LOA_low  = round(loa_low, 4),
    BA_LOA_high = round(loa_high, 4),
    n           = r$n
  )
}

# ==================== 5. 输出并保存 ====================
result_df <- do.call(rbind, res_list)
print(result_df, row.names = FALSE)
write.csv(result_df, "icc_result.csv", row.names = FALSE)
cat("\n结果已保存为 icc_result.csv\n")