# generate dummy data
set.seed(123)
x <- seq(0, 1, length.out = 100)
y_true <- sin(5*x)
y <- y_true + rnorm(length(x), mean = 0, sd = 0.3)

# Fit LOESS model
span <- 0.4
fit <- loess(y ~ x, span = span, degree = 2)

# Predict LOESS curve
x_grid <- x
y_fit <- predict(fit, newdata = data.frame(x = x_grid))

# Define a neighborhood around a target point; just for illustration
x_target <- x[floor(0.3*length(x))]
x_dist <- abs(x - x_target)
neighborhood_size <- floor(span * length(x))
neighborhood_idx <- order(x_dist)[1:neighborhood_size]

# Plot
plot(x, y, col = "blue", main = "LOESS Fit with Example Neighborhood",
     xlab = "x", ylab = "y", bg="transparent")

points(x_target, y[which(x == x_target)], col = "red", pch = 19, cex = 1.5)
points(x[neighborhood_idx], y[neighborhood_idx], col = "red")

lines(x_grid, y_fit, col = "darkgreen", lwd = 2)
lines(x_grid, y_true, col="black", lwd=1, lty=2)

legend(
  "topright",
  legend = c("All points", "Target point", "Neighborhood", "LOESS curve", "True curve"),
  col = c("blue", "red", "red", "darkgreen", "black"), 
  pch = c(1,19, 1, NA, NA), 
  lty = c(NA, NA, NA, 1, 2), 
  lwd = c(NA, NA, NA, 1, 2)
)

