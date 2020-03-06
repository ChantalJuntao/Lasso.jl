using TestImages, Images, NNLS

img = testimage("pirate.tif")
A = map(x->Float64(x.val), restrict(restrict(img))[1:30, 15:30])

x = zeros(16)
x[3] = -2
x[5] = 1
x[8] = 5
x[9] = 4
x[12] = -2
x[13] = -1


b =  A*x

xneg = A\b

xnonneg = nnls(A,b)

xlasso = fit(LassoPath,A,b; algorithm = NaiveCoordinateDescent)
xlassononneg = fit(LassoPath, A, b; algorithm = NaiveCoordinateDescent, nonneg = true)

plot(1:1:16, xneg, color = "red")
plot(1:1:16, xnonneg, color = "blue")
# plot(1:1:9, xlasso.coefs[:,40], color = "black")
plot(1:1:16, xlasso.coefs[:,end], color = "green")
plot(1:1:16, xlassononneg.coefs[:,end], color = "purple")

cla()
plot(1:1:30, b, color = "red")
plot(1:1:30, A*xlasso.coefs[:,10], color = "orange")
plot(1:1:30, A*xlasso.coefs[:,20], color = "yellow")
plot(1:1:30, A*xlasso.coefs[:,30], color = "green")

#λ doesn't make a lot of sense. Overfitting seems to be an issue
plot(1:1:30, A*xnonneg, color = "blue")
plot(1:1:30, A*xlassononneg.coefs[:,10], color = "cyan")
plot(1:1:30, A*xlassononneg.coefs[:,20], color = "pink")
plot(1:1:30, A*xlassononneg.coefs[:,30], color = "purple")


@test sum(abs.(xneg - xlasso.coefs[:,Int(ceil(end/2))])) >= sum(abs.(xnonneg - xlassononneg.coefs[:,Int(ceil(end/2))]))
