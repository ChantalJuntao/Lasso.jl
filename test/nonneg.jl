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

plot(1:1:16, xneg, color = "green")
plot(1:1:16, xnonneg, color = "red")
# plot(1:1:9, xlasso.coefs[:,40], color = "black")
plot(1:1:16, xlasso.coefs[:,end], color = "blue")
plot(1:1:16, xlassononneg.coefs[:,end], color = "purple")


cla()
for i = 1:20
    plot(1:1:16, xlasso.coefs[:,i], color = [0.0, 0.0, i/20])
end
for i = 21:40
    plot(1:1:16, xlasso.coefs[:,i], color = [0.0, (i-20)/20, 1.0])
end
for i = 41:56
    plot(1:1:16, xlasso.coefs[:,i], color = [(i-40)/20, 1.0, 1.0])
end
plot(1:1:16, xneg, color = "green")



cla()
for i = 1:20
    plot(1:1:16, xlassononneg.coefs[:,i], color = [0.0, 0.0, i/20])
end
for i = 21:40
    plot(1:1:16, xlassononneg.coefs[:,i], color = [0.0, (i-20)/20, 1.0])
end
for i = 41:57
    plot(1:1:16, xlassononneg.coefs[:,i], color = [(i-40)/20, 1.0, 1.0])
end
plot(1:1:16, xnonneg, color = "red")


#for regular lasso, smaller lamda means results that approach the best result
cla()
for i = 1:20
    plot(1:1:30, A*xlasso.coefs[:,i], color = [0.0, 0.0, i/20])
end
for i = 21:40
    plot(1:1:30, A*xlasso.coefs[:,i], color = [0.0, (i-20)/20, 1.0])
end
for i = 41:56
    plot(1:1:30, A*xlasso.coefs[:,i], color = [(i-40)/20, 1.0, 1.0])
end
plot(1:1:30, A*xneg, color = "green")


#for my nonnegative lasso implementation, the best result seems to sit in the middle
cla()
for i = 1:20
    plot(1:1:30, A*xlassononneg.coefs[:,i], color = [0.0, 0.0, i/20])
end
for i = 21:40
    plot(1:1:30, A*xlassononneg.coefs[:,i], color = [0.0, (i-20)/20, 1.0])
end
for i = 41:57
    plot(1:1:30, A*xlassononneg.coefs[:,i], color = [(i-40)/20, 1.0, 1.0])
end
plot(1:1:30, A*xnonneg, color = "red")


@test sum(abs.(xneg - xlasso.coefs[:,Int(ceil(end/2))])) >= sum(abs.(xnonneg - xlassononneg.coefs[:,Int(ceil(end/2))]))
