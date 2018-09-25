from scipy.optimize import minimize

test = blackflytest()
print("starting snap")
test.setAndSnap(1.0, 1.0)
print("done")
res = minimize(test.runOptimization, np.array([1,1]), method = 'Nelder-mead')