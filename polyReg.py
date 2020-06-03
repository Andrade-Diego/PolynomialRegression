import numpy as np
import matplotlib.pyplot as mp
import csv
import random

'''read csv as 2d numpy array of floats'''
def fileReader(path):
    data = (list(), list())
    with open(path, newline = '') as dataFile:
        reader = csv.reader(dataFile, delimiter = ',')
        for row in reader:
            data[0].append(float(row[0]))           #holds x values
            data[1].append(float(row[1]))           #holds y values
    return np.array(data)

'''graphs a given line and all the points of the given dataset'''
def graphRegression(dsID, xData, yData, thetas, mse):
	minim = np.amin(xData.T[1])
	maxim = np.amax(xData.T[1])
	x = np.linspace(minim, maxim, 500)
	y = 0
	eqnStr = "f(x) ="

	#creates the equation out of the lists, one to be graphed the other to print
	for i in range(0, len(thetas)):
		y += thetas[i] * x ** i
		eqnStr += (" %2.4f x^%i +" %(thetas[i], i))

	#removes last +
	eqnStr = eqnStr[:-1]
	#plot datapoints
	mp.plot(x, y)
	#plot line
	mp.scatter(xData.T[1],yData)
	print(eqnStr)
	mp.title("Dataset %i\nOrder: %i     MSE: %2.4f"  %(dsID, len(thetas)-1, mse))
	mp.show()
	return eqnStr

'''calculate mean squared error for a model'''
def MSEhelper(xData, yData, thetas):
    sumSquareErrors = 0

	#loops through datapoints
    for i in range(0, len(yData)):
        estimatedY = 0

		#dot product to find sum of features multiplied by their respective thetas
        estimatedY = np.dot(xData[i], thetas)

        sumSquareErrors += (yData[i] - estimatedY) ** 2
    mse = sumSquareErrors / len(yData)
    return mse

def weightUpdate(xData, yData, thetas):
	#learning rate
    ALPHA = .001

	#maximum number of iterations
    iterations = 100000000

    mse = 999999
    for i in range(0, iterations):
        prevMSE = mse

		#array for predicted y's
        prediction = np.dot(xData, thetas)

		#adjustment to thetas array
        thetas = thetas - (1/len(yData))*ALPHA*(np.dot(xData.T, prediction - yData))
        mse = MSEhelper(xData, yData, thetas)

		#breaks loop if progress slows between updates
        if(prevMSE - mse < .000001):
            break
    print ("mse:    ", mse)
    return thetas, mse


'''return regressionline of a given order'''
def polyReg(dsID, data, order):
	#populates thetas with random weights
    thetas = np.full(order + 1, random.random())
    xData = []
    yData = []

	#populates x and y data, x will have more features depending on order
    for i in range (0, len(data[0])):
        previousX = data[0][i]
        features = []
        for o in range (0, order+1):
            features.append(previousX ** o)
        xData.append(np.array(features))
        yData.append(data[1][i])

    thetas, mse = weightUpdate(np.array(xData), np.array(yData), thetas)
    line = graphRegression(dsID, np.array(xData), np.array(yData), thetas, mse)
    return mse, line

if __name__ == "__main__":
    #read data into variables
    synth1 = fileReader("synthetic-1.csv")
    synth2 = fileReader("synthetic-2.csv")
    synth3 = fileReader("synthetic-3.csv")
    ds1o1 = polyReg(1, synth1, 1)
    ds1o2 = polyReg(1, synth1, 2)
    ds1o4 = polyReg(1, synth1, 4)
    ds1o7 = polyReg(1, synth1, 7)
    ds2o1 = polyReg(2, synth2, 1)
    ds2o2 = polyReg(2, synth2, 2)
    ds2o4 = polyReg(2, synth2, 4)
    ds2o7 = polyReg(2, synth2, 7)
    ds3o1 = polyReg(3, synth3, 1)
    ds3o2 = polyReg(3, synth3, 2)
    ds3o4 = polyReg(3, synth3, 4)
    ds3o7 = polyReg(3, synth3, 7)
