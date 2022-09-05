# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:07:37 2017

@author: Nathan Larson
CSC 391/310
Dr. Lutz Hamel

This program will analyze data based on marijuana sales from January 2014
to September 2016.
"""
import pandas
import os
import numpy
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime

directory = "C:\\Users\\natel\\Documents\\Python Data Files\\CSV Files"
medicalSlope = 0.0
retailSlope = 0.0
linearScorem = 0.0
linearScorer = 0.0
treeScorem = 0.0
treeScorer = 0.0
KNN_Scorem = 0.0
KNN_Scorer = 0.0
black = '\033[0m'  # black (default)
red  = '\033[31m' # red
green  = '\033[32m' # green
orange  = '\033[33m' # orange
blue  = '\033[34m' # blue
purple  = '\033[35m' # purple
try:
    os.chdir(directory)
except:
    print("Directory Not Found.")
    
try:
    marijuana_df = pandas.read_csv("marijuana_gross_sales.csv")
except:
    print("CSV file not found.")
    
     # drops NaN values to clean up the data set
marijuana_df = marijuana_df.dropna()
    # reverse the data set since the csv file is sorted as newest entry = first entry,
    # and we want oldest entry = first entry to simplify things
marijuana_df = marijuana_df[::-1]

def getNumericMonth(monthArray):
    numericMonth = numpy.empty(monthArray.size)
    for i in range(monthArray.size):
        if(monthArray[i].strip() == "JANUARY"):
            numericMonth[i] = 1
        elif(monthArray[i].strip() == "FEBRUARY"):
            numericMonth[i] = 2
        elif(monthArray[i].strip() == "MARCH"):
            numericMonth[i] = 3
        elif(monthArray[i].strip() == "APRIL"):
            numericMonth[i] = 4
        elif(monthArray[i].strip() == "MAY"):
            numericMonth[i] = 5
        elif(monthArray[i].strip() == "JUNE"):
            numericMonth[i] = 6
        elif(monthArray[i].strip() == "JULY"):
            numericMonth[i] = 7
        elif(monthArray[i].strip() == "AUGUST"):
            numericMonth[i] = 8
        elif(monthArray[i].strip() == "SEPTEMBER"):
            numericMonth[i] = 9
        elif(monthArray[i].strip() == "OCTOBER"):
            numericMonth[i] = 10
        elif(monthArray[i].strip() == "NOVEMBER"):
            numericMonth[i] = 11
        elif(monthArray[i].strip() == "DECEMBER"):
            numericMonth[i] = 12
        else:
            numericMonth[i] = 0
    return numericMonth

def showPlot():
    global marijuana_df
    marijuana_df = marijuana_df.set_index('GROSS_SALES_TYPE')

    medicalSales_df = marijuana_df.loc['Medical Total Gross Sales']
    retailSales_df = marijuana_df.loc['Retail Total Gross Sales']
    
    marijuana_df = marijuana_df.reset_index()
    medicalSales_df = medicalSales_df.reset_index()
    retailSales_df = retailSales_df.reset_index()
    
    plt.plot(retailSales_df.daysPassed, retailSales_df.GROSS_SALES, 'b-')
    plt.plot(medicalSales_df.daysPassed, medicalSales_df.GROSS_SALES, 'g-')
    
    plt.legend(['retail sales', 'medical sales'])
    plt.xlabel('Days Since January 01, 2014')
    plt.ylabel('Gross Revenue ($)')
    plt.show(block = True)
    return
    

def predictLinearSales():
    year = int(input("Enter a year you would like to predict(integer value): "))
    month = str(input("Enter the full month you would like to predict(ie. January): "))
    monthValue = getNumericMonth(numpy.array([month.upper()]))
    days = toDate(numpy.array([year]), monthValue)
    print("\nPredicted Medical Sales: " + green + "$" + str("%.2f" % float(days * medicalSlope)) + black + " with " + green + str("%.2f" % (100 * linearScorem)) + black + "% accuracy.")
    print(blue + "Predicted Retail Sales: " + green + "$" + str("%.2f" % float(days * retailSlope)) + black + " with " + green + str("%.2f" % (100 * linearScorer)) + black + "% accuracy.\n")
    
    
def linearRegression():
    # declare global variables to allow manipulation of variables
    # used in a number of functions
    global marijuana_df
    global medicalSlope
    global retailSlope
    global linearScorem
    global linearScorer
    
    # set selected index of the data frame
    marijuana_df = marijuana_df.set_index('GROSS_SALES_TYPE')
    
    # create two seperate data frames for medical sales and retail sales
    medicalSales_df = marijuana_df.loc['Medical Total Gross Sales']
    retailSales_df = marijuana_df.loc['Retail Total Gross Sales']
    
    # initialize both medical and retail linear regression models
    linearModelm = LinearRegression()
    linearModelr = LinearRegression()
    
    # reset index to make manipulating the data frames easier
    marijuana_df = marijuana_df.reset_index()
    medicalSales_df = medicalSales_df.reset_index()
    retailSales_df = retailSales_df.reset_index()
    
    # initialize both retail and medical X values
    Xm = medicalSales_df.drop(['YEAR','GROSS_SALES_TYPE','MONTH','GROSS_SALES', 'numMonth'], axis = 1)
    Xr = retailSales_df.drop(['YEAR','GROSS_SALES_TYPE','MONTH','GROSS_SALES', 'numMonth'], axis = 1)
    ym = medicalSales_df['GROSS_SALES']
    yr = retailSales_df['GROSS_SALES']
    
    # fit columns to the model
    linearModelm.fit(Xm, ym)
    linearModelr.fit(Xr, yr)
    
    # model prediction based off given columns to "learn" from
    ypm = linearModelm.predict(Xm)
    ypr = linearModelr.predict(Xr)
    
    # calculate and display linear regression R^2 score for each data frame
    linearScorem = linearModelm.score(Xm, ym)
    linearScorer = linearModelr.score(Xr, yr)
    print( green + "\nMedical Sales Linear Regression R^2 Score: {}".format(linearScorem))
    print( blue + "Retail Sales Linear Regression R^2 Score: {}\n".format(linearScorer) + black)
    
    # plot medical sales regression model to figure
    plt.plot(Xm.daysPassed, ym, 'g-')
    plt.plot(Xm.daysPassed, ypm, 'g--')
    plt.title("Linear Regression Models")
    plt.plot(Xr.daysPassed, yr, 'b-')
    plt.plot(Xr.daysPassed, ypr, 'b--')
    plt.legend(["Actual Medical Sales", "Predicted Medical Sales", "Actual Retail Sales", "Predicted Retail Sales"], loc = 'upper left')
    plt.ylabel("Sales ($)")
    plt.xlabel("Days Since January 01, 2014")
    
    
    medicalSlope = float(linearModelm.coef_)
    retailSlope = float(linearModelr.coef_)
    
    plt.show(block = True)
    
    return


def regressionTree():
    # declare global variables to allow manipulation of variables
    # used in a number of functions
    global marijuana_df
    global treeScorem
    global treeScorer
    
    # set selected index of the data frame
    marijuana_df = marijuana_df.set_index('GROSS_SALES_TYPE')
    
    # create two seperate data frames for medical sales and retail sales
    medicalSales_df = marijuana_df.loc['Medical Total Gross Sales']
    retailSales_df = marijuana_df.loc['Retail Total Gross Sales']
    
    # initialize both medical and retail regression tree models
    treeModelm = DecisionTreeRegressor()
    treeModelr = DecisionTreeRegressor()
    param_grid ={'max_depth': list(range(1, 25))}
    gridm = GridSearchCV(treeModelm, param_grid, cv = 5)
    gridr = GridSearchCV(treeModelr, param_grid, cv = 5)
    
    # reset index to make manipulating the data frames easier
    marijuana_df = marijuana_df.reset_index()
    medicalSales_df = medicalSales_df.reset_index()
    retailSales_df = retailSales_df.reset_index()
    
    # initialize both retail and medical X values
    Xm = medicalSales_df.drop(['YEAR','GROSS_SALES_TYPE','MONTH','GROSS_SALES', 'numMonth'], axis = 1)
    Xr = retailSales_df.drop(['YEAR','GROSS_SALES_TYPE','MONTH','GROSS_SALES', 'numMonth'], axis = 1)
    ym = medicalSales_df['GROSS_SALES']
    yr = retailSales_df['GROSS_SALES']
    
    # perform grid search
    gridm.fit(Xm, ym)
    gridr.fit(Xr, yr)
    
    # print grid search results
    print(green + "\nBest Medical Parameters: {}".format(gridm.best_params_))
    print(blue + "Best Retail Parameters: {}\n".format(gridr.best_params_))

    # setting models to the best estimator parameters
    treeModelm = gridm.best_estimator_
    treeModelr = gridr.best_estimator_
    
    # fit columns to the model
    treeModelm.fit(Xm, ym)
    treeModelr.fit(Xr, yr)
    
    # plot model with data
    xfit = numpy.arange(0.0, 1000, 1)[:, numpy.newaxis]
    ymfit = treeModelm.predict(xfit)
    yrfit = treeModelr.predict(xfit)
    plt.title("Regression Tree Models")
    plt.scatter(Xm, ym, color = 'green')
    plt.scatter(Xr, yr, color = 'blue')
    plt.plot(xfit, ymfit, 'g-')
    plt.plot(xfit, yrfit, 'b-')
    plt.legend(["max_depth = {}".format(treeModelm.max_depth), "max_depth = {}".format(treeModelr.max_depth), "Medical Sales Data", "Retail Sales Data"])
    plt.ylabel("Sales ($)")
    plt.xlabel("Days Since January 01, 2014")
    
    # print R^2 values for the two models
    print(green + "Medical Sales Regression Tree R^2 Score: {}".format(treeModelm.score(Xm, ym)))
    print(blue + "Retail Sales Regression Tree R^2 Score: {}\n".format(treeModelr.score(Xr, yr)) + black)
    
    plt.show(block = True)
    
    
def KNNregression():
    # declare global variables to allow manipulation of variables
    # used in a number of functions
    global marijuana_df
    global KNNScorem
    global KNNScorer
    
    # set selected index of the data frame
    marijuana_df = marijuana_df.set_index('GROSS_SALES_TYPE')
    
    # create two seperate data frames for medical sales and retail sales
    medicalSales_df = marijuana_df.loc['Medical Total Gross Sales']
    retailSales_df = marijuana_df.loc['Retail Total Gross Sales']
    
    # initialize both medical and retail regression tree models
    KNNmodel_m = KNeighborsRegressor(n_neighbors = 3)
    KNNmodel_r = KNeighborsRegressor(n_neighbors = 2)

    # reset index to make manipulating the data frames easier
    marijuana_df = marijuana_df.reset_index()
    medicalSales_df = medicalSales_df.reset_index()
    retailSales_df = retailSales_df.reset_index()
    
    # initialize both retail and medical X values
    Xm = medicalSales_df.drop(['YEAR','GROSS_SALES_TYPE','MONTH','GROSS_SALES', 'numMonth'], axis = 1)
    Xr = retailSales_df.drop(['YEAR','GROSS_SALES_TYPE','MONTH','GROSS_SALES', 'numMonth'], axis = 1)
    ym = medicalSales_df['GROSS_SALES']
    yr = retailSales_df['GROSS_SALES']

    # fit models to data
    KNNmodel_m.fit(Xm, ym)
    KNNmodel_r.fit(Xr, yr)
    
    # compute KNN R^2 score
    print(green + "\nMedical Sales KNN Regression R^2 Score: {}".format(KNNmodel_m.score(Xm, ym)))
    print(blue + "Retail Sales KNN Regression R^2 Score: {}\n".format(KNNmodel_r.score(Xr, yr)) + black)
    
    
    # plot model with data
    xfit = pandas.DataFrame([i for i in range(0, 1000)])
    ymfit = KNNmodel_m.predict(xfit)
    yrfit = KNNmodel_r.predict(xfit)
    plt.title("KNN Regression Models")
    plt.scatter(Xm, ym, color = "green")
    plt.scatter(Xr, yr, color = "blue")
    plt.plot(xfit, ymfit, 'g-')
    plt.plot(xfit, yrfit, 'b-')
    plt.legend(["n-neighbors: {}".format(KNNmodel_m.n_neighbors), "n-neighbors: {}".format(KNNmodel_r.n_neighbors), "Medical Sales Data", "Retail Sales Data"])
    plt.ylabel("Sales ($)")
    plt.xlabel("Days Since January 01, 2014")
    
    plt.show(block = True)
    
    
def toDate(years, months):
    elapsedDays = numpy.empty(months.size)
    for i in range(years.size):
        total = 0
        if(str(years[i]).strip() == "2015"):
            total += 365
        elif(str(years[i]).strip() == "2016"):
            total += (365*2)
        elif(int(years[i]) > 2016):
            total += (365 * (int(years[i]) - 2014))
            
        if(months[i] >= 2):
            total += 28
        if(months[i] >= 3):
            total += 31
        if(months[i] >= 4):
            total += 30
        if(months[i] >= 5):
            total += 31
        if(months[i] >= 6):
            total += 30
        if(months[i] >= 7):
            total += 31
        if(months[i] >= 8):
            total += 31
        if(months[i] >= 9):
            total += 30
        if(months[i] >= 10):
            total += 31
        if(months[i] >= 11):
            total += 30
        if(months[i] >= 12):
            total += 31
        elapsedDays[i] = total
    return elapsedDays

def printChoice():
    print('1.) Exit')
    print('2.) Print DataFrame')
    print('3.) Plot Data')
    print('4.) Linear Regression')
    print('5.) Regression Tree')
    print('6.) Predict Future Sales using Linear Regression Model')
    print(red + '\t -Must Create Model Before Predicting (option 4)' + black)
    print('7.) KNN Regression')
    
# adds column to give the month a numerical value
marijuana_df['numMonth'] = getNumericMonth(marijuana_df.MONTH.values)
marijuana_df['daysPassed'] = toDate(marijuana_df.YEAR.values, marijuana_df.numMonth.values)

print("\n\nThis program analyzes and displays data about Marijuana use in Colorado from 2014-2016.")
print('What would you like to do?')
printChoice()
x = int(input('selection(number): '))
while(x != 1):
    if(x == 2):
        print(marijuana_df)
    elif(x == 3):
        print("\ndisplaying graph...\n")
        showPlot()
    elif(x == 4):
        print("\ndisplaying graph...\n")
        linearRegression()
    elif(x == 5):
        print("\ndisplaying graph...\n")
        regressionTree()
    elif(x == 6):
        if(medicalSlope == 0.0 and retailSlope == 0.0):
            print(red + "\nYou first must choose the Linear Regression option to build a model!\n" + black)
        else:
            predictLinearSales()
    elif(x == 7):
        print("\ndisplaying graph...\n")
        KNNregression()
    elif(x == 8):
        pass
    else:
        print("\nnot a valid choice!\n")
    printChoice()
    x = int(input('\nselection(number): '))
        
print("Goodbye!")





