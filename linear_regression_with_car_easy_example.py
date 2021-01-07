
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
predictedCO2 = loaded_model.score(X_test, y_test)
print(predictedCO2)
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = reg.predict(X_test)
print('coefficient: \n  %.2f' , reg.coef_)
print('Coefficient of determination:  %.2f' % accuracy_score(predictedCO2,y_test))

X1=int (input("enter weight of car"))
X2= int (input("enter volume of car"))
XF=np.array([[X1,X2]])
CO2 = loaded_model.predict(XF)
print('the final co2 is',CO2)
