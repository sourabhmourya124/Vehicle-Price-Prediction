from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')
model = pickle.load(open("random_forest_model.pkl", "rb"))
print('[INFO] model loaded')
# model.predict([[2000,27000,0,6,0,1,0,1]])
@app.route('/',methods=['GET'])
def Home():
    return render_template('content.html')

@app.route("/predict", methods = ["POST"])
def predict():
    Fuel_Type_Diesel = 0
    # if request.methods == "POST":
    Year = int(request.form.get("Year"))
        # Selling_Price= float(request.form.get["Selling_Price"])
    Present_Price= float(request.form.get("Present_Price"))
    Kms_Driven= float(request.form.get("Kms_Driven"))
    Fuel_Type_Petrol= request.form.get("Fuel_Type_Petrol")
    if (Fuel_Type_Petrol == "Petrol"):
        Fuel_Type_Petrol = 1
    elif (Fuel_Type_Diesel == "Diesel"):
        Fuel_Type_Diesel = 1
    else:
        Fuel_Type_Diesel = 0
        Fuel_Type_Petrol = 0
        
    
    Seller_Type_Individual = request.form.get("Seller_Type_Individual")
    if (Seller_Type_Individual == "Individual"):
        Seller_Type_Individual = 1
    else:
        Seller_Type_Individual = 0
    Transmission_Manual = request.form.get("Transmission_Manual")
    if (Transmission_Manual=="Manual"):
        Transmission_Manual = 1
    else:
        Transmission_Manual = 0
    Owner_Individual = request.form.get("Owner_Individual")
    if Owner_Individual == "Owner_Individual":
        Owner_Individual =1
    else:
        Owner_Individual =0
    Year = 2022-Year    
    prediction=model.predict([[Present_Price,Kms_Driven,Owner_Individual,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Manual]])
    output = prediction[0]
    if output<0:
        return render_template('predict.html',prediction_text="Sorry you cannot sell this car")
    else:        
        return render_template('predict.html',prediction_text="You Can Sell The Car at {}".format(output))
    # else:
    #     return render_template('Predict.html')


if __name__== "__main__":
    app.run(debug=True)   