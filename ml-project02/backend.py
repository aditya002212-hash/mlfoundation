from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import Field,computed_field,BaseModel
from typing import Annotated,Literal
import pickle
import pandas as pd

# loading ml model
model=pickle.load(open('model.pkl','rb'))

# pydantic model for validation 
class Input(BaseModel):
    age: Annotated[int,Field(...,gt=0)]
    height:Annotated[float,Field(...,gt=0)]
    weight:Annotated[float,Field(...,gt=0)]
    city:Annotated[str,Field(...)]
    smoker:Annotated[bool,Field(...,)]
    income:Annotated[float,Field(...,gt=0)]
    occupation:Annotated[str,Field(...,description='your occupation ')]
    
    # property in model to create bmi 
    @computed_field
    @property
    def bmi(self)->int:
        bmi=self.weight/self.height**2
        return bmi
    
    # property in model to create risk
    @computed_field
    @property
    def risk(self)->str:
        if self.bmi>30 and self.smoker:
            return 'high'
        elif self.bmi>27 and self.smoker:
            return 'mid'
        else:
            return 'low'
        
    # property in model to create age_group
    @computed_field
    @property
    def age_group(self)->str:
        if self.age < 60:
            if self.age < 45:
                if self.age <25:
                    return 'young'
                else:
                    return 'adult'
            else:
                return 'middleage'
        else:
            return 'senior'
        
    # property in model to create tier
    @computed_field
    @property
    def city_tier(self)->str:
        tier1= [
                "Mumbai",
                "Delhi",
                "Bengaluru",
                "Chennai",
                "Kolkata",
                "Hyderabad",
                "Pune",
                "Ahmedabad"]
        tier2= [
                "Agra", "Ajmer", "Akola", "Aligarh", "Amravati", "Amritsar", "Anand",
                "Asansol", "Aurangabad", "Bareilly", "Bardhaman", "Belagavi", "Berhampur",
                "Bhavnagar", "Bhiwandi", "Bhopal", "Bhubaneswar", "Bikaner", "Bilaspur",
                "Bokaro Steel City", "Bellary", "Bhilai", "Chandigarh", "Coimbatore",
                "Cuttack", "Dahod", "Dehradun", "Dhule", "Dombivli", "Dhanbad", "Durgapur",
                "Erode", "Faridabad", "Ghaziabad", "Gorakhpur", "Guntur", "Gurgaon",
                "Guwahati", "Gwalior", "Hamirpur", "Hubballi–Dharwad", "Indore", "Jabalpur",
                "Jaipur", "Jalandhar", "Jalgaon", "Jammu", "Jamshedpur", "Jamnagar",
                "Jhansi", "Jodhpur", "Kalaburagi", "Kakinada", "Kannur", "Kanpur",
                "Karimnagar", "Karnal", "Kochi", "Kolhapur", "Kollam", "Kota", "Kozhikode",
                "Kumbakonam", "Kurnool", "Ludhiana", "Lucknow", "Madurai", "Malappuram",
                "Mathura", "Mangaluru", "Meerut", "Mohali", "Moradabad", "Mysuru", "Nagpur",
                "Nanded", "Nadiad", "Nashik", "Nellore", "Noida", "Patna", "Pimpri-Chinchwad",
                "Puducherry", "Purulia", "Prayagraj", "Raipur", "Rajamahendravaram", "Rajkot",
                "Ranchi", "Rourkela", "Ratlam", "Raichur", "Saharanpur", "Salem", "Sangli",
                "Shimla", "Siliguri", "Solapur", "Srinagar", "Surat", "Thanjavur",
                "Thiruvananthapuram", "Thrissur", "Tiruchirappalli", "Tirunelveli",
                "Tiruvannamalai", "Ujjain", "Vijayapura", "Vadodara", "Varanasi",
                "Vasai-Virar", "Vijayawada", "Visakhapatnam", "Vellore", "Warangal"]
        
        if self.city in tier1:
            return 1
        elif self.city in tier2:
            return 2
        else:
            return 3

# fastapi app

app=FastAPI()


# post method to get the data of the person

@app.post('/predict')
def predict(data :Input):
    input=pd.DataFrame([{
        'income_lpa':data.income,
        'smoker':data.smoker, 
        'occupation':data.occupation, 
        'bmi':data.bmi,
        'age_group':data.age_group,
        'risk':data.risk,
        'city_tier':data.city_tier
    }])

    premium_class=model.predict(input)[0]
    return JSONResponse(status_code=200,content={'predicted_cat':premium_class})
