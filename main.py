import pandas as pd # import file reader
from sklearn.model_selection import train_test_split #divide data
from sklearn.tree import DecisionTreeClassifier #make decision tree / logic
from sklearn.metrics import accuracy_score #check how accurate the predictions with final answers

def returnStanding(yValue) : #returns string instead of boolean ./ int
    if  yValue == True:
        return "Passed"
    else:
        return "Failed"


df = pd.read_excel("dataset/students.xlsx")  #read excel
dimension = df.shape # stores the size


school_encoded = df["school"].map({
    "GP": 0,
    "MS": 1
}) # converts string values to int
df["school_encoded"] = school_encoded # creates a new column to be passed in X



internet_access = df["internet"].map({
    "no": 0,
    "yes": 1
}) # converts string values to int
df["internet_access"] = internet_access # creates a new column to be passed in X


sex = df["sex"].map({
    "F": 0,
    "M": 1
}) # converts string values to int
df["sex"] = sex # creates a new column to be passed in X




X = df[[ 
        "internet_access",
        "sex",
        "studytime",
        "absences",
        "failures"
        ]] #stores the independent variables:

y = df["G3"] >=12 #stores dependent var

standing = y.map(returnStanding) #used the function to return string

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) #split the values to training + testing

#model = DecisionTreeClassifier() #stored the model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train) #teaching it based on the datas

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

test_predictions = model.predict(X_test) #stores the predicted value of ai
test_accuracy = accuracy_score(y_test, test_predictions)


# UNFAMILIAR STUDENT
while True:
    student_name = input("Enter student name (0 to exit): ")

    if student_name == "0":
        break

    internet_access = input("Has internet access? (0 for no, 1 for yes): ")
    sex = input("Enter sex (0 for F, 1 for M): ")
    study_time = float(input("Enter study time: "))
    absences = int(input("Enter absences: "))
    past_failures = int(input("Enter past failures: "))

    new_student = pd.DataFrame([{
        "internet_access": int(internet_access),
        "sex": int(sex),
        "studytime": study_time,
        "absences": absences,
        "failures": past_failures
    }])

    new_student_prediction = model.predict(new_student)
    new_student_result = returnStanding(new_student_prediction[0])

    print("STUDENT:", student_name)
    print("Result:", new_student_result)
    print("")

# print("")
# print("MODEL STATS")
# print("")
# print("")


# print(y.value_counts())
# print(y_test.value_counts())
# print("Training shape", X_train.shape)
# print("Testing shape", X_test.shape)
# print("")

# print("Training Accuracy:", train_accuracy)
# print("Testing Accuracy:", test_accuracy)

# print("")
# print(model.feature_importances_)
# print("")
# print("")
