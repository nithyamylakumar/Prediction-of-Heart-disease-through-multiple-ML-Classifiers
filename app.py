import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import numpy as np
from PIL import Image
import altair as alt
import hiplot as hip
from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression


st.header("Heart disease dataset from UCI ", anchor = None)
st.subheader("Reason for the study:")
st.write("Heart disease or Cardiovascular disease (CVD) is a class of diseases that involve the **heart** or **blood vessels**. Cardiovascular diseases are the leading cause of death globally. \n Together CVD resulted in **17.9 million deaths (32.1%) in 2015**. Deaths, at a given age, from CVD are more common and have been increasing in much of the developing world, while rates have declined in most of the developed world since the 1970s.")
df = pd.read_csv("heart.csv")

st.header('Objective of the EDA')
st.markdown(
    '<p style="color:blue; font-size:22px"> The objectives of the EDA are as follows:- To get an overview of the distribution of the dataset, Check for missing numerical values, outliers or other anomalies in the dataset and Discover patterns and relationships between variables in the dataset. \n iv. Check the underlying assumptions in the dataset ', unsafe_allow_html=True
)

st.title('Dataset Information')
st.markdown(
    '<p style= "color: black; font-size:22px"> This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0). The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.', unsafe_allow_html=True
)

st.header('Sample dataset')
st.table(df.iloc[1:].head(10))

st.header('Data Description')
st.table(pd.read_csv('data_description.csv'))


st.title('Important points about the dataset')
st.markdown(
    '<p style= "color: black; font-size:22px"> 1. Sex, Fasting_BS, Ex_Induced_Ang, and Target are character variables and their data type should be object. But since they are encoded (i.e., 1 and 0 ) their data type is given as int64' , unsafe_allow_html=True
)
# Statistical properties of dataset

st.dataframe(df.describe())


# Univariate analysis of the features

tab1, tab2, tab3, tab4, tab5= st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Prediction of Heart Disease", "Evaluation Metrics"])

x = df.columns.tolist()
y = df.columns.tolist()

with tab1:
    st.header('Univariate analysis of the different columns')
    st.text('Our feature variable of interest is the Target variable. It denotes the presence or absence or heart disease in a patient. From the univariate analysis it is evident that there are 165 patients suffering from heart disease and 138 patients who do not have a heart disease.')
    st.title("Histogram")
    x_axis_1 = st.selectbox(label = "Select an attribute", options = x, key = 1)
    plt.figure(figsize=(6,5))
    img_1 = px.histogram(df, x = x_axis_1, color = 'Target', text_auto= True)
    st.plotly_chart(img_1)

    st.header("Distribution plot")
    x_axis_2 = st.selectbox(label = "Select an attribute", options = x, key = 2 )
    plt.figure(figsize= (6,5))
    img_2 = px.histogram(df, x= x_axis_2, color='Sex', marginal="box",hover_data=df.columns, text_auto= True)
    st.plotly_chart(img_2)

with tab2:
    st.header('Bivariate analysis of different attributes')
    x_axis_3 = st.selectbox(label = "Select an attribute", options = x, key = 3 )
    y_axis_3 = st.selectbox(label = "Select an attribute", options = y, key = 4 )
    #colour = ['Sex','Fasting_BS', 'Slope', 'Ex_Induced_Ang','Target']
    img_3 = alt.Chart(df).mark_circle(size=60).encode(
    x=x_axis_3,
    y= y_axis_3,
    color= 'Sex',
    tooltip=['Cholesterol', 'Fasting_BS', 'Sex']
).interactive()
    st.altair_chart(img_3)

with tab3:

    # st.header('Hi Plot', anchor = None)
    # img_4 = hip.Experiment.from_dataframe(df)
    # img_4.to_streamlit(ret = "selected_uids", key = 'hip').display()


    st.header('Facet Plot', anchor = None)
    x_axis_4 = st.selectbox(label = "Select a column for x", options = x, key = 5 )
    y_axis_4 = st.selectbox(label = "Select a column for y", options = y, key = 6 )

    img_5 = alt.Chart(df).mark_point().encode(
        x = alt.X(x_axis_4, axis = alt.Axis(title = x_axis_4,grid = False)), y = alt.Y(y_axis_4, axis = alt.Axis(title = y_axis_4,grid = False)),
        color= alt.Color('Target'),tooltip =['Target', 'Sex', 'Max_Heart_Rate'],
        facet= alt.Facet('Target',columns=2)).interactive()
    st.altair_chart(img_5)


    st.header('Pairplot', anchor = None)
    img_6 = sns.pairplot(
    df,
    x_vars=["Age", "Resting_BP", "Cholesterol", "Max_Heart_Rate", "Oldpeak"],
    y_vars=["Age", "Resting_BP", "Cholesterol", "Max_Heart_Rate", "Oldpeak"],
    hue = 'Target',
    markers=["o", "s"],
    diag_kind="hist",
)
    st.pyplot(img_6)

with tab4:

        image = Image.open('image.jpg')
        st.image(image,caption='Do your part by caring for your HEART' )

        st.write("""
    # Heart disease Prediction App
    Check your cardiac health at your own space and time within seconds.
    """)

        import streamlit as st 
        import numpy as np 

        import matplotlib.pyplot as plt
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay
        from sklearn.metrics import accuracy_score
        from sklearn import datasets, svm, metrics

        st.title('Choose your classifier')

        st.write("""
        # Explore the various options 
        """)


        classifier_name = st.selectbox(
            'Select classifier',
            ('KNN', 'SVM','Decision Tree', 'Random Forest', 'Logistic Regression')
        )

    
    
        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == 'SVM':
                C = st.slider('C', 0.01, 10.0)
                params['C'] = C
            elif clf_name == 'KNN':
                K = st.slider('K', 1, 15)
                params['K'] = K
            elif clf_name == 'DecisionTree':
                max_depth = st.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth

            elif clf_name == 'Logistic Regression':
                random_state = st.slider('random_state', 0 , 42)
                params['random_state'] = random_state
            else:
                max_depth = st.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
                n_estimators = st.slider('n_estimators', 1, 100)
                params['n_estimators'] = n_estimators
            return params

        params = add_parameter_ui(classifier_name)

        def get_classifier(clf_name, params):
            clf = None
            if clf_name == 'SVM':
                clf = SVC(C=params['C'])
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            elif clf_name == 'DecisionTree':
                clf = DecisionTreeClassifier(max_depth = params['max_depth'])
            elif clf_name == 'Logistic Regression':
                clf = LogisticRegression(random_state = params['random_state'])
            else:
                clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                    max_depth=params['max_depth'], random_state=1234)
            return clf

        
        st.header("""Select your features""")
        st.write(' By providing the following biological parameters, the app would be able to predict the health of your heart with a good accuracy.')
        Age = st.slider('Enter your age: ', 0,100)
        Sex  = st.radio('Sex', (0,1))
        Chest_pain_type = st.radio('Chest pain type',(0,1,2,3))
        Resting_BP = st.text_input('Resting Blood Pressure', 130)
        Cholesterol = st.text_input('Serum cholestoral in mg/dl:', 200)
        Fasting_BS = st.radio('Fasting blood sugar',(0,1))
        Rest_ECG = st.radio('Resting electrocardiographic results:', (0,1))
        Max_Heart_Rate = st.text_input('Maximum heart rate achieved:', 170)
        Ex_Induced_Ang = st.radio('Exercise induced angina: ',(0,1))
        Oldpeak = st.text_input('oldpeak ', 2)
        Slope = st.radio('he slope of the peak exercise ST segmen: ', (0,1,2))
        BLD_VES_Fluros = st.radio('number of major vessels',(0,1,2,3))
        Defect_type= st.radio('thal',(0,1,2,3))

        features =  {
            'Age' : Age,
            'Sex' : Sex,
            'Chest_pain_type' : Chest_pain_type,
            'Resting_BP': Resting_BP,
            'Cholesterol': Cholesterol,
            'Fasting_BS': Fasting_BS,
            'Rest_ECG': Rest_ECG,
            'Max_Heart_Rate': Max_Heart_Rate,
            'Ex_Induced_Ang': Ex_Induced_Ang,
            'Oldpeak': Oldpeak,
            'Slope': Slope,
            'BLD_VES_Fluros': BLD_VES_Fluros,
            'Defect_type': Defect_type
                  }

        feature = pd.DataFrame(features, index = [0])
        data = pd.read_csv('heart.csv')
        X = data.drop('Target', axis =1)
        X = np.array(X)
        y = data['Target']
        st.write('Shape of dataset:', X.shape)
        st.write('number of classes:', len(np.unique(y)))

        clf = get_classifier(classifier_name, params)
        #### CLASSIFICATION ####

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        user_pred  = clf.predict(np.array(feature)[:,:])

        st.header('PREDICTED OUTPUT')

        if user_pred == 1 :
            st.write('You have taken the correct step towards monitoring the health of your heart. Progressive care for your heart is advisable towards maintaining a better lifestyle.')
        
        elif user_pred == 0:
            st.write('Your chances of getting a heart disease is quite low. Maintain the good health. Cheers')

        #acc = accuracy_score(y_test, user_pred)

        st.write(f'Classifier = {classifier_name}')
        #st.write(f'Accuracy =', acc)
        st.write(f'Prediction = {user_pred}')

with tab5:
    
        st.header("Classification Report")
        report = pd.DataFrame(classification_report(y_test, y_pred,output_dict=True))
        st.dataframe(report)

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
        
        st.header("Confusion matrix")
        st.markdown(
        '<p style= "color: black; font-size:22px">  it is a performance measurement for machine learning classification problem where output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.',  unsafe_allow_html=True
        )
        fig=plt.figure(figsize=(1,1))
        sns.heatmap(cm,annot=True)
        st.pyplot(plt)

        st.write(f'No heart disease predicted as no heart disease= {cm[0,0]}')
        st.write(f'No heart disease predicted as yes = {cm[0,1]}')
        st.write(f'Presence of heart disease predicted as no heart disease = {cm[1,0]}')
        st.write(f'Presence of heart disease predicted as yes= {cm[1,1]}')
        
        st.write('For 0')

        st.write(f'True positive = {cm[0,0]}')
        st.write(f'False Negative = {cm[0,1]}')
        st.write(f'False positive = {cm[1,0]}')
        st.write(f'True Negative= {cm[1,1]}')

        st.write('For 1' )

        st.write(f'True positive = {cm[1,1]}')
        st.write(f'Flase Negative = {cm[1,0]}')
        st.write(f'False positive = {cm[0,1]}')
        st.write(f'True Negative = {cm[0,0]}')
        #st.write(confusion_matrix(y_test, y_pred))
       

