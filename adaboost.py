# import libraries
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np

# set page title
st.set_page_config(page_title="Streamlit App")

st.title("ADABOOST")
st.text("")

# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2-D feature vector
(X, y) = make_blobs(n_samples=2000, n_features=2, centers=2, cluster_std=2, random_state=1)

fig, ax = plt.subplots()

#col1,col2 = st.columns(2)
col1, padding, col2 = st.columns((10,2,10))
#col1, col2 = st.columns(2, gap = "large")

with col1:
    st.subheader("Dataset")
    ax.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=30)
    #ax.set_title("Data")
    st.pyplot(fig)

depth = st.sidebar.slider("Depth of a Decision Tree", min_value=1,max_value=5,step=1)
n_est = st.sidebar.slider("Number of estimators", min_value=1, max_value=500, step=1)
lr = st.sidebar.slider("Learning Rate", min_value=0.1,max_value=5.0,step=0.1)

# create and fit AdaBoost model
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),n_estimators=n_est, learning_rate=lr, random_state=42)
ada.fit(X, y)

# make predictions and calculate accuracy
y_pred = ada.predict(X)
acc = accuracy_score(y, y_pred)


# create a mesh to plot decision boundaries
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

with col2:
# plot the (testing) classification data with decision boundaries
    st.subheader("Decision boundaries")
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=30, cmap=plt.cm.RdYlBu)
    #ax.set_title("Data with Decision Boundaries")
    st.pyplot(fig)

# print accuracy
#st.write(f"Accuracy: {acc}")
#print(f"Accuracy: {acc}")



st.text("")
st.text("")
st.subheader("Performance")

accuracy = metrics.accuracy_score(y, y_pred)
error = metrics.mean_squared_error(y, y_pred)
#st.write("Accuracy:  ",metrics.accuracy_score(y, y_pred))
st.success("Accuracy :  {:.2f}".format(accuracy))
#st.write("MSE",metrics.mean_squared_error(y, y_pred))
st.success("MSE      :  {:.2f}".format(error))

