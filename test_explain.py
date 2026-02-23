import pickle, numpy as np, pandas as pd, io, base64, random, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, shap, lime, lime.lime_tabular

model=pickle.load(open('knn_model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
feature_names=pickle.load(open('features.pkl','rb'))
X_train_scaled=pickle.load(open('X_train_scaled.pkl','rb'))
X_train_summary=pickle.load(open('X_train_summary.pkl','rb'))
X_train_np=X_train_scaled if isinstance(X_train_scaled,np.ndarray) else X_train_scaled.values

shap_exp=shap.KernelExplainer(model.predict, X_train_summary)
lime_exp=lime.lime_tabular.LimeTabularExplainer(X_train_np,feature_names=feature_names,
    class_names=['No CHD','CHD'],mode='classification',random_state=42)

edu={'education_1.0':0,'education_2.0':1,'education_3.0':0,'education_4.0':0}
raw={'age':55,'sex':1,'cigsPerDay':10,'BPMeds':0,'prevalentStroke':0,'prevalentHyp':1,
     'diabetes':0,'totChol':250,'BMI':28,'heartRate':80,'glucose':90,'pulse_pressure':60,**edu}
df=pd.DataFrame([raw])[feature_names]
scaled=scaler.transform(df)

print('Step 1: SHAP values...')
np.random.seed(42)
sv=shap_exp.shap_values(scaled,nsamples=100)
print('  shap values type:', type(sv), 'value[0]:', sv[0][:3])

print('Step 2: Force plot...')
shap.force_plot(shap_exp.expected_value,sv[0],df,matplotlib=True,show=False)
b=io.BytesIO(); plt.savefig(b,format='png',bbox_inches='tight',dpi=100,facecolor='#0f1623')
plt.close('all'); b.seek(0); f1=base64.b64encode(b.read()).decode()
print('  force plot bytes:', len(f1))

print('Step 3: Bar plot...')
shap.bar_plot(sv[0],feature_names=feature_names,max_display=10,show=False)
b2=io.BytesIO(); plt.savefig(b2,format='png',bbox_inches='tight',dpi=100,facecolor='#0f1623')
plt.close('all'); b2.seek(0); f2=base64.b64encode(b2.read()).decode()
print('  bar plot bytes:', len(f2))

print('Step 4: LIME...')
np.random.seed(42)
exp=lime_exp.explain_instance(scaled[0],model.predict_proba,num_features=10,num_samples=3000)
fig=exp.as_pyplot_figure()
b3=io.BytesIO(); fig.savefig(b3,format='png',bbox_inches='tight',dpi=100,facecolor='#0f1623')
plt.close('all'); b3.seek(0); f3=base64.b64encode(b3.read()).decode()
print('  lime bytes:', len(f3))
print('ALL PASSED!')
