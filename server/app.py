import os
from flask import Flask,jsonify,request
import numpy as np
import pandas as pd
import simplejson as json
import time
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

rows = 50000
modelnumber = 7

SITE_ROOT = os.path.realpath(os.path.dirname(__file__))

# Set up the app and point it to Vue
app = Flask(__name__, static_folder='../client/dist/', static_url_path='/')

@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route("/api/datasetlist")
def get_datasetlist():
    path = SITE_ROOT+'/db/'
    dir_list = os.listdir(path)
    return jsonify(dir_list)


@app.route("/api/<dataset>/featurevalue/<featurename>")
def get_featurevalue(dataset,featurename):
    types = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/featuretypes.csv')
    df = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/features.csv',nrows=rows)
    featuretype = (types.loc[types['featurename']==featurename].iloc[0])['type']
    if featuretype=='class':
        featuretype = 'class'
    elif (featuretype == 'float')| (featuretype=='int'):
        featuretype = 'numerical'
    else:
        featuretype = 'categorical'
    result = {'name':featurename,'type':featuretype,'value':df[featurename].tolist()}
    return jsonify(result)

'''
@description: 
@param {*} types
@param {*} df: the subset dataset
@param {*} df2: the whole dataset
@return {*}
'''
def statisticdistribution(types,df,df2):
    start_time = time.time()
    statistics = []
    classfeature = types.loc[types['type']=='class']['featurename'].iloc[0] # find out which feature is used for classification
    for index, feature in types.iterrows():
        featureobj = {"feature":feature['featurename']}
        featurelist= []
        if feature['type']=='key': # this feature is key attribute
            continue
        elif feature['type'] in ['int','float']:
            minvalue = np.floor(df2[feature['featurename']].min())
            maxvalue = np.ceil(df2[feature['featurename']].max())
            grouped = df.groupby(classfeature)
            for name,group in grouped:
                interval = 20
                if(feature['type']=='int'):
                    interval=int(maxvalue-minvalue)
                    if interval>20:
                        interval=20
                if interval <=1:
                    interval =1 
                hist,bin_edges = np.histogram(group[feature['featurename']],range=(minvalue,maxvalue),bins=interval)
                density_hist,density_bin_edges = np.histogram(group[feature['featurename']],range=(minvalue,maxvalue),bins=interval,density=True)
                featurelist.append({'class':name,"hist":hist.tolist(),"bin_edges":bin_edges.tolist(),'density_hist':density_hist.tolist()})
            featureobj['type']="numerical"
            featureobj['ftype']=feature['type']
        elif feature['type'] in ['binary','categorical']:
            result = df.value_counts(subset=[feature['featurename'],classfeature])
            tmpobj = {}
            tmpcount={}
            for index,count in result.items():
                if index[1] in tmpcount:
                    tmpcount[index[1]]+=count
                else:
                    tmpcount[index[1]]=count
                if index[0] in tmpobj:
                    tmpobj[index[0]].append({'class':index[1],'value':count})
                else:
                    tmpobj[index[0]]=[{'class':index[1],'value':count}]
            for key, value in tmpobj.items():
                for v in value:
                    v['density'] = v['value']/tmpcount[v['class']]
                temp = {'category':key,'values':value}
                featurelist.append(temp)
            featureobj['type']="categorical"
            featureobj['ftype']=feature['type']
        else: # class
            result = df.value_counts(subset=[classfeature])
            for index,count in result.items():
                featurelist.append({'category':index[0],'value':count})
            featureobj['type']="categorical"
            featureobj['ftype']=feature['type']

        featureobj['data']=featurelist
        statistics.append(featureobj)

    print("--- %s seconds ---" % (time.time() - start_time))
    return statistics


@app.route("/api/<dataset>/featureinfos",methods=['GET'])
def get_featureinfo(dataset):
    start_time = time.time()
    types = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/featuretypes.csv')
    classfeature = types.loc[types['type']=='class']['featurename'].iloc[0] # find out which feature is used for classification


    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    df = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/features.csv',nrows=rows)
    
    # df = pd.read_sql_query("select * from "+str(dataset)+"_features",conn) # too slow than read csv
    print("--- %s seconds ---" % (time.time() - start_time))

    classes = df[classfeature].unique()

    datainfo={
        'dataamount':df.shape[0],
        'category':classfeature,
        'classfeature':{'name':classfeature,'type':'class','value':df[classfeature].tolist()},
        'featureamount':types.shape[0]
    }
    
    return jsonify({'datainfo':datainfo,'classes':classes.astype(str).tolist(),'featureinfo':statisticdistribution(types,df,df)})

@app.route("/api/<dataset>/featureinfo/subset",methods=['POST'])
def get_featureinfo_subset(dataset):
    # get instance ids for selected subset
    request_json = request.get_json()
    subset = request_json.get('subset')

    start_time = time.time()
    types = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/featuretypes.csv')

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    df = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/features.csv',nrows=rows)
    df_subset = df.iloc[subset,:]
    # df = pd.read_sql_query("select * from "+str(dataset)+"_features",conn) # too slow than read csv
    print("--- %s seconds ---" % (time.time() - start_time))

    return jsonify(statisticdistribution(types,df_subset,df))

@app.route("/api/<dataset>/calFeatureProjection/<method>",methods=['GET'])
def calprojection(dataset,method):
    showtype = "projection"
    mypath = SITE_ROOT+'/db/'+dataset+'/featureproj/'+method+'.json'
    if os.path.exists(mypath):
        with open(mypath, "r") as outfile:
            result = json.load(outfile)
    else:
        result= []

    return jsonify({'type':showtype,'data':result})

@app.route("/api/<dataset>/shapproj/<modelname>")
def get_shapproj(dataset,modelname):
    with open(SITE_ROOT+'/db/'+dataset+'/shapproj/'+modelname+'.json', "r") as outfile:
        result = json.load(outfile)

    return jsonify(result)



@app.route("/api/<dataset>/predictions")
def get_predictions(dataset):
    df = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/predictions.csv',nrows=rows)
    result={}
    for column in df:
        result[column]=df[column].tolist()

    return jsonify(result)

def calmodelsinfos(dataset,modelnames):
    df = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/predictions.csv',usecols=modelnames+['key','label'])
    predictions=[]
    predictions.append({'name':'label','value':df['label'].tolist()})
    statistics = []

    with open(SITE_ROOT+'/db/'+dataset+'/models.json') as f:
        models = json.load(f)

        featureuse = []
        shapdata=[]

        models.sort(key=lambda x: x['accuracy'], reverse=True)

        for model in models:
            if(model['model'] not in modelnames):
                continue

            # featureuse
            featureuse.append({'name':model['model'],'features':model['features']})

            # predictions
            acc = accuracy_score(df['label'],df[model['model']])
            precision = precision_score(df['label'],df[model['model']],average='weighted')
            recall = recall_score(df['label'],df[model['model']],average='weighted')
            f1 = f1_score(df['label'],df[model['model']],average='weighted')
            predictions.append({'name':model['model'],'accuracy':acc,'value':df[model['model']].tolist()})
            statistics.append({'name':model['model'],'accuracy':acc,'precision':precision,'recall':recall,'f1':f1})

            # shap values
            shapvalues = []
            # print(np.amax(np.array(list(model['shap'].values()))))
            for classname in model['classes']:
                tmp = []
                for x in zip(*(model['shap'][classname])):
                    q1 = np.percentile(x,25)
                    q3 = np.percentile(x,75)
                    median = np.percentile(x,50)
                    std = np.std(x)
                    # interQuantileRange = q3 - q1
                    minv = np.min(x)
                    maxv = np.max(x)
                    iqr=q3-q1
                    r0 = max(minv, q1 - iqr * 1.5)
                    r1 = min(maxv, q3 + iqr * 1.5)
                    # print(x,minv,maxv)
                    if(round(minv,3)==round(maxv,3)):
                        minv=minv-0.001
                        maxv=maxv+0.001
                    density_hist,density_bin_edges = np.histogram(x,density=True,range=(minv,maxv))
                    tmp.append({'class':classname,'q1':q1,'q3':q3,'median':median,'r0':r0, 'r1':r1, 'sum':getabsavg(x),'avg':np.average(x),'std':std, 'min': minv, 'max':maxv,"bin_edges":density_bin_edges.tolist(),'density_hist':density_hist.tolist()})
                shapvalues.append(tmp)
            
            modelshap = []
            for i, feature in enumerate(model['features']):
                shap = []
                for shapvalue in shapvalues:
                    shap.append(shapvalue[i])
                modelshap.append({'feature':feature,'shap':shap})

            shapdata.append({'model':model['model'],'values':modelshap})

    return {'featureuse':featureuse,'predictions':predictions,'shapvalues':shapdata, 'dataids':df['key'].tolist(),'statistics':statistics}

def getabsavg(x):
    possum=0
    posnum=0
    negsum=0
    negnum=0
    for i in x:
        if i>0:
            possum=possum+i
            posnum=posnum+1
        elif i<0:
            negsum=negsum+i
            negnum=negnum+1
    return [possum/posnum if posnum>0 else 0,-negsum/negnum if negnum>0 else 0]


@app.route("/api/<dataset>/selectedmodelinfos",methods=['POST'])
def calselectedmodelinfos(dataset):
    request_json = request.get_json()
    models = request_json.get('models')

    result = calmodelsinfos(dataset,models)

    return jsonify({'featureuse':result['featureuse'],'predictions':result['predictions'],'shapvalues':result['shapvalues'], 'statistics':result['statistics']})


@app.route("/api/<dataset>/modelinfos")
def get_modelinfos(dataset):
    with open(SITE_ROOT+'/db/'+dataset+'/models.json') as f:
        models = json.load(f)
        models.sort(key=lambda x: x['accuracy'], reverse=True)
        modelinfos = [{'name':model['model'],'accuracy':model['accuracy'],'precision':model['precision'] if 'precision' in model else '*','f1':model['f1'] if 'f1' in model else '*'} for model in models]
        modelnames = [model['model'] for model in models]
        if(len(modelnames)>7):
            selectedmodelnames=modelnames[:7]
        else:
            selectedmodelnames=modelnames
    
    result = calmodelsinfos(dataset,selectedmodelnames)

    return jsonify({'modelnames':modelnames,'modelinfos':modelinfos,'selectedmodelnames':selectedmodelnames,'featureuse':result['featureuse'],'predictions':result['predictions'],'shapvalues':result['shapvalues'], 'dataids':result['dataids'],'statistics':result['statistics']})

@app.route("/api/<dataset>/statistics/subset",methods=['POST'])
def get_modelstatistics_subset(dataset):
    request_json = request.get_json()
    subset = request_json.get('subset')
    modelnames = request_json.get('models')

    df = pd.read_csv(SITE_ROOT+'/db/'+dataset+'/predictions.csv',usecols=modelnames+['key','label']).iloc[subset]
    statistics = []

    for modelname in modelnames:
        labels = np.unique(df['label']+df[modelname])
        acc = accuracy_score(df['label'],df[modelname])
        precision = precision_score(df['label'],df[modelname],average='weighted')
        recall = recall_score(df['label'],df[modelname],average='weighted')
        f1 = f1_score(df['label'],df[modelname],average='weighted')

        statistics.append({'name':modelname,'accuracy':acc,'precision':precision,'recall':recall,'f1':f1})

    return jsonify({'statistics':statistics})


@app.route("/api/<dataset>/shap/subset",methods=['POST'])
def get_shapvalues_subset(dataset):
    request_json = request.get_json()
    subset = request_json.get('subset')
    modelnames = request_json.get('models')

    if len(subset)==0:
        return jsonify([])

    with open(SITE_ROOT+'/db/'+dataset+'/models.json') as f:
        models = json.load(f)

        shapdata=[]

        models.sort(key=lambda x: x['accuracy'], reverse=True)

        for model in models:
            if(model['model'] not in modelnames):
                continue
            # shap values
            shapvalues = []
            for classname in model['classes']:
                tmp = []
                for x in zip(*(np.array(model['shap'][classname])[subset,:])):
                    q1 = np.percentile(x,25)
                    q3 = np.percentile(x,75)
                    median = np.percentile(x,50)
                    # interQuantileRange = q3 - q1
                    std = np.std(x)
                    minv = np.min(x)
                    maxv = np.max(x)
                    iqr=q3-q1
                    r0 = max(minv, q1 - iqr * 1.5)
                    r1 = min(maxv, q3 + iqr * 1.5)
                    if(round(minv,3)==round(maxv,3)):
                        minv=minv-0.001
                        maxv=maxv+0.001
                    density_hist,density_bin_edges = np.histogram(x,density=True,range=(minv,maxv))
                    tmp.append({'class':classname,'q1':q1,'q3':q3,'median':median,'r0':r0, 'r1':r1, 'sum':getabsavg(x),'avg':np.average(x),'std':std, 'min': minv, 'max':maxv,"bin_edges":density_bin_edges.tolist(),'density_hist':density_hist.tolist()})
                shapvalues.append(tmp)
            
            modelshap = []
            for i, feature in enumerate(model['features']):
                shap = []
                for shapvalue in shapvalues:
                    shap.append(shapvalue[i])
                modelshap.append({'feature':feature,'shap':shap})

            shapdata.append({'model':model['model'],'values':modelshap})

        return jsonify(shapdata)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(port=port)