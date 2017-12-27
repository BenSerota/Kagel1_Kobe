import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Suppress warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Set this to True to ignore leakage and get a faster code
quick_test = False

datadir = 'Data'
datafilename = os.path.join(datadir, 'data.csv')
resultsfilename = os.path.join(datadir, 'results.csv')

# Load data
rawdata = pd.read_csv(datafilename)

# Define variables definitions

# Categorical variables
cat_vars = ['action_type', 'season', 'shot_type', 'shot_zone_area',
            'shot_zone_basic', 'shot_zone_range', 'opponent']

# Irrelevant variables
drop_vars = ['combined_shot_type', 'game_id', 'lat', 'lon', 'team_id',
             'team_name', 'matchup']

# Date variables
date_vars = ['game_date']

# Output variable
pred_var = 'shot_made_flag'

# Prediction's columns
pred_cols = ['shot_id', 'shot_made_flag']

# new variables
# home or not?
rawdata['home'] = rawdata['matchup'].apply(lambda x: 1 if (x.find('@') < 0) else 0)

rawdata['secondsFromPeriodEnd'] = 60*rawdata['minutes_remaining'] + \
                                  rawdata['seconds_remaining']
rawdata['secondsFromPeriodStart'] = 60*(11 - rawdata['minutes_remaining']) + \
                                    (60 - rawdata['seconds_remaining'])
rawdata['secondsFromGameStart'] = (rawdata['period'] <= 4).astype(int)*\
                                  (rawdata['period']-1)*12*60 + \
                                  (rawdata['period'] > 4).astype(int)*\
                                  ((rawdata['period']-4)*5*60 + 3*12*60) + \
                                  rawdata['secondsFromPeriodStart']
rawdata['angle'] = np.arctan2(rawdata['loc_x'], rawdata['loc_y'])


for var in cat_vars:
	# Replace the categorical feature with N binary features
	cat_list = pd.get_dummies(rawdata[var], prefix = var)
	data1 = rawdata.join(cat_list) 					# adds data columns to data
	rawdata = data1

for var in date_vars:
	# Replace the date string with a cyclic representation of the weekday and
	# the time of year. The year itself is already represented by the season
	# feature
	datevar = pd.to_datetime(rawdata[var], format = "%Y-%m-%d")
	weekday = 2 * np.pi * datevar.dt.dayofweek / 7
	yearday = 2 * np.pi * datevar.dt.dayofyear / 365
	month = 2 * np.pi * datevar.dt.month / 12
	rawdata[var + '_weekday_x'] = np.sin(weekday)
	rawdata[var + '_weekday_y'] = np.cos(weekday)
	rawdata[var + '_yearday_x'] = np.sin(yearday)
	rawdata[var + '_yearday_y'] = np.cos(yearday)
	rawdata[var + '_month_x'] = np.sin(month)
	rawdata[var + '_month_y'] = np.cos(month)

data_vars = rawdata.columns.values.tolist()
to_keep = [i for i in data_vars if (i not in cat_vars and
                                    i not in drop_vars and
                                    i not in date_vars)]
rawdata = rawdata[to_keep]  # generating a new dataset, overriding old one.

X_cols = [i for i in rawdata.columns if i not in pred_var]  # predict by cols
Y_cols = [pred_var]

# # PCA
# tmpdata = rawdata[X_cols]
# for var in X_cols:
# 	x = tmpdata[var]
# 	variance = np.var(x)
# 	if variance == 0:
# 		tmpdata.drop(var, axis = 1, inplace = True)
# 		X_cols.remove(var)
# 	else:
# 		tmpdata[var] = (x - np.mean(x))/variance	# normalizing
# components = 20
# pca_cols = ['PCA_' + str(i) for i in range(components)]
# pca = PCA(n_components = components).fit(tmpdata)
# pcadata = pd.DataFrame(data = pca.transform(tmpdata),
#                        index = tmpdata.index,
#                        columns = pca_cols)
# X_cols = X_cols + pca_cols
# rawdata = rawdata.join(pcadata)

# Split data
test_rows = pd.isnull(rawdata[pred_var])			# gives the indeces of all the rows to predict
traindata = rawdata[~test_rows]
evaldata = rawdata[test_rows]
preddata = evaldata[pred_cols]						# what we hand-in as output

# Prepare the training data
X0 = traindata[X_cols]
Y0 = np.ravel(traindata[Y_cols])

# Set the model
lr = LogisticRegression()

# Choose a subset of features by recursive features elimination
n_features = 20
rfe = RFE(lr, n_features) 	# chooses the best N features from which to do LR
rfe = rfe.fit(X0, Y0)
X_cols_rfe = [X_cols[i] for i in range(len(X_cols)) if rfe.support_[i]]
print('RFE chosen features: ', X_cols_rfe)
print(len(X_cols_rfe))
plt.bar(np.arange(len(rfe.ranking_)), rfe.ranking_)
plt.xticks(X_cols)

# # Choose features by cross-validation based recursive features elimination
# DECIDED NOT TO INCLUDE RFE, AS PREDICTION WAS BETTER WITH 20 FEATURES

# rfecv = RFECV(estimator = lr, cv = 3, scoring = 'neg_log_loss')
# rfecv.fit(X0, Y0)
# X_cols_rfecv = [X_cols[i] for i in range(len(X_cols)) if rfecv.support_[i]]
# print('RFECV chosen features: ', X_cols_rfecv)
# print(len(X_cols_rfecv))
X_cols_rfecv = []

# Setting Variance Threshold
# Choose all features with high enough variance
p = 0.9
th = p*(1 - p)  # Assumed binary variable (still good for other features)
vt = VarianceThreshold()
vt = vt.fit(X0)
X_cols_vt = [X_cols[i] for i in range(len(X_cols)) if vt.variances_[i] > th]
print('VT chosen features: ', X_cols_vt)
print(len(X_cols_vt))

# LDA
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X0, Y0)
X0_lda = lda.transform(X0)

# Filter columns
# unique is just for to not calculate one param multiple times
X_cols_f = np.unique(np.hstack([X_cols_rfe, X_cols_rfecv, X_cols_vt]))
print('Final chosen features: ', X_cols_f)
print(len(X_cols_f))
X0 = X0[X_cols_f]
X0.loc[:, 'LDA'] = X0_lda		# adding value of single LDA dim

# Test the model
scores = -cross_val_score(lr, X0, Y0, cv = 10, scoring = 'neg_log_loss')
print('CV log-loss: ', np.mean(scores), '+/-', np.std(scores))

# For test only - Leakage problem
if quick_test: 	#(with leakage = quick and dirty)
	
	# Fit the model
	lr = LogisticRegression()
	lr = lr.fit(X0, Y0)  # trains on training data. Here the flesh lies.
	
	# Generate the model's prediction
	Xlda = lda.transform(evaldata[X_cols])
	X = evaldata[X_cols_f]
	X.loc[:, 'LDA'] = Xlda
	Y = lr.predict_proba(X)  # without _proba , this would have given 0/1
	preddata.loc[:, pred_var] = Y[:, 1]


# Leakage handle
else:		#without leakage
	
	c = 0
	C = len(evaldata)
	for t in evaldata.index:
		
		# Filter only past data
		ind = traindata.index
		ind_t = ind[ind < t]
		if len(ind_t) == 0:		# if row picked is first
			preddata.loc[t, pred_var] = 0.5     # For the first shot, just guess
			continue
		Xt = traindata.loc[ind_t, X_cols_f]
		Yt = np.ravel(traindata.loc[ind_t, Y_cols])
		
		# Fit the model
		lda = LinearDiscriminantAnalysis()
		lda = lda.fit(traindata.loc[ind_t, X_cols], Yt)
		Xt.loc[:, 'LDA'] = lda.transform(traindata.loc[ind_t, X_cols])
		lr = LogisticRegression()
		lr = lr.fit(Xt, Yt)  # trains on training data. Here the flesh lies.
		
		# Generate the model's prediction
		X = evaldata.loc[[t], X_cols_f]
		# using the LDA axis found, generate and add "LDA value"
		X.loc[:, 'LDA'] = lda.transform(evaldata.loc[[t], X_cols])
		Y = lr.predict_proba(X)     # without _proba , this would have given 0/1
		preddata.loc[t, pred_var] = Y[0, 1]
		
		# Display progress
		c = c + 1
		print(t, ': ', c, '/', C)
		
preddata.to_csv(resultsfilename, header = True, index = False)


# plotting:
