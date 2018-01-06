import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold, \
	SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Rectangle, Arc

# Suppress warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Set this to True to ignore leakage and get a faster code
quick_test = True

datadir = 'Data'
datafilename = os.path.join(datadir, 'data.csv')
resultsfilename = os.path.join(datadir, 'results.csv')
fullresultsfilename = os.path.join(datadir, 'results_full.csv')

# Load data
rawdata = pd.read_csv(datafilename)

# Sort chronologically and redefine the index (for leakage handling)
rawdata['date'] = pd.to_datetime(rawdata['game_date'])
rawdata.sort_values(['date', 'game_event_id'],
                    ascending = True, inplace = True)
rawdata.drop('date', axis = 1, inplace = True)
rawdata['index'] = range(len(rawdata))
rawdata.set_index('index', drop = True, inplace = True)

# Define variables definitions

# Categorical variables
cat_vars = ['action_type', 'combined_shot_type', 'period', 'season',
            'shot_type', 'shot_zone_area', 'shot_zone_basic',
            'shot_zone_range', 'opponent']

# Irrelevant variables
drop_vars = ['game_id', 'lat', 'lon', 'team_id',
             'team_name', 'matchup']

# Date variables
date_vars = ['game_date']

# Output variable
pred_var = 'shot_made_flag'

# Prediction's columns
pred_cols = ['shot_id', 'shot_made_flag']

# new variables
# home or not?
rawdata['home'] = rawdata['matchup'].apply(
		lambda x: 1 if (x.find('@') < 0) else 0)

rawdata['seconds_from_period_end'] = 60*rawdata['minutes_remaining'] +\
                                     rawdata['seconds_remaining']
rawdata['seconds_from_period_start'] = 60*(11 - rawdata['minutes_remaining']) +\
                                       (60 - rawdata['seconds_remaining'])
rawdata['seconds_from_game_start'] = (rawdata['period'] <= 4).astype(int) *\
                                     (rawdata['period']-1)*12*60 +\
                                     (rawdata['period'] > 4).astype(int) *\
                                     ((rawdata['period']-4)*5*60 + 3*12*60) +\
                                     rawdata['seconds_from_period_start']
rawdata['period_last_5_seconds'] = (rawdata['seconds_from_period_end'] < 6).\
	astype(int)
rawdata['game_last_5_seconds'] = rawdata['period_last_5_seconds'] *\
                                 (rawdata['period'] > 3).astype(int)
rawdata['angle'] = np.arctan2(rawdata['loc_x'], rawdata['loc_y'])

for var in date_vars:
	# # Replace the date string with a cyclic representation of the weekday and
	# # the time of year. The year itself is already represented by the season
	# # feature
	# datevar = pd.to_datetime(rawdata[var], format = "%Y-%m-%d")
	# weekday = 2 * np.pi * datevar.dt.dayofweek / 7
	# yearday = 2 * np.pi * datevar.dt.dayofyear / 365
	# month = 2 * np.pi * datevar.dt.month / 12
	# rawdata[var + '_weekday_x'] = np.sin(weekday)
	# rawdata[var + '_weekday_y'] = np.cos(weekday)
	# rawdata[var + '_yearday_x'] = np.sin(yearday)
	# rawdata[var + '_yearday_y'] = np.cos(yearday)
	# rawdata[var + '_month_x'] = np.sin(month)
	# rawdata[var + '_month_y'] = np.cos(month)
	
	# Convert the date variable to month, week no. and week day, and treat
	# them as categorical features
	datevar = pd.to_datetime(rawdata[var], format = "%Y-%m-%d")
	rawdata[var + '_month'] = datevar.dt.month
	rawdata[var + '_month_x'] = np.sin(2 * np.pi * rawdata[var + '_month'] / 12)
	rawdata[var + '_month_y'] = np.cos(2 * np.pi * rawdata[var + '_month'] / 12)
	cat_vars.append(var + '_month')
	# rawdata[var + '_week'] = datevar.dt.week
	# cat_vars.append(var + '_week')
	rawdata[var + '_weekday'] = datevar.dt.dayofweek
	rawdata[var + '_weekday_x'] = np.sin(
			2 * np.pi * rawdata[var + '_weekday'] / 7)
	rawdata[var + '_weekday_y'] = np.cos(
		2 * np.pi * rawdata[var + '_weekday'] / 7)
	cat_vars.append(var + '_weekday')

for var in cat_vars:
	# Replace the categorical feature with N binary features
	cat_list = pd.get_dummies(rawdata[var], prefix = var, prefix_sep = ': ')
	data1 = rawdata.join(cat_list)  # adds data columns to data
	rawdata = data1
	
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
test_rows = pd.isnull(rawdata[pred_var])    # gives indices of rows to predict
traindata = rawdata[~test_rows]
evaldata = rawdata[test_rows]

# Prepare the training data
X0 = traindata[X_cols]
Y0 = np.ravel(traindata[Y_cols])

# Plot RFE rankings
lr = LogisticRegression()
rfe = RFE(lr, 1)
rfe = rfe.fit(X0, Y0)
feature_df = pd.DataFrame(rfe.ranking_, index = X_cols, columns = ["ranking"])
feature_df.sort_values("ranking", inplace = True)
plt.figure()
x_axis = np.arange(len(feature_df))
plt.bar(x_axis, feature_df["ranking"])
plt.xticks(x_axis, feature_df.index, rotation = 'vertical')
plt.show()

# Choose a subset of features by recursive features elimination
lr = LogisticRegression()
n_features = 10
rfe = RFE(lr, n_features)  # chooses the best N features from which to do LR
rfe.fit(X0, Y0)
X_cols_rfe = [X_cols[i] for i in range(len(X_cols)) if rfe.get_support()[i]]
print('RFE chosen features: ', X_cols_rfe)
print(len(X_cols_rfe))
# X_cols_rfe = []

# # Choose features by cross-validation based recursive features elimination
# lr = LogisticRegression()
# rfecv = RFECV(estimator = lr, cv = 5, scoring = 'neg_log_loss')
# rfecv.fit(X0, Y0)
# X_cols_rfecv = [X_cols[i] for i in range(len(X_cols))
#                 if rfecv.get_support()[i]]
# print('RFECV chosen features: ', X_cols_rfecv)
# print(len(X_cols_rfecv))
X_cols_rfecv = []

# # Include the first n features used by a decision tree classifier
# dtc_mid = 0.05
# n_features = 10     # max(len(X_cols_rfe), len(X_cols_rfecv))
# dtc = DecisionTreeClassifier(min_impurity_decrease = dtc_mid)
# dtc.fit(X0, Y0)
# # M = np.mean(clf.feature_importances_)
# # X_cols_dtc = [X_cols[i] for i in range(len(X_cols))
# #               if clf.feature_importances_[i] >= M]
# feature_df = pd.DataFrame(dtc.feature_importances_,
#                           index = X_cols,
#                           columns = ["importance"])
# X_cols_dtc = feature_df.sort_values("importance", ascending = False).head(
# 		n_features).index
# print('DTC chosen features: ', X_cols_dtc)
# print(len(X_cols_dtc))
X_cols_dtc = []

# # Choose features with high mutual information with the predicted variable
# n_features = max(len(X_cols_rfe), len(X_cols_rfecv))
# skb = SelectKBest(score_func = mutual_info_classif, k = n_features)
# skb.fit(X0, Y0)
# X_cols_skb = [X_cols[i] for i in range(len(X_cols)) if skb.get_support()[i]]
# print('SKB chosen features: ', X_cols_skb)
# print(len(X_cols_skb))
X_cols_skb = []

# # Setting Variance Threshold
# # Choose all features with high enough variance
# p = 0.9
# th = p*(1 - p)  # Assumed binary variable (still good for other features)
# vt = VarianceThreshold()
# vt.fit(X0)
# X_cols_vt = [X_cols[i] for i in range(len(X_cols)) if vt.variances_[i] > th]
# print('VT chosen features: ', X_cols_vt)
# print(len(X_cols_vt))
X_cols_vt = []

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X0, Y0)
X0_lda = lda.transform(X0)

# Decision tree regressor
dtr_mid = 0.0005
dtr = DecisionTreeRegressor(min_impurity_decrease = dtr_mid)
dtr.fit(X0, Y0)
X0_dtr = dtr.predict(X0)

# Plot Regression Tree features' importances
feature_df = pd.DataFrame(dtr.feature_importances_,
                          index = X_cols,
                          columns = ["importance"])
feature_df.sort_values("importance", ascending = False, inplace = True)
plt.figure()
x_axis = np.arange(len(feature_df))
plt.bar(x_axis, feature_df["importance"])
plt.xticks(x_axis, feature_df.index, rotation = 'vertical')
plt.show()

# Filter columns
# unique is just to not calculate one param multiple times
X_cols_f = np.unique(np.hstack([X_cols_rfe, X_cols_rfecv,
                                X_cols_dtc, X_cols_skb, X_cols_vt]))
print('Final chosen features: ', X_cols_f)
print(len(X_cols_f))
X0 = X0[X_cols_f]
X0.loc[:, 'LDA'] = X0_lda		# adding value of single LDA dim
X0.loc[:, 'DTR'] = X0_dtr       # adding value of DTR prediction

# Test the model
lr = LogisticRegression()
scores = -cross_val_score(lr, X0, Y0, cv = 10, scoring = 'neg_log_loss')
print('CV log-loss: ', np.mean(scores), '+/-', np.std(scores))

# For test only - Leakage problem
if quick_test: 	# (with leakage, quick and dirty)
	
	# Fit the model
	lr = LogisticRegression()
	lr.fit(X0, Y0)  # trains on training data. Here the flesh lies.
	
	# Generate the model's prediction
	Xlda = lda.transform(evaldata[X_cols])
	Xdtr = dtr.predict(evaldata[X_cols])
	X = evaldata[X_cols_f]
	X.loc[:, 'LDA'] = Xlda
	X.loc[:, 'DTR'] = Xdtr
	Y = lr.predict_proba(X)  # without _proba , this would have given 0/1
	evaldata.loc[:, pred_var] = Y[:, 1]

# Leakage handle
else:		# without leakage
	
	c = 0
	C = len(evaldata)
	for t in evaldata.index:
		
		# Filter only past data
		ind = traindata.index
		ind_t = ind[ind < t]
		if len(ind_t) == 0:		# if row picked is first
			evaldata.loc[t, pred_var] = 0.5     # For the first shot, just guess
			continue
		Xt = traindata.loc[ind_t, X_cols_f]
		Yt = np.ravel(traindata.loc[ind_t, Y_cols])
		
		# Fit the model
		lda = LinearDiscriminantAnalysis()
		lda.fit(traindata.loc[ind_t, X_cols], Yt)
		Xt.loc[:, 'LDA'] = lda.transform(traindata.loc[ind_t, X_cols])
		dtr = DecisionTreeRegressor(min_impurity_decrease = dtr_mid)
		dtr.fit(traindata.loc[ind_t, X_cols], Yt)
		Xt.loc[:, 'DTR'] = dtr.predict(traindata.loc[ind_t, X_cols])
		lr = LogisticRegression()
		lr.fit(Xt, Yt)  # trains on training data. Here the flesh lies.

		# Generate the model's prediction
		X = evaldata.loc[[t], X_cols_f]
		# using the LDA axis found, generate and add "LDA value"
		X.loc[:, 'LDA'] = lda.transform(evaldata.loc[[t], X_cols])
		X.loc[:, 'DTR'] = dtr.predict(evaldata.loc[[t], X_cols])
		Y = lr.predict_proba(X)     # without _proba , this would have given 0/1
		evaldata.loc[t, pred_var] = Y[0, 1]
		
		# Display progress
		c = c + 1
		print(t, ': ', c, '/', C)

# For future plots
evaldata.to_csv(fullresultsfilename, header = True, index = False)

# what we hand-in as output
preddata = evaldata[pred_cols]
preddata.to_csv(resultsfilename, header = True, index = False)
