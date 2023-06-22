# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:28:16 2021

@author: Aicha-PC
"""



import csv
import pandas
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import chart_studio.plotly
#import plotly.plotly as py
#import plotly.tools as tls
#import plotly.figure_factory as ff
#import plotly.graph_objs as go
import addfips
import numpy 
import numpy as np
import pandas
import pandas as pd
import plotly.figure_factory as ff
import plotly
import plotly.graph_objs as go
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.panel import PooledOLS
import statsmodels
import statsmodels.api as sm
from linearmodels.panel import compare
import seaborn 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas import DataFrame
import statsmodels.formula.api as smf
import regex as re
import spacy
import csv
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
#from statsmodels.iolib.summary3 import summary_col
#from summary3 import summary_col 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available

#%% 
import pandas as pd 
popsharetime = pd.read_csv("agepopulationsharevstime.csv", delimiter=';', encoding='cp1252') 
popsharetime.head()
#%%
colors = ["#A60842", "#181A40","#061256"]
style.use('seaborn-white')
under_20_graph = popsharetime.plot(x = 'agegroup', y = ["2009","2020"], figsize = (12,8),color = colors, linewidth=5,  linestyle='-')
ax = plt.gca()
ax.set_ylabel("Share %",fontsize=26)
ax.set_xlabel("Age-group",fontsize=26)
ax.tick_params(axis='x', labelsize=26)
ax.tick_params(axis='y', labelsize=26)
ax.yaxis.grid()
ax.legend()
plt.legend(fontsize=26)         
print('Type:', type(under_20_graph))
plt.savefig('shareagegroupovertime.png', bbox_inches='tight')

#%% load data for macro regression 

import pandas as pd 
macroregdata = pd.read_csv("Book1panelforestimation.csv", delimiter=',', encoding='cp1252') 
macroregdata.head()

#%% THIS IF WE IGNORE HOURS LM AND LY IN THE REGRESSION MODEL.

emacreg1 = macroregdata[macroregdata['HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN'].notna()]

#%%

emacreg1['mid_age'] = emacreg1['de_25_49'] +emacreg1['de_50_64']
#%%
examplereemacreg1 = macroregdata[macroregdata['HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN'].notna()]

#%%
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
for c in [c for c in emacreg1.columns if emacreg1[c].dtype in numerics]:
    emacreg1[c] = np.log(emacreg1[c])


#     emacreg1[c] = np.log10(emacreg1[c])


#%%


examplereemacreg1DK = emacreg1.loc[emacreg1['country'] == 'DK']
examplereemacreg1IT = emacreg1.loc[emacreg1['country'] == 'IT']
examplereemacreg1OE = emacreg1.loc[emacreg1['country'] == 'OE']
examplereemacreg1ES = emacreg1.loc[emacreg1['country'] == 'ES']
examplereemacreg1NL = emacreg1.loc[emacreg1['country'] == 'NL']
examplereemacreg1SD = emacreg1.loc[emacreg1['country'] == 'SD']
examplereemacreg1SW = emacreg1.loc[emacreg1['country'] == 'SW']
examplereemacreg1BG = emacreg1.loc[emacreg1['country'] == 'BG']
examplereemacreg1FN = emacreg1.loc[emacreg1['country'] == 'FN']
examplereemacreg1BD = emacreg1.loc[emacreg1['country'] == 'BD']
examplereemacreg1FR = emacreg1.loc[emacreg1['country'] == 'FR']



#%% emacreg1r.savingrate
colors = ["#A60842", "#181A40","#061256"]
style.use('seaborn-white')
#under_20_graph = examplereemacreg1DK.plot(x = 'time', y = ["savingrate","r"], figsize = (12,8),color = colors, linewidth=5,  linestyle='-')
#ax = plt.gca()
#ax.set_ylabel(" %",fontsize=26)
#ax.set_xlabel("time",fontsize=26)
#ax.tick_params(axis='x', labelsize=26)
#ax.tick_params(axis='y', labelsize=26)
#ax.yaxis.grid()
#ax.legend()
#plt.legend(fontsize=26)         
#print('Type:', type(under_20_graph))
#plt.savefig('sAVINGRATEINTERETRATEtime.png', bbox_inches='tight')




#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(ncols=2)
sns.regplot(x='r', y='savingrate', data=examplereemacreg1, ax=axs[0])
#sns.regplot(x='r', y='savingrate', data=examplereemacreg1, ax=axs[1])
sns.boxplot(x='country',y='savingrate', data=examplereemacreg1, ax=axs[1])

#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(ncols=3)
sns.regplot(x='r', y='savingrate', data=examplereemacreg1, ax=axs[0])
sns.regplot(x='returnonequity', y='savingrate', data=examplereemacreg1, ax=axs[1])
sns.boxplot(x='country',y='savingrate', data=examplereemacreg1, ax=axs[2])
#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(ncols=4)
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1DK, ax=axs[0])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1IT, ax=axs[1])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1OE, ax=axs[2])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1ES, ax=axs[3])
#%%
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(ncols=4)
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1NL, ax=axs[0])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1SD, ax=axs[1])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1SW, ax=axs[2])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1BG, ax=axs[3])
#%%
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(ncols=3)
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1FN, ax=axs[0])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1BD, ax=axs[1])
sns.regplot(x='r', y='HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN', data=examplereemacreg1FR, ax=axs[2])

#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig
sns.regplot(x='r', y='savingrate', data=examplereemacreg1DK)
  

#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig
sns.boxplot(x='country',y='savingrate', data=examplereemacreg1, color='white')

ax = plt.gca()
ax.set_ylabel(" Log Saving Rate",fontsize=16)
ax.set_xlabel(" Country ",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
ax.legend()
plt.legend(fontsize=26)         
plt.savefig('logsavingratecountry.png', bbox_inches='tight')



#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig
sns.regplot(x='r', y='savingrate', data=examplereemacreg1)
ax = plt.gca()
ax.set_ylabel(" Log Saving Rate",fontsize=16)
ax.set_xlabel(" Log Interest Rate",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
ax.legend()
plt.legend(fontsize=26)         

plt.savefig('logsavingrateinterestrate.png', bbox_inches='tight')



#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig
sns.regplot(x='returnonequity', y='savingrate', data=examplereemacreg1)

ax = plt.gca()
ax.set_ylabel(" Log Saving Rate",fontsize=16)
ax.set_xlabel(" Log Return on Equity",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
ax.legend()
plt.legend(fontsize=26)         

plt.savefig('logsavingratereturnonequity.png', bbox_inches='tight')



#%% 

emacreg1  = emacreg1.set_index(['country','time'])
#%% savingrate,time,country,averagewage,ly,lm,de_0_14,de_15_24,de_25_49,de_50_64,de_65,r,HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,HSLDSNPISHBALNETSAVINGRATEPERCENTOFNETDINADJ

exog_vars = ['de_0_14','de_15_24','mid_age','de_65']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok this
print ("OLS regression model for the association between demgraphic factors and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1redemal = consTC.fit()
print(reg1redemal)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2redemal = cons.fit()
print(reg2redemal)
#%% ok this
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3redemal = cons.fit()
print(reg3redemal)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4redemal = consRE.fit()
print(reg4redemal)

#%%

import wooldridge as woo
import numpy as np
import linearmodels as plm
import scipy.stats as stats
#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# WE REJECT H0

#%%




#%%
#from statsmodels.iolib.summary2 import summary_col

from statsmodels.iolib.summary3 import summary_col
#from summary2 import summary_col 
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1demograph = summary_col([reg1redemal,reg2redemal,reg3redemal,reg4redemal],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1demographall1.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1demograph.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%  TO DO: ESTIMATE ONE BY ONE W AND WO FE



exog_vars = ['de_0_14']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1redem14 = consTC.fit()
print(reg1redem14)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2redem14 = cons.fit()
print(reg2redem14)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3redem14 = cons.fit()
print(reg3redem14)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4redem14 = consRE.fit()
print(reg4redem14)



#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1demograph = summary_col([reg1redem14,reg2redem14,reg3redem14,reg4redem14],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1demograp0141.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1demograph.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%% ,'de_15_24','de_25_49','de_50_64','de_65'


exog_vars = ['de_15_24']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1redem24 = consTC.fit()
print(reg1redem24)
#%%
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2redem24 = cons.fit()
print(reg2redem24)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3redem24 = cons.fit()
print(reg3redem24)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4redem24 = consRE.fit()
print(reg4redem24)



#%%

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1demograph = summary_col([reg1redem24,reg2redem24,reg3redem24,reg4redem24],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1demograp1524.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1demograph.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%    

exog_vars = ['mid_age']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1redem49 = consTC.fit()
print(reg1redem49)
#%%
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2redem49 = cons.fit()
print(reg2redem49)
#%%
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3redem49 = cons.fit()
print(reg3redem49)
#%%
print ("OLS regression model for the association between demgraphic factors and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4redem49 = consRE.fit()
print(reg4redem49)



#%%

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1demograph = summary_col([reg1redem49,reg2redem49,reg3redem49,reg4redem49],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1demograp2549.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1demograph.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%

exog_vars = ['de_50_64']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1redem64 = consTC.fit()
print(reg1redem64)
#%%
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2redem64 = cons.fit()
print(reg2redem64)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3redem64 = cons.fit()
print(reg3redem64)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4redem64 = consRE.fit()
print(reg4redem64)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%
from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1demograph = summary_col([reg1redem64,reg2redem64,reg3redem64,reg4redem64],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1demograp5064.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1demograph.as_latex())
dfoutput.write(endtex)
dfoutput.close()
#%%  

exog_vars = ['de_65']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok 
print ("OLS regression model for the association between demgraphic factors and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1redem65 = consTC.fit()
print(reg1redem65)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2redem65 = cons.fit()
print(reg2redem65)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3redem65 = cons.fit()
print(reg3redem65)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4redem65 = consRE.fit()
print(reg4redem65)



#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%
from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1demograph = summary_col([reg1redem65,reg2redem65,reg3redem65,reg4redem65],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1demograp65.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1demograph.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%% table 1 in template reg1redem14 reg1redem24  reg1redem49  reg1redem65 
from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1demograph = summary_col([reg1redem14,reg1redem24,reg1redem49,reg1redem65],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1demographyfortemplate.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1demograph.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%  
exog_vars = ['averagewage']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1reavg = consTC.fit()
print(reg1reavg)
#%%
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2reavg  = cons.fit()
print(reg2reavg)
#%% ok +
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3reavg = cons.fit()
print(reg3reavg)
#%% ok +
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4reavg = consRE.fit()
print(reg4reavg)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 
#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1wage = summary_col([reg1reavg,reg2reavg,reg3reavg,reg4reavg],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1wagerate.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1wage.as_latex())
dfoutput.write(endtex)
dfoutput.close()
#%%  
exog_vars = ['ly']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between hours worked by young individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1rely = consTC.fit()
print(reg1rely)
#%% ok + 
print ("OLS regression model for the association between hours worked by young individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2rely = cons.fit()
print(reg2rely)
#%% ok +
print ("OLS regression model for the association between hours worked by young individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3rely = cons.fit()
print(reg3rely)
#%% ok + 
print ("OLS regression model for the association between hours worked by young individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4rely = consRE.fit()
print(reg4rely)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 
#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1ly = summary_col([reg1rely,reg2rely,reg3rely,reg4rely],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1ly.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1ly.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%  
exog_vars = ['lm']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1relm = consTC.fit()
print(reg1relm)
#%% ok + 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2relm = cons.fit()
print(reg2relm)
#%%
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3relm = cons.fit()
print(reg3relm)
#%% ok + 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4relm = consRE.fit()
print(reg4relm)
#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%


from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


olslm = summary_col([reg1relm,reg2relm,reg3relm,reg4relm],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1lm.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(olslm.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%
#emacreg1r = emacreg1[~emacreg1.isin([np.nan, np.inf, -np.inf]).any(1)]


emacreg1 = emacreg1[~emacreg1.isin([np.inf, -np.inf]).any(1)]

#emacreg1.dtypes 
#%%
#mergeddebtwealth1["Year"] = mergeddebtwealth1.Year.astype(float)

emacreg1['rnew'] =  emacreg1['r']
#%%
emacreg1['rnew'] = 1 + emacreg1['r']
#%%
exog_vars = ['rnew']
exog = sm.add_constant(emacreg1[exog_vars])
#%%
print ("OLS regression model for the association between interest rate and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1rernew = consTC.fit()
print(reg1rernew)
#%%
print ("OLS regression model for the association between interest rate and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2rernew = cons.fit()
print(reg2rernew)
#%%
print ("OLS regression model for the association between interest rate and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3rernew = cons.fit()
print(reg3rernew)
#%%
print ("OLS regression model for the association between interest rate and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4rernew = consRE.fit()
print(reg4rernew)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1interest = summary_col([reg1rernew,reg2rernew,reg3rernew,reg4rernew],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("ols1interestrate.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1interest.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%


#%% 



exog_vars = ['ly','averagewage','rnew']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ for ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun1 = consTC.fit()
print(reg1refun1)
#%% ok+ for ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun1 = cons.fit()
print(reg2refun1)
#%% ok+ for ly and averagewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun1 = cons.fit()
print(reg3refun1)
#%%  ok+ for ly  and ok r 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun1 = consRE.fit()
print(reg4refun1)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun1 = summary_col([reg1refun1,reg2refun1,reg3refun1,reg4refun1],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun1.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun1.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%


exog_vars = ['ly','rnew']
exog = sm.add_constant(emacreg1[exog_vars])


#%% ok+ for ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun2 = consTC.fit()
print(reg1refun2)
#%% ok+ for ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun2 = cons.fit()
print(reg2refun2)
#%% ok+ for ly and ok for r 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun2 = cons.fit()
print(reg3refun2)
#%% ok+ for ly and ok for r 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun2 = consRE.fit()
print(reg4refun2)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun2= summary_col([reg1refun2,reg2refun2,reg3refun2,reg4refun2],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun2.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun2.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%


exog_vars = ['averagewage','rnew']
exog = sm.add_constant(emacreg1[exog_vars])


#%% ok+ for qverwqge 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun3 = consTC.fit()
print(reg1refun3)
#%%
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun3 = cons.fit()
print(reg2refun3)
#%%ok+ for qverwqge 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun3 = cons.fit()
print(reg3refun3)
#%%
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun3 = consRE.fit()
print(reg4refun3)



#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun3= summary_col([reg1refun3,reg2refun3,reg3refun3,reg4refun3],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun3.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun3.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%


exog_vars = ['lm','rnew']
exog = sm.add_constant(emacreg1[exog_vars])


#%% ok for lm
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun4 = consTC.fit()
print(reg1refun4)
#%% ok+ for lm
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun4 = cons.fit()
print(reg2refun4)
#%%
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun4 = cons.fit()
print(reg3refun4)
#%% ok+ for lm
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun4 = consRE.fit()
print(reg4refun4)

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun4= summary_col([reg1refun4,reg2refun4,reg3refun4,reg4refun4],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun4.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun4.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%% 'de_25_49','de_50_64'



exog_vars = ['ly','averagewage','rnew','de_0_14','de_15_24','mid_age','de_65','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun5 = consTC.fit()
print(reg1refun5)
#%%  ok+ ly and ok+avergaewage
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun5 = cons.fit()
print(reg2refun5)
#%% ok+ ly and ok+avergaewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun5 = cons.fit()
print(reg3refun5)
#%% ok+ ly and ok+avergaewage
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun5 = consRE.fit()
print(reg4refun5)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun5 = summary_col([reg1refun5,reg2refun5,reg3refun5,reg4refun5],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun5.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun5.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%% ,'de_15_24','de_25_49','de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_0_14','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly and ok+ r 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun6 = consTC.fit()
print(reg1refun6)
#%% ok+ ly and ok+avergaewage  ok for dem
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun6 = cons.fit()
print(reg2refun6)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun6 = cons.fit()
print(reg3refun6) 
#%%  ok+ ly and ok dem
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun6 = consRE.fit()
print(reg4refun6)

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 
#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun6 = summary_col([reg1refun6,reg2refun6,reg3refun6,reg4refun6],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun6.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun6.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%% ,,'de_25_49','de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_15_24','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun7 = consTC.fit()
print(reg1refun7)
#%%  ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun7 = cons.fit()
print(reg2refun7)
#%%  ok+ ly ok + for averagewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun7 = cons.fit()
print(reg3refun7)
#%%ok+ ly 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun7 = consRE.fit()
print(reg4refun7)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun7 = summary_col([reg1refun7,reg2refun7,reg3refun7,reg4refun7],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun7.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun7.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%% ,,','de_50_64'','de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_25_49','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun8 = consTC.fit()
print(reg1refun8)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun8 = cons.fit()
print(reg2refun8)
#%% ok+ ly ok + for averagewage 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun8 = cons.fit()
print(reg3refun8)
#%% ok+ ly ok for r
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun8 = consRE.fit()
print(reg4refun8)

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 
#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun8 = summary_col([reg1refun8,reg2refun8,reg3refun8,reg4refun8],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun8.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun8.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%% ,,',,'de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','mid_age','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun9 = consTC.fit()
print(reg1refun9)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun9 = cons.fit()
print(reg2refun9)
#%% ok+ ly  ok+ averagewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun9 = cons.fit()
print(reg3refun9)
#%%  ok+ ly ok for r 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun9 = consRE.fit()
print(reg4refun9)

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun9 = summary_col([reg1refun9,reg2refun9,reg3refun9,reg4refun9],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun9.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun9.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%% ,,',,'de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','mid_age','de_65','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly  
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun10 = consTC.fit()
print(reg1refun10)
#%% ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun10 = cons.fit()
print(reg2refun10)
#%%  ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun10 = cons.fit()
print(reg3refun10)
#%%  ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun10 = consRE.fit()
print(reg4refun10)

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun10 = summary_col([reg1refun10,reg2refun10,reg3refun10,reg4refun10],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun10.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun10.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%% ,,',,'de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_65','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1refun11 = consTC.fit()
print(reg1refun11)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2refun11 = cons.fit()
print(reg2refun11)
#%% ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3refun11 = cons.fit()
print(reg3refun11)
#%% ok+ ly  
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4refun11 = consRE.fit()
print(reg4refun11)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 


#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfun11 = summary_col([reg1refun11,reg2refun11,reg3refun11,reg4refun11],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfun11.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfun11.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%



from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfunaall = summary_col([reg4refun1,reg3refun2,reg4refun2,reg4refun8,reg4refun9],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsols1savingfunallforr.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfunaall.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%

exog_vars = ['ly','averagewage','rnew','capitalconv','returnonequity']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly
print ("OLS regression model for the association between capita roe and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1novcap1 = consTC.fit()
print(reg1novcap1)
#%% ok+ ly 
print ("OLS regression model for the association between capita roe and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2novcap1 = cons.fit()
print(reg2novcap1)
#%% ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between capita roe and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3novcap1 = cons.fit()
print(reg3novcap1)
#%% ok+ ly  
print ("OLS regression model for the association between capita roe and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4novcap1 = consRE.fit()
print(reg4novcap1)

#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 
#%%


from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfunaall = summary_col([reg1novcap1,reg2novcap1,reg3novcap1,reg4novcap1],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsavingfunallforrcapital1.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfunaall.as_latex())
dfoutput.write(endtex)
dfoutput.close()




#%%
exog_vars = ['returnonequity']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly
print ("OLS regression model for the association between capita roe and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1novcap1roe = consTC.fit()
print(reg1novcap1roe)
#%% ok+ ly 
print ("OLS regression model for the association between capita roe and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2novcap1roe = cons.fit()
print(reg2novcap1roe)
#%% ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between capita roe and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3novcap1roe = cons.fit()
print(reg3novcap1roe)
#%% ok+ ly  
print ("OLS regression model for the association between capita roe and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4novcap1roe = consRE.fit()
print(reg4novcap1roe)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 
#%%


from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfunaall = summary_col([reg1novcap1roe,reg2novcap1roe,reg3novcap1roe,reg4novcap1roe],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsavingfunallforrcapital1roe.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfunaall.as_latex())
dfoutput.write(endtex)
dfoutput.close()




#%%

exog_vars = ['capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly
print ("OLS regression model for the association between capita roe and saving ")
consTC = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=True)
reg1novcap1k = consTC.fit()
print(reg1novcap1k)
#%% ok+ ly 
print ("OLS regression model for the association between capita roe and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=True)
reg2novcap1k = cons.fit()
print(reg2novcap1k)
#%% ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between capita roe and saving ")
cons = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=True,time_effects=False)
reg3novcap1k = cons.fit()
print(reg3novcap1k)
#%% ok+ ly  
print ("OLS regression model for the association between capita roe and saving ")
consRE = PanelOLS(emacreg1.HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,exog,entity_effects=False,time_effects=False)
reg4novcap1k = consRE.fit()
print(reg4novcap1k)


#%% # h0= random is prefered

results_fe = consTC.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov


results_re = consRE.fit()
b_re = results_re.params
b_re_cov = results_re.cov
#%%
# Hausman test of FE vs. RE
# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)

# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
# 

#%%
from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfunaall = summary_col([reg1novcap1k,reg2novcap1k,reg3novcap1k,reg4novcap1k],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsavingfunallforrcapital1k.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfunaall.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%
#%% SAVING RATE

#%% savingrate,time,country,averagewage,ly,lm,de_0_14,de_15_24,de_25_49,de_50_64,de_65,r,HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN,HSLDSNPISHBALNETSAVINGRATEPERCENTOFNETDINADJ

exog_vars = ['de_0_14','de_15_24','de_25_49','de_50_64','de_65']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok this
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1redemals = cons.fit()
print(reg1redemals)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2redemals = cons.fit()
print(reg2redemals)
#%% ok this
print ("OLS regression model for the association between demgraphic factors and saving rate ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3redemals = cons.fit()
print(reg3redemals)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4redemals = cons.fit()
print(reg4redemals)



#%%
exog_vars = ['de_0_14']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1redem14s = cons.fit()
print(reg1redem14s)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2redem14s = cons.fit()
print(reg2redem14s)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3redem14s = cons.fit()
print(reg3redem14s)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4redem14s = cons.fit()
print(reg4redem14s)

#%%
exog_vars = ['de_15_24']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1redem24s = cons.fit()
print(reg1redem24s)
#%%
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2redem24s = cons.fit()
print(reg2redem24s)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3redem24s = cons.fit()
print(reg3redem24s)
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4redem24s = cons.fit()
print(reg4redem24s)


#%%


exog_vars = ['de_25_49']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1redem49s = cons.fit()
print(reg1redem49s)
#%%
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2redem49s = cons.fit()
print(reg2redem49s)
#%%
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3redem49s = cons.fit()
print(reg3redem49s)
#%%
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4redem49s = cons.fit()
print(reg4redem49s)

#%%


#%%

exog_vars = ['de_50_64']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1redem64s = cons.fit()
print(reg1redem64s)
#%%
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2redem64s = cons.fit()
print(reg2redem64s)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3redem64s = cons.fit()
print(reg3redem64s)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4redem64s = cons.fit()
print(reg4redem64s)

#%%

exog_vars = ['de_65']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok 
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1redem65s = cons.fit()
print(reg1redem65s)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2redem65s = cons.fit()
print(reg2redem65s)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3redem65s = cons.fit()
print(reg3redem65s)
#%% ok +
print ("OLS regression model for the association between demgraphic factors and saving rate")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4redem65s = cons.fit()
print(reg4redem65s)

#%% 





#%%
exog_vars = ['rnew']
exog = sm.add_constant(emacreg1[exog_vars])
#%%
print ("OLS regression model for the association between interest rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1rernews= cons.fit()
print(reg1rernews)
#%%
print ("OLS regression model for the association between interest rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2rernews= cons.fit()
print(reg2rernews)
#%%
print ("OLS regression model for the association between interest rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3rernews= cons.fit()
print(reg3rernews)
#%%
print ("OLS regression model for the association between interest rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4rernews= cons.fit()
print(reg4rernews)


#%%

exog_vars = ['ly','averagewage','rnew','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ for ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving rate ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun1s= cons.fit()
print(reg1refun1s)
#%% ok+ for ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving rate ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun1s= cons.fit()
print(reg2refun1s)
#%% ok+ for ly and averagewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving rate ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun1s= cons.fit()
print(reg3refun1s)
#%%  ok+ for ly  and ok r 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving rate ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun1s= cons.fit()
print(reg4refun1s)


#%%

#%%  
exog_vars = ['averagewage']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1reavgs= cons.fit()
print(reg1reavgs)
#%%
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2reavgs = cons.fit()
print(reg2reavgs)
#%% ok +
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3reavgs= cons.fit()
print(reg3reavgs)
#%% ok +
print ("OLS regression model for the association between AVERAGE WAGE rate and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4reavgs= cons.fit()
print(reg4reavgs)

#%%  
exog_vars = ['ly']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok +
print ("OLS regression model for the association between hours worked by young individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1relys= cons.fit()
print(reg1relys)
#%% ok + 
print ("OLS regression model for the association between hours worked by young individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2relys= cons.fit()
print(reg2relys)
#%% ok +
print ("OLS regression model for the association between hours worked by young individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3relys= cons.fit()
print(reg3relys)
#%% ok + 
print ("OLS regression model for the association between hours worked by young individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4relys= cons.fit()
print(reg4relys)


#%%


exog_vars = ['ly','rnew','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])


#%% ok+ for ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun2s= cons.fit()
print(reg1refun2s)
#%% ok+ for ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun2s= cons.fit()
print(reg2refun2s)
#%% ok+ for ly and ok for r 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun2s= cons.fit()
print(reg3refun2s)
#%% ok+ for ly and ok for r 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun2s= cons.fit()
print(reg4refun2s)


#%%

exog_vars = ['averagewage','rnew','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])


#%% ok+ for qverwqge 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun3s= cons.fit()
print(reg1refun3s)
#%%
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun3s= cons.fit()
print(reg2refun3s)
#%%ok+ for qverwqge 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun3s= cons.fit()
print(reg3refun3s)
#%%
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun3s= cons.fit()
print(reg4refun3s)




#%%

#%%



exog_vars = ['ly','averagewage','rnew','de_0_14','de_15_24','mid_age','de_65','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun5s= cons.fit()
print(reg1refun5s)
#%%  ok+ ly and ok+avergaewage
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun5s= cons.fit()
print(reg2refun5s)
#%% ok+ ly and ok+avergaewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun5s= cons.fit()
print(reg3refun5s)
#%% ok+ ly and ok+avergaewage
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun5s= cons.fit()
print(reg4refun5s)


#%%

#%% ,'de_15_24','de_25_49','de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_0_14','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly and ok+ r 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun6s= cons.fit()
print(reg1refun6s)
#%% ok+ ly and ok+avergaewage  ok for dem
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun6s= cons.fit()
print(reg2refun6s)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun6s= cons.fit()
print(reg3refun6s)
#%%  ok+ ly and ok dem
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun6s= cons.fit()
print(reg4refun6s)


#%%


#%% ,,'de_25_49','de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_15_24','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun7s= cons.fit()
print(reg1refun7s)
#%%  ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun7s= cons.fit()
print(reg2refun7s)
#%%  ok+ ly ok + for averagewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun7s= cons.fit()
print(reg3refun7s)
#%%ok+ ly 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun7s= cons.fit()
print(reg4refun7s)


#%%


#%% ,,','de_50_64'','de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_25_49','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun8s= cons.fit()
print(reg1refun8s)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun8s= cons.fit()
print(reg2refun8s)
#%% ok+ ly ok + for averagewage 
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun8s= cons.fit()
print(reg3refun8s)
#%% ok+ ly ok for r
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun8s= cons.fit()
print(reg4refun8s)





#%% ,,',,'de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','mid_age','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun9s= cons.fit()
print(reg1refun9s)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun9s= cons.fit()
print(reg2refun9s)
#%% ok+ ly  ok+ averagewage
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun9s= cons.fit()
print(reg3refun9s)
#%%  ok+ ly ok for r 
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun9s= cons.fit()
print(reg4refun9s)


#%%


#%% ,,',,'de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','mid_age','de_65','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly  
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun10s= cons.fit()
print(reg1refun10s)
#%% ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun10s= cons.fit()
print(reg2refun10s)
#%%  ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun10s= cons.fit()
print(reg3refun10s)
#%%  ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun10s= cons.fit()
print(reg4refun10s)




#%% ,,',,'de_50_64','de_65'

exog_vars = ['ly','averagewage','rnew','de_65','returnonequity','capitalconv']
exog = sm.add_constant(emacreg1[exog_vars])
#%%  ok+ ly
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=True)
reg1refun11s= cons.fit()
print(reg1refun11s)
#%% ok+ ly 
print ("OLS regression model for the association between hours worked by middle age individuals and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=True)
reg2refun11s= cons.fit()
print(reg2refun11s)
#%% ok+ ly  ok+ for wagerate
print ("OLS regression model for the association between hours worked by middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=True,time_effects=False)
reg3refun11s= cons.fit()
print(reg3refun11s)
#%% ok+ ly  
print ("OLS regression model for the association between hours worked by  middle age individuals  and saving ")
cons = PanelOLS(emacreg1.savingrate,exog,entity_effects=False,time_effects=False)
reg4refun11s= cons.fit()
print(reg4refun11s)


#%%









#%%

from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols18888 = summary_col([reg1redemals, reg1redem14s,reg1redem24s,reg1redem49s, reg1redem64s,reg1redem65s,reg1rernews,reg2rernews,reg3rernews,reg1refun1s,reg1reavgs, reg4reavgs,reg1relys,reg1refun2s,reg1refun5s],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olsosavingraterdemo1.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols18888.as_latex())
dfoutput.write(endtex)
dfoutput.close()




#%%
import pandas as pd

#%%

#%%

#%%

#examplereemacreg1DK = emacreg1.loc[emacreg1['country']]

#%%



#%%






#%%
#import numpy as np
#import pandas as pd
#from scipy import stats



#%% Perform Hausman-Test







#%%

import statsmodels.formula.api as smf
#%% 
emacreg1capitalscatter = macroregdata[macroregdata['HSLDSNPISHBALNETSAVINGOFHOUSEHOLDSEURCURN'].notna()]

#%%
emacreg1capitalscatter = emacreg1capitalscatter[emacreg1capitalscatter['lagedinteresrate'].notna()]

#%%

emacreg1capitalscatter = emacreg1capitalscatter[emacreg1capitalscatter['output'].notna()]

#%% estimate rho policy rule


exog_vars = ['lagedinteresrate','output']
exog = emacreg1capitalscatter[exog_vars]
#%%

#%%
print("regression for policy rule ")
withoutconstantTC = sm.OLS(endog=emacreg1capitalscatter.r, exog=emacreg1capitalscatter[exog_vars],missing='drop').fit()

#%%
print(withoutconstantTC.summary())



#%% to updaate table     
# reg4rernew,reg1refun1,reg1refun2,reg4refun3,reg4refun5,
# reg1refun6,reg1refun7,reg1refun8,reg1refun9,reg1refun10,reg1refun11,
# reg1novcap1,reg1novcap1roe,reg1novcap1k






from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfunaall = summary_col([reg4rernew,reg1refun1,reg1refun2,reg4refun3,reg4refun5],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olstemplate1.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfunaall.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%



from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfunaall = summary_col([reg1refun6,reg1refun7,reg1refun9,reg1refun10,reg1refun11],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olstemplate2.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfunaall.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%


from statsmodels.iolib.summary3 import summary_col

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


ols1savingfunaall = summary_col([reg1novcap1,reg1novcap1roe,reg1novcap1k],stars=True,float_format='%0.2f',show='se') 


dfoutput = open("olstemplate3.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(ols1savingfunaall.as_latex())
dfoutput.write(endtex)
dfoutput.close()

