# AUEB M.Sc. in Data Science (part-time)
## Course: Practical Data Science
## Homework: 3
## Author: Spiros Politis

----------

**Disclaimer**: the work presented here was done in the context of a data science course and is provided solely for demonstration purposes. The author does not claim that it is necessarily complete and / or correct in any way. Its purpose is to provide a guide for other students that may have to tackle problems similar to the ones mentioned in the homework.

----------

## 1. Env setup

### 1.1. Create Conda virtual env

```
conda create -n msc-ds-pds-homework-3 python=3.6
source activate msc-ds-pds-homework-3
```

###  1.2. Install required packages
```
pip install -r requirements.txt
```

## 2. Running the web service

CD to the directory that contains load_forecasting_web.py

Then:

### 2.1. On Linux

In a terminal, type the following:

```
export FLASK_APP=load_forecasting_web
export FLASK_DEBUG=1
export FLASK_RUN_PORT=9181
flask run
```

### 2.2. On Windows

In the command line, type the following:

```
set FLASK_APP=load_forecasting_web
set FLASK_DEBUG=1
set FLASK_RUN_PORT=9181
flask run
```

## 3. Testing the web service

The values of the test paramaters provided below come from the first row of the pre-processed (including engineered features), data set the was created up to and including par. 1.4 of homework 2. The test case was generated as below:

```
energy_df[[
    "appliances", "hour", "press_mm_hg", "lights", "windspeed", "rh_in_mean", "day_of_month", "tdewpoint", "day_of_week"
]].head(1)
```

### 3.1. Testing with CURL

```
curl -i -H "Accept: application/json" -H "Content-Type: application/json" -X GET "http://127.0.0.1:9181/forecast?day_of_month=1&tdewpoint=10.291667&day_of_week=0&hour=0&lights=13.333333&press_mm_hg=755.833333&windspeed=8.416667&rh_in_mean=49.983941"
```

### 3.2. Testing with a browser

Point your broswer at:

http://127.0.0.1:9181/forecast?day_of_month=1&tdewpoint=10.291667&day_of_week=0&hour=0&lights=13.333333&press_mm_hg=755.833333&windspeed=8.416667&rh_in_mean=49.983941