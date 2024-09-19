num_epochs = 2
train_dev_test_split = (0.8, 0.1, 0.1)
target_column = "loan_status"
selected_columns = [
    "issue_d",
    "loan_status",
    "loan_amnt",
    "term",
    "int_rate",
    "purpose",
    "fico_range_low",
    "grade",
    "inq_last_6mths",
    "revol_util",
    "delinq_2yrs",
    "pub_rec",
    "open_acc",
    "total_acc",
    "annual_inc",
    "emp_length",
    "home_ownership",
    "verification_status",
    "dti",
    "desc",
    "earliest_cr_line",
    "total_rev_hi_lim",
]
categorical_features = ['term','purpose', 'home_ownership', 'verification_status']
