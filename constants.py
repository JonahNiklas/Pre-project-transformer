from transformer_config import TransformerConfig

year_range = (2007, 2014)
num_epochs = 1
train_dev_test_split = (0.8, 0.1, 0.1)
target_column = "loan_status"
embedding_dimension = 50
over_sampling_ratio = 0.5
transformer_config = TransformerConfig.model_validate(
    {
        "d_ff": 50,
        "max_seq_length": 75,
        "dropout": 0.1,
        "num_heads": 5, # TODO: should be 8, that is what the paper uses
        "activation": "relu",
        "num_layers": 1,
    }
)
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
categorical_features = ["term", "purpose", "home_ownership", "verification_status"]
