import boto3

athena = boto3.client("athena")
glue = boto3.client("glue")
sagemaker = boto3.client("sagemaker")


def get_all_model_package_groups():
    group_names = []
    next_token = None

    while True:
        params = {"MaxResults": 100}
        if next_token:
            params["NextToken"] = next_token

        response = sagemaker.list_model_package_groups(**params)
        group_names.extend([g["ModelPackageGroupName"] for g in response["ModelPackageGroupSummaryList"]])

        next_token = response.get("NextToken")
        if not next_token:
            break

    return group_names


model_name = "aqsol-uq"

# Get the model package group ARN
group = sagemaker.describe_model_package_group(ModelPackageGroupName=model_name)
group_arn = group["ModelPackageGroupArn"]

# Get tags from the group
tags = sagemaker.list_tags(ResourceArn=group_arn)["Tags"]

# Get the FeatureSet (workbench_input)
fg_name = next(tag["Value"] for tag in tags if tag["Key"] == "workbench_input")

# Get the feature group ARN
fg = sagemaker.describe_feature_group(FeatureGroupName=fg_name)
fg_arn = fg["FeatureGroupArn"]

# Get tags
tags = sagemaker.list_tags(ResourceArn=fg_arn)["Tags"]

# Get the data source input
ds_name = next(tag["Value"] for tag in tags if tag["Key"] == "workbench_input")

table = glue.get_table(DatabaseName="workbench", Name=ds_name)
update_time = table["Table"]["UpdateTime"]
print(update_time)
print(type(update_time))
