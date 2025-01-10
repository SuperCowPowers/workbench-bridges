import boto3

# Debug: Check role or user info
sts_client = boto3.client("sts")
caller_identity = sts_client.get_caller_identity()
print("Caller Identity:", caller_identity)
