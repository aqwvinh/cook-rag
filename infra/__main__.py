import glob
import os

import pulumi
import pulumi_gcp as gcp

# Get pulumi config
config = pulumi.Config()
bucket_name = config.get("bucketName")
region = gcp.config.region

bucket = gcp.storage.Bucket(
    "cook-rag",
    name=bucket_name,
    location=region,
    uniform_bucket_level_access=True,
    force_destroy=True,  # in case we need to delete the bucket if budget alert is triggered
    lifecycle_rules=[
        {
            "action": {"type": "Delete"},
            "condition": {"age": 365},
        }
    ],
)

# Upload PDFs to the bucket
pdf_dir = os.path.join(os.path.dirname(__file__), "..", "data")
for path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
    gcp.storage.BucketObject(
        os.path.basename(path),
        bucket=bucket.name,
        name=os.path.basename(path),
        source=pulumi.FileAsset(path),
        content_type="application/pdf",
    )

# Export bucket URL
pulumi.export("bucket_url", bucket.url)
