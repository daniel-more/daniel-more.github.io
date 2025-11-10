from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import Athena, Glue
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Sagemaker, SagemakerNotebook
from diagrams.aws.storage import S3
from diagrams.generic.compute import Rack

# with Diagram("MARCO Crystallization ML Pipeline", show=False, direction="LR"):
with Diagram(
    "MARCO Crystallization ML Pipeline",
    show=False,
    direction="LR",
    graph_attr={
        "size": "28,28!",  # width,height — increase for larger diagrams
        "dpi": "300",  # higher resolution
    },
):
    # Storage
    s3_raw = S3("S3\nRaw MARCO Images")
    s3_delta = S3("S3\nDelta/Parquet\nTrain/Test")
    s3_model = S3("S3\nModel Artifacts")

    # Catalog + Query
    glue = Glue("Glue Catalog")
    athena = Athena("Athena SQL")

    # Training Environment
    with Cluster("SageMaker Training Environment"):
        notebook = SagemakerNotebook("Notebook\n(PyTorch + PyAthena)")
        training = Rack("PyTorch\nResNet50")
        sm_job = Sagemaker("SageMaker\nTraining Job")

    # Logging
    logs = Cloudwatch("CloudWatch Logs")

    # Data Flow
    s3_raw >> Edge(label="ETL / Convert") >> s3_delta
    s3_delta >> glue >> athena >> notebook
    notebook >> training
    training >> sm_job >> s3_model
    training >> logs
